static char help[] = "Second Order TVD Finite Volume Example.\n";
#include <petscdmplex.h>
#include <petscdmforest.h>
#include <petscds.h>
#include <petscts.h>
#include <petscsf.h> /* For SplitFaces() */

#define DIM 2                   /* Geometric dimension */
#define ALEN(a) (sizeof(a)/sizeof((a)[0]))

static PetscFunctionList PhysicsList;

/* Represents continuum physical equations. */
typedef struct _n_Physics *Physics;

/* Physical model includes boundary conditions, initial conditions, and functionals of interest. It is
 * discretization-independent, but its members depend on the scenario being solved. */
typedef struct _n_Model *Model;

/* 'User' implements a discretization of a continuous model. */
typedef struct _n_User *User;
typedef PetscErrorCode (*SolutionFunction)(Model,PetscReal,const PetscReal*,PetscScalar*,void*);
typedef PetscErrorCode (*SetUpBCFunction)(PetscDS,Physics);
typedef PetscErrorCode (*FunctionalFunction)(Model,PetscReal,const PetscReal*,const PetscScalar*,PetscReal*,void*);
typedef PetscErrorCode (*SetupFields)(Physics,PetscSection);
static PetscErrorCode ModelSolutionSetDefault(Model,SolutionFunction,void*);
static PetscErrorCode ModelFunctionalRegister(Model,const char*,PetscInt*,FunctionalFunction,void*);
static PetscErrorCode OutputVTK(DM,const char*,PetscViewer*);

struct FieldDescription {
  const char *name;
  PetscInt dof;
};

typedef struct _n_FunctionalLink *FunctionalLink;
struct _n_FunctionalLink {
  char               *name;
  FunctionalFunction func;
  void               *ctx;
  PetscInt           offset;
  FunctionalLink     next;
};

struct _n_Physics {
  PetscRiemannFunc riemann;
  PetscInt         dof;          /* number of degrees of freedom per cell */
  PetscReal        maxspeed;     /* kludge to pick initial time step, need to add monitoring and step control */
  void             *data;
  PetscInt         nfields;
  const struct FieldDescription *field_desc;
};

struct _n_Model {
  MPI_Comm         comm;        /* Does not do collective communicaton, but some error conditions can be collective */
  Physics          physics;
  FunctionalLink   functionalRegistry;
  PetscInt         maxComputed;
  PetscInt         numMonitored;
  FunctionalLink   *functionalMonitored;
  PetscInt         numCall;
  FunctionalLink   *functionalCall;
  SolutionFunction solution;
  SetUpBCFunction  setupbc;
  void             *solutionctx;
  PetscReal        maxspeed;    /* estimate of global maximum speed (for CFL calculation) */
  PetscReal        bounds[2*DIM];
  DMBoundaryType   bcs[3];
  PetscErrorCode   (*errorIndicator)(PetscInt, PetscReal, PetscInt, const PetscScalar[], const PetscScalar[], PetscReal *, void *);
  void             *errorCtx;
};

struct _n_User {
  PetscInt numSplitFaces;
  PetscInt vtkInterval;   /* For monitor */
  char outputBasename[PETSC_MAX_PATH_LEN]; /* Basename for output files */
  PetscInt monitorStepOffset;
  Model    model;
  PetscBool vtkmon;
  Vec grad;
};

PETSC_STATIC_INLINE PetscReal DotDIMReal(const PetscReal *x,const PetscReal *y)
{
  PetscInt  i;
  PetscReal prod=0.0;

  for (i=0; i<DIM; i++) prod += x[i]*y[i];
  return prod;
}
PETSC_STATIC_INLINE PetscReal NormDIM(const PetscReal *x) { return PetscSqrtReal(PetscAbsReal(DotDIMReal(x,x))); }

PETSC_STATIC_INLINE PetscReal Dot2Real(const PetscReal *x,const PetscReal *y) { return x[0]*y[0] + x[1]*y[1];}
PETSC_STATIC_INLINE PetscReal Norm2Real(const PetscReal *x) { return PetscSqrtReal(PetscAbsReal(Dot2Real(x,x)));}
PETSC_STATIC_INLINE void Normalize2Real(PetscReal *x) { PetscReal a = 1./Norm2Real(x); x[0] *= a; x[1] *= a; }
PETSC_STATIC_INLINE void Waxpy2Real(PetscReal a,const PetscReal *x,const PetscReal *y,PetscReal *w) { w[0] = a*x[0] + y[0]; w[1] = a*x[1] + y[1]; }
PETSC_STATIC_INLINE void Scale2Real(PetscReal a,const PetscReal *x,PetscReal *y) { y[0] = a*x[0]; y[1] = a*x[1]; }

/******************* Advect ********************/
typedef enum {ADVECT_SOL_TILTED,ADVECT_SOL_BUMP,ADVECT_SOL_BUMP_CAVITY} AdvectSolType;
static const char *const AdvectSolTypes[] = {"TILTED","BUMP","BUMP_CAVITY","AdvectSolType","ADVECT_SOL_",0};
typedef enum {ADVECT_SOL_BUMP_CONE,ADVECT_SOL_BUMP_COS} AdvectSolBumpType;
static const char *const AdvectSolBumpTypes[] = {"CONE","COS","AdvectSolBumpType","ADVECT_SOL_BUMP_",0};

typedef struct {
  PetscReal wind[DIM];
} Physics_Advect_Tilted;
typedef struct {
  PetscReal         center[DIM];
  PetscReal         radius;
  AdvectSolBumpType type;
} Physics_Advect_Bump;

typedef struct {
  PetscReal     inflowState;
  AdvectSolType soltype;
  union {
    Physics_Advect_Tilted tilted;
    Physics_Advect_Bump   bump;
  } sol;
  struct {
    PetscInt Solution;
    PetscInt Error;
  } functional;
} Physics_Advect;

static const struct FieldDescription PhysicsFields_Advect[] = {{"U",1},{NULL,0}};

static PetscErrorCode PhysicsBoundary_Advect_Inflow(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *xI, PetscScalar *xG, void *ctx)
{
  Physics        phys    = (Physics)ctx;
  Physics_Advect *advect = (Physics_Advect*)phys->data;

  PetscFunctionBeginUser;
  xG[0] = advect->inflowState;
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsBoundary_Advect_Outflow(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *xI, PetscScalar *xG, void *ctx)
{
  PetscFunctionBeginUser;
  xG[0] = xI[0];
  PetscFunctionReturn(0);
}

static void PhysicsRiemann_Advect(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n, const PetscScalar *xL, const PetscScalar *xR, PetscInt numConstants, const PetscScalar constants[], PetscScalar *flux, Physics phys)
{
  Physics_Advect *advect = (Physics_Advect*)phys->data;
  PetscReal      wind[DIM],wn;

  switch (advect->soltype) {
  case ADVECT_SOL_TILTED: {
    Physics_Advect_Tilted *tilted = &advect->sol.tilted;
    wind[0] = tilted->wind[0];
    wind[1] = tilted->wind[1];
  } break;
  case ADVECT_SOL_BUMP:
    wind[0] = -qp[1];
    wind[1] = qp[0];
    break;
  case ADVECT_SOL_BUMP_CAVITY:
    {
      PetscInt  i;
      PetscReal comp2[3] = {0.,0.,0.}, rad2;

      rad2 = 0.;
      for (i = 0; i < dim; i++) {
        comp2[i] = qp[i] * qp[i];
        rad2    += comp2[i];
      }

      wind[0] = -qp[1];
      wind[1] = qp[0];
      if (rad2 > 1.) {
        PetscInt  maxI = 0;
        PetscReal maxComp2 = comp2[0];

        for (i = 1; i < dim; i++) {
          if (comp2[i] > maxComp2) {
            maxI     = i;
            maxComp2 = comp2[i];
          }
        }
        wind[maxI] = 0.;
      }
    }
    break;
  default:
  {
    PetscInt i;
    for (i = 0; i < DIM; ++i) wind[i] = 0.0;
  }
  /* default: SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for solution type %s",AdvectSolBumpTypes[advect->soltype]); */
  }
  wn      = Dot2Real(wind, n);
  flux[0] = (wn > 0 ? xL[0] : xR[0]) * wn;
}

static PetscErrorCode PhysicsSolution_Advect(Model mod,PetscReal time,const PetscReal *x,PetscScalar *u,void *ctx)
{
  Physics        phys    = (Physics)ctx;
  Physics_Advect *advect = (Physics_Advect*)phys->data;

  PetscFunctionBeginUser;
  switch (advect->soltype) {
  case ADVECT_SOL_TILTED: {
    PetscReal             x0[DIM];
    Physics_Advect_Tilted *tilted = &advect->sol.tilted;
    Waxpy2Real(-time,tilted->wind,x,x0);
    if (x0[1] > 0) u[0] = 1.*x[0] + 3.*x[1];
    else u[0] = advect->inflowState;
  } break;
  case ADVECT_SOL_BUMP_CAVITY:
  case ADVECT_SOL_BUMP: {
    Physics_Advect_Bump *bump = &advect->sol.bump;
    PetscReal           x0[DIM],v[DIM],r,cost,sint;
    cost  = PetscCosReal(time);
    sint  = PetscSinReal(time);
    x0[0] = cost*x[0] + sint*x[1];
    x0[1] = -sint*x[0] + cost*x[1];
    Waxpy2Real(-1,bump->center,x0,v);
    r = Norm2Real(v);
    switch (bump->type) {
    case ADVECT_SOL_BUMP_CONE:
      u[0] = PetscMax(1 - r/bump->radius,0);
      break;
    case ADVECT_SOL_BUMP_COS:
      u[0] = 0.5 + 0.5*PetscCosReal(PetscMin(r/bump->radius,1)*PETSC_PI);
      break;
    }
  } break;
  default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown solution type");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsFunctional_Advect(Model mod,PetscReal time,const PetscReal *x,const PetscScalar *y,PetscReal *f,void *ctx)
{
  Physics        phys    = (Physics)ctx;
  Physics_Advect *advect = (Physics_Advect*)phys->data;
  PetscScalar    yexact[1];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PhysicsSolution_Advect(mod,time,x,yexact,phys);CHKERRQ(ierr);
  f[advect->functional.Solution] = PetscRealPart(y[0]);
  f[advect->functional.Error] = PetscAbsScalar(y[0]-yexact[0]);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetUpBC_Advect(PetscDS prob, Physics phys)
{
  PetscErrorCode ierr;
  const PetscInt inflowids[] = {100,200,300},outflowids[] = {101};

  PetscFunctionBeginUser;
  /* Register "canned" boundary conditions and defaults for where to apply. */
  ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "inflow",  "Face Sets", 0, 0, NULL, (void (*)(void)) PhysicsBoundary_Advect_Inflow,  ALEN(inflowids),  inflowids,  phys);CHKERRQ(ierr);
  ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "outflow", "Face Sets", 0, 0, NULL, (void (*)(void)) PhysicsBoundary_Advect_Outflow, ALEN(outflowids), outflowids, phys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCreate_Advect(Model mod,Physics phys,PetscOptionItems *PetscOptionsObject)
{
  Physics_Advect *advect;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  phys->field_desc = PhysicsFields_Advect;
  phys->riemann    = (PetscRiemannFunc)PhysicsRiemann_Advect;
  ierr = PetscNew(&advect);CHKERRQ(ierr);
  phys->data       = advect;
  mod->setupbc = SetUpBC_Advect;

  ierr = PetscOptionsHead(PetscOptionsObject,"Advect options");CHKERRQ(ierr);
  {
    PetscInt two = 2,dof = 1;
    advect->soltype = ADVECT_SOL_TILTED;
    ierr = PetscOptionsEnum("-advect_sol_type","solution type","",AdvectSolTypes,(PetscEnum)advect->soltype,(PetscEnum*)&advect->soltype,NULL);CHKERRQ(ierr);
    switch (advect->soltype) {
    case ADVECT_SOL_TILTED: {
      Physics_Advect_Tilted *tilted = &advect->sol.tilted;
      two = 2;
      tilted->wind[0] = 0.0;
      tilted->wind[1] = 1.0;
      ierr = PetscOptionsRealArray("-advect_tilted_wind","background wind vx,vy","",tilted->wind,&two,NULL);CHKERRQ(ierr);
      advect->inflowState = -2.0;
      ierr = PetscOptionsRealArray("-advect_tilted_inflow","Inflow state","",&advect->inflowState,&dof,NULL);CHKERRQ(ierr);
      phys->maxspeed = Norm2Real(tilted->wind);
    } break;
    case ADVECT_SOL_BUMP_CAVITY:
    case ADVECT_SOL_BUMP: {
      Physics_Advect_Bump *bump = &advect->sol.bump;
      two = 2;
      bump->center[0] = 2.;
      bump->center[1] = 0.;
      ierr = PetscOptionsRealArray("-advect_bump_center","location of center of bump x,y","",bump->center,&two,NULL);CHKERRQ(ierr);
      bump->radius = 0.9;
      ierr = PetscOptionsReal("-advect_bump_radius","radius of bump","",bump->radius,&bump->radius,NULL);CHKERRQ(ierr);
      bump->type = ADVECT_SOL_BUMP_CONE;
      ierr = PetscOptionsEnum("-advect_bump_type","type of bump","",AdvectSolBumpTypes,(PetscEnum)bump->type,(PetscEnum*)&bump->type,NULL);CHKERRQ(ierr);
      phys->maxspeed = 3.;       /* radius of mesh, kludge */
    } break;
    }
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  /* Initial/transient solution with default boundary conditions */
  ierr = ModelSolutionSetDefault(mod,PhysicsSolution_Advect,phys);CHKERRQ(ierr);
  /* Register "canned" functionals */
  ierr = ModelFunctionalRegister(mod,"Solution",&advect->functional.Solution,PhysicsFunctional_Advect,phys);CHKERRQ(ierr);
  ierr = ModelFunctionalRegister(mod,"Error",&advect->functional.Error,PhysicsFunctional_Advect,phys);CHKERRQ(ierr);
  mod->bcs[0] = mod->bcs[1] = mod->bcs[2] = DM_BOUNDARY_GHOSTED;
  PetscFunctionReturn(0);
}

/******************* Shallow Water ********************/
typedef struct {
  PetscReal gravity;
  PetscReal boundaryHeight;
  struct {
    PetscInt Height;
    PetscInt Speed;
    PetscInt Energy;
  } functional;
} Physics_SW;
typedef struct {
  PetscReal h;
  PetscReal uh[DIM];
} SWNode;
typedef union {
  SWNode    swnode;
  PetscReal vals[DIM+1];
} SWNodeUnion;

static const struct FieldDescription PhysicsFields_SW[] = {{"Height",1},{"Momentum",DIM},{NULL,0}};

/*
 * h_t + div(uh) = 0
 * (uh)_t + div (u\otimes uh + g h^2 / 2 I) = 0
 *
 * */
static PetscErrorCode SWFlux(Physics phys,const PetscReal *n,const SWNode *x,SWNode *f)
{
  Physics_SW  *sw = (Physics_SW*)phys->data;
  PetscReal   uhn,u[DIM];
  PetscInt     i;

  PetscFunctionBeginUser;
  Scale2Real(1./x->h,x->uh,u);
  uhn  = x->uh[0] * n[0] + x->uh[1] * n[1];
  f->h = uhn;
  for (i=0; i<DIM; i++) f->uh[i] = u[i] * uhn + sw->gravity * PetscSqr(x->h) * n[i];
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsBoundary_SW_Wall(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *xI, PetscScalar *xG, void *ctx)
{
  PetscFunctionBeginUser;
  xG[0] = xI[0];
  xG[1] = -xI[1];
  xG[2] = -xI[2];
  PetscFunctionReturn(0);
}

static void PhysicsRiemann_SW(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n, const PetscScalar *xL, const PetscScalar *xR, PetscInt numConstants, const PetscScalar constants[], PetscScalar *flux, Physics phys)
{
  Physics_SW   *sw = (Physics_SW*)phys->data;
  PetscReal    cL,cR,speed;
  PetscReal    nn[DIM];
#if !defined(PETSC_USE_COMPLEX)
  const SWNode *uL = (const SWNode*)xL,*uR = (const SWNode*)xR;
#else
  SWNodeUnion  uLreal, uRreal;
  const SWNode *uL = &uLreal.swnode;
  const SWNode *uR = &uRreal.swnode;
#endif
  SWNodeUnion  fL,fR;
  PetscInt     i;
  PetscReal    zero=0.;

#if defined(PETSC_USE_COMPLEX)
  uLreal.swnode.h = 0; uRreal.swnode.h = 0;
  for (i = 0; i < 1+dim; i++) uLreal.vals[i] = PetscRealPart(xL[i]);
  for (i = 0; i < 1+dim; i++) uRreal.vals[i] = PetscRealPart(xR[i]);
#endif
  if (uL->h < 0 || uR->h < 0) {for (i=0; i<1+dim; i++) flux[i] = zero/zero; return;} /* SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Reconstructed thickness is negative"); */
  nn[0] = n[0];
  nn[1] = n[1];
  Normalize2Real(nn);
  SWFlux(phys,nn,uL,&(fL.swnode));
  SWFlux(phys,nn,uR,&(fR.swnode));
  cL    = PetscSqrtReal(sw->gravity*uL->h);
  cR    = PetscSqrtReal(sw->gravity*uR->h); /* gravity wave speed */
  speed = PetscMax(PetscAbsReal(Dot2Real(uL->uh,nn)/uL->h) + cL,PetscAbsReal(Dot2Real(uR->uh,nn)/uR->h) + cR);
  for (i=0; i<1+dim; i++) flux[i] = (0.5*(fL.vals[i] + fR.vals[i]) + 0.5*speed*(xL[i] - xR[i])) * Norm2Real(n);
}

static PetscErrorCode PhysicsSolution_SW(Model mod,PetscReal time,const PetscReal *x,PetscScalar *u,void *ctx)
{
  PetscReal dx[2],r,sigma;
  PetscBool bc1 = PETSC_FALSE;
  PetscOptionsGetBool(NULL, NULL, "-bc", &bc1, NULL);
  PetscFunctionBeginUser;
    if (time != 0.0) SETERRQ1(mod->comm, PETSC_ERR_SUP, "No solution known for time %g", (double) time);
    if (bc1) {
        dx[0] = x[0] - 1.5;
        dx[1] = x[1] - 1.0;
        r = Norm2Real(dx);
        sigma = 0.5;
        u[0] = 1 + 2 * PetscExpReal(-PetscSqr(r) / (2 * PetscSqr(sigma)));
        u[1] = 0.0;
        u[2] = 0.0;
    } else {
        if (x[0] <= 0.30)
            u[0] = 1.3;
        else
            u[0] = 1.0;
        u[1] = 0.0;
        u[2] = 0.0;
    }
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsFunctional_SW(Model mod,PetscReal time,const PetscReal *coord,const PetscScalar *xx,PetscReal *f,void *ctx)
{
  Physics      phys = (Physics)ctx;
  Physics_SW   *sw  = (Physics_SW*)phys->data;
  const SWNode *x   = (const SWNode*)xx;
  PetscReal  u[2];
  PetscReal    h;

  PetscFunctionBeginUser;
  h = x->h;
  Scale2Real(1./x->h,x->uh,u);
  f[sw->functional.Height] = h;
  f[sw->functional.Speed]  = Norm2Real(u) + PetscSqrtReal(sw->gravity*h);
  f[sw->functional.Energy] = 0.5*(Dot2Real(x->uh,u) + sw->gravity*PetscSqr(h));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetUpBC_SW(PetscDS prob,Physics phys)
{
  PetscErrorCode ierr;
  const PetscInt wallids_plex[] = {1, 2, 3, 4};
  PetscFunctionBeginUser;
  ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "wall", "Face Sets", 0, 0, NULL, (void (*)(void)) PhysicsBoundary_SW_Wall, ALEN(wallids_plex), wallids_plex, phys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCreate_SW(Model mod,Physics phys,PetscOptionItems *PetscOptionsObject)
{
  Physics_SW     *sw;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  phys->field_desc = PhysicsFields_SW;
  phys->riemann = (PetscRiemannFunc) PhysicsRiemann_SW;
  ierr          = PetscNew(&sw);CHKERRQ(ierr);
  phys->data    = sw;
  mod->setupbc  = SetUpBC_SW;

  ierr          = PetscOptionsHead(PetscOptionsObject,"SW options");CHKERRQ(ierr);
  {
    sw->gravity = 1.0;
    ierr = PetscOptionsReal("-sw_gravity","Gravitational constant","",sw->gravity,&sw->gravity,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  phys->maxspeed = PetscSqrtReal(2.0*sw->gravity); /* Mach 1 for depth of 2 */

  ierr = ModelSolutionSetDefault(mod,PhysicsSolution_SW,phys);CHKERRQ(ierr);
  ierr = ModelFunctionalRegister(mod,"Height",&sw->functional.Height,PhysicsFunctional_SW,phys);CHKERRQ(ierr);
  ierr = ModelFunctionalRegister(mod,"Speed",&sw->functional.Speed,PhysicsFunctional_SW,phys);CHKERRQ(ierr);
  ierr = ModelFunctionalRegister(mod,"Energy",&sw->functional.Energy,PhysicsFunctional_SW,phys);CHKERRQ(ierr);

  mod->bcs[0] = mod->bcs[1] = mod->bcs[2] = DM_BOUNDARY_GHOSTED;

  PetscFunctionReturn(0);
}

/******************* Euler Density Shock (EULER_IV_SHOCK,EULER_SS_SHOCK) ********************/
/* An initial-value and self-similar solutions of the compressible Euler equations */
/* Ravi Samtaney and D. I. Pullin */
/* Phys. Fluids 8, 2650 (1996); http://dx.doi.org/10.1063/1.869050 */
typedef enum {EULER_PAR_GAMMA,EULER_PAR_RHOR,EULER_PAR_AMACH,EULER_PAR_ITANA,EULER_PAR_SIZE} EulerParamIdx;
typedef enum {EULER_IV_SHOCK,EULER_SS_SHOCK,EULER_SHOCK_TUBE,EULER_LINEAR_WAVE} EulerType;
typedef struct {
  PetscReal r;
  PetscReal ru[DIM];
  PetscReal E;
} EulerNode;
typedef union {
  EulerNode eulernode;
  PetscReal vals[DIM+2];
} EulerNodeUnion;
typedef PetscErrorCode (*EquationOfState)(const PetscReal*, const EulerNode*, PetscReal*);
typedef struct {
  EulerType       type;
  PetscReal       pars[EULER_PAR_SIZE];
  EquationOfState sound;
  struct {
    PetscInt Density;
    PetscInt Momentum;
    PetscInt Energy;
    PetscInt Pressure;
    PetscInt Speed;
  } monitor;
} Physics_Euler;

static const struct FieldDescription PhysicsFields_Euler[] = {{"Density",1},{"Momentum",DIM},{"Energy",1},{NULL,0}};

/* initial condition */
int initLinearWave(EulerNode *ux, const PetscReal gamma, const PetscReal coord[], const PetscReal Lx);
static PetscErrorCode PhysicsSolution_Euler(Model mod, PetscReal time, const PetscReal *x, PetscScalar *u, void *ctx)
{
  PetscInt i;
  Physics         phys = (Physics)ctx;
  Physics_Euler   *eu  = (Physics_Euler*)phys->data;
  EulerNode       *uu  = (EulerNode*)u;
  PetscReal        p0,gamma,c;
  PetscFunctionBeginUser;
  if (time != 0.0) SETERRQ1(mod->comm,PETSC_ERR_SUP,"No solution known for time %g",(double)time);

  for (i=0; i<DIM; i++) uu->ru[i] = 0.0; /* zero out initial velocity */
  /* set E and rho */
  gamma = eu->pars[EULER_PAR_GAMMA];

  if (eu->type==EULER_IV_SHOCK || eu->type==EULER_SS_SHOCK) {
    /******************* Euler Density Shock ********************/
    /* On initial-value and self-similar solutions of the compressible Euler equations */
    /* Ravi Samtaney and D. I. Pullin */
    /* Phys. Fluids 8, 2650 (1996); http://dx.doi.org/10.1063/1.869050 */
    /* initial conditions 1: left of shock, 0: left of discontinuity 2: right of discontinuity,  */
    p0 = 1.;
    if (x[0] < 0.0 + x[1]*eu->pars[EULER_PAR_ITANA]) {
      if (x[0] < mod->bounds[0]*0.5) { /* left of shock (1) */
        PetscReal amach,rho,press,gas1,p1;
        amach = eu->pars[EULER_PAR_AMACH];
        rho = 1.;
        press = p0;
        p1 = press*(1.0+2.0*gamma/(gamma+1.0)*(amach*amach-1.0));
        gas1 = (gamma-1.0)/(gamma+1.0);
        uu->r = rho*(p1/press+gas1)/(gas1*p1/press+1.0);
        uu->ru[0]   = ((uu->r - rho)*PetscSqrtReal(gamma*press/rho)*amach);
        uu->E = p1/(gamma-1.0) + .5/uu->r*uu->ru[0]*uu->ru[0];
      }
      else { /* left of discontinuity (0) */
        uu->r = 1.; /* rho = 1 */
        uu->E = p0/(gamma-1.0);
      }
    }
    else { /* right of discontinuity (2) */
      uu->r = eu->pars[EULER_PAR_RHOR];
      uu->E = p0/(gamma-1.0);
    }
  }
  else if (eu->type==EULER_SHOCK_TUBE) {
    /* For (x<x0) set (rho,u,p)=(8,0,10) and for (x>x0) set (rho,u,p)=(1,0,1). Choose x0 to the midpoint of the domain in the x-direction. */
    if (x[0] < 0.0 ) {
      uu->r = 8.;
      uu->E = 10./(gamma-1.);
    }
    else {
      uu->r = 1.;
      uu->E = 1./(gamma-1.);
    }
  }
  else if (eu->type==EULER_LINEAR_WAVE) {
    initLinearWave( uu, gamma, x, mod->bounds[1] - mod->bounds[0]);
  }
  else SETERRQ1(mod->comm,PETSC_ERR_SUP,"Unknown type %d",eu->type);

  /* set phys->maxspeed: (mod->maxspeed = phys->maxspeed) in main; */
  eu->sound(&gamma,uu,&c);
  c = (uu->ru[0]/uu->r) + c;
  if (c > phys->maxspeed) phys->maxspeed = c;

  PetscFunctionReturn(0);
}

static PetscErrorCode Pressure_PG(const PetscReal gamma,const EulerNode *x,PetscReal *p)
{
  PetscReal ru2;

  PetscFunctionBeginUser;
  ru2  = DotDIMReal(x->ru,x->ru);
  (*p)=(x->E - 0.5*ru2/x->r)*(gamma - 1.0); /* (E - rho V^2/2)(gamma-1) = e rho (gamma-1) */
  PetscFunctionReturn(0);
}

static PetscErrorCode SpeedOfSound_PG(const PetscReal *gamma, const EulerNode *x, PetscReal *c)
{
  PetscReal p;

  PetscFunctionBeginUser;
  Pressure_PG(*gamma,x,&p);
  if (p<0.) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"negative pressure time %g -- NEED TO FIX!!!!!!",(double) p);
  /* pars[EULER_PAR_GAMMA] = heat capacity ratio */
  (*c)=PetscSqrtReal(*gamma * p / x->r);
  PetscFunctionReturn(0);
}

/*
 * x = (rho,rho*(u_1),...,rho*e)^T
 * x_t+div(f_1(x))+...+div(f_DIM(x)) = 0
 *
 * f_i(x) = u_i*x+(0,0,...,p,...,p*u_i)^T
 *
 */
static PetscErrorCode EulerFlux(Physics phys,const PetscReal *n,const EulerNode *x,EulerNode *f)
{
  Physics_Euler *eu = (Physics_Euler*)phys->data;
  PetscReal     nu,p;
  PetscInt      i;

  PetscFunctionBeginUser;
  Pressure_PG(eu->pars[EULER_PAR_GAMMA],x,&p);
  nu = DotDIMReal(x->ru,n);
  f->r = nu;   /* A rho u */
  nu /= x->r;  /* A u */
  for (i=0; i<DIM; i++) f->ru[i] = nu * x->ru[i] + n[i]*p;  /* r u^2 + p */
  f->E = nu * (x->E + p); /* u(e+p) */
  PetscFunctionReturn(0);
}

/* PetscReal* => EulerNode* conversion */
static PetscErrorCode PhysicsBoundary_Euler_Wall(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx)
{
  PetscInt    i;
  const EulerNode *xI = (const EulerNode*)a_xI;
  EulerNode       *xG = (EulerNode*)a_xG;
  Physics         phys = (Physics)ctx;
  Physics_Euler   *eu  = (Physics_Euler*)phys->data;
  PetscFunctionBeginUser;
  xG->r = xI->r;           /* ghost cell density - same */
  xG->E = xI->E;           /* ghost cell energy - same */
  if (n[1] != 0.) {        /* top and bottom */
    xG->ru[0] =  xI->ru[0]; /* copy tang to wall */
    xG->ru[1] = -xI->ru[1]; /* reflect perp to t/b wall */
  }
  else { /* sides */
    for (i=0; i<DIM; i++) xG->ru[i] = xI->ru[i]; /* copy */
  }
  if (eu->type == EULER_LINEAR_WAVE) { /* debug */
#if 0
    PetscPrintf(PETSC_COMM_WORLD,"%s coord=%g,%g\n",PETSC_FUNCTION_NAME,c[0],c[1]);
#endif
  }
  PetscFunctionReturn(0);
}
int godunovflux( const PetscScalar *ul, const PetscScalar *ur, PetscScalar *flux, const PetscReal *nn, const int *ndim, const PetscReal *gamma);
/* PetscReal* => EulerNode* conversion */
static void PhysicsRiemann_Euler_Godunov( PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n,
                                          const PetscScalar *xL, const PetscScalar *xR, PetscInt numConstants, const PetscScalar constants[], PetscScalar *flux, Physics phys)
{
  Physics_Euler   *eu = (Physics_Euler*)phys->data;
  PetscReal       cL,cR,speed,velL,velR,nn[DIM],s2;
  PetscInt        i;
  PetscErrorCode  ierr;
  PetscFunctionBeginUser;

  for (i=0,s2=0.; i<DIM; i++) {
    nn[i] = n[i];
    s2 += nn[i]*nn[i];
  }
  s2 = PetscSqrtReal(s2); /* |n|_2 = sum(n^2)^1/2 */
  for (i=0.; i<DIM; i++) nn[i] /= s2;
  if (0) { /* Rusanov */
    const EulerNode *uL = (const EulerNode*)xL,*uR = (const EulerNode*)xR;
    EulerNodeUnion  fL,fR;
    EulerFlux(phys,nn,uL,&(fL.eulernode));
    EulerFlux(phys,nn,uR,&(fR.eulernode));
    ierr = eu->sound(&eu->pars[EULER_PAR_GAMMA],uL,&cL);if (ierr) exit(13);
    ierr = eu->sound(&eu->pars[EULER_PAR_GAMMA],uR,&cR);if (ierr) exit(14);
    velL = DotDIMReal(uL->ru,nn)/uL->r;
    velR = DotDIMReal(uR->ru,nn)/uR->r;
    speed = PetscMax(velR + cR, velL + cL);
    for (i=0; i<2+dim; i++) flux[i] = 0.5*((fL.vals[i]+fR.vals[i]) + speed*(xL[i] - xR[i]))*s2;
  }
  else {
    int dim = DIM;
    /* int iwave =  */
    godunovflux(xL, xR, flux, nn, &dim, &eu->pars[EULER_PAR_GAMMA]);
    for (i=0; i<2+dim; i++) flux[i] *= s2;
  }
  PetscFunctionReturnVoid();
}

static PetscErrorCode PhysicsFunctional_Euler(Model mod,PetscReal time,const PetscReal *coord,const PetscScalar *xx,PetscReal *f,void *ctx)
{
  Physics         phys = (Physics)ctx;
  Physics_Euler   *eu  = (Physics_Euler*)phys->data;
  const EulerNode *x   = (const EulerNode*)xx;
  PetscReal       p;

  PetscFunctionBeginUser;
  f[eu->monitor.Density]  = x->r;
  f[eu->monitor.Momentum] = NormDIM(x->ru);
  f[eu->monitor.Energy]   = x->E;
  f[eu->monitor.Speed]    = NormDIM(x->ru)/x->r;
  Pressure_PG(eu->pars[EULER_PAR_GAMMA], x, &p);
  f[eu->monitor.Pressure] = p;
  PetscFunctionReturn(0);
}

static PetscErrorCode SetUpBC_Euler(PetscDS prob,Physics phys)
{
  PetscErrorCode  ierr;
  Physics_Euler   *eu = (Physics_Euler *) phys->data;
  if (eu->type == EULER_LINEAR_WAVE) {
    const PetscInt wallids[] = {100,101};
    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "wall", "Face Sets", 0, 0, NULL, (void (*)(void)) PhysicsBoundary_Euler_Wall, ALEN(wallids), wallids, phys);CHKERRQ(ierr);
  }
  else {
    const PetscInt wallids[] = {100,101,200,300};
    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "wall", "Face Sets", 0, 0, NULL, (void (*)(void)) PhysicsBoundary_Euler_Wall, ALEN(wallids), wallids, phys);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCreate_Euler(Model mod,Physics phys,PetscOptionItems *PetscOptionsObject)
{
  Physics_Euler   *eu;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  phys->field_desc = PhysicsFields_Euler;
  phys->riemann = (PetscRiemannFunc) PhysicsRiemann_Euler_Godunov;
  ierr = PetscNew(&eu);CHKERRQ(ierr);
  phys->data    = eu;
  mod->setupbc = SetUpBC_Euler;
  ierr = PetscOptionsHead(PetscOptionsObject,"Euler options");CHKERRQ(ierr);
  {
    PetscReal alpha;
    char type[64] = "linear_wave";
    PetscBool  is;
    mod->bcs[0] = mod->bcs[1] = mod->bcs[2] = DM_BOUNDARY_GHOSTED;
    eu->pars[EULER_PAR_GAMMA] = 1.4;
    eu->pars[EULER_PAR_AMACH] = 2.02;
    eu->pars[EULER_PAR_RHOR] = 3.0;
    eu->pars[EULER_PAR_ITANA] = 0.57735026918963; /* angle of Euler self similar (SS) shock */
    ierr = PetscOptionsReal("-eu_gamma","Heat capacity ratio","",eu->pars[EULER_PAR_GAMMA],&eu->pars[EULER_PAR_GAMMA],NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eu_amach","Shock speed (Mach)","",eu->pars[EULER_PAR_AMACH],&eu->pars[EULER_PAR_AMACH],NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eu_rho2","Density right of discontinuity","",eu->pars[EULER_PAR_RHOR],&eu->pars[EULER_PAR_RHOR],NULL);CHKERRQ(ierr);
    alpha = 60.;
    ierr = PetscOptionsReal("-eu_alpha","Angle of discontinuity","",alpha,&alpha,NULL);CHKERRQ(ierr);
    if (alpha<=0. || alpha>90.) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Alpha bust be > 0 and <= 90 (%g)",alpha);
    eu->pars[EULER_PAR_ITANA] = 1./PetscTanReal( alpha * PETSC_PI / 180.0 );
    ierr = PetscOptionsString("-eu_type","Type of Euler test","",type,type,sizeof(type),NULL);CHKERRQ(ierr);
    ierr = PetscStrcmp(type,"linear_wave", &is);CHKERRQ(ierr);
    if (is) {
      eu->type = EULER_LINEAR_WAVE;
      mod->bcs[0] = mod->bcs[1] = mod->bcs[2] = DM_BOUNDARY_PERIODIC;
      mod->bcs[1] = DM_BOUNDARY_GHOSTED; /* debug */
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%s set Euler type: %s\n",PETSC_FUNCTION_NAME,"linear_wave");CHKERRQ(ierr);
    }
    else {
      if (DIM != 2) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"DIM must be 2 unless linear wave test %s",type);
      ierr = PetscStrcmp(type,"iv_shock", &is);CHKERRQ(ierr);
      if (is) {
        eu->type = EULER_IV_SHOCK;
        ierr = PetscPrintf(PETSC_COMM_WORLD,"%s set Euler type: %s\n",PETSC_FUNCTION_NAME,"iv_shock");CHKERRQ(ierr);
      }
      else {
        ierr = PetscStrcmp(type,"ss_shock", &is);CHKERRQ(ierr);
        if (is) {
          eu->type = EULER_SS_SHOCK;
          ierr = PetscPrintf(PETSC_COMM_WORLD,"%s set Euler type: %s\n",PETSC_FUNCTION_NAME,"ss_shock");CHKERRQ(ierr);
        }
        else {
          ierr = PetscStrcmp(type,"shock_tube", &is);CHKERRQ(ierr);
          if (is) eu->type = EULER_SHOCK_TUBE;
          else SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unknown Euler type %s",type);
          ierr = PetscPrintf(PETSC_COMM_WORLD,"%s set Euler type: %s\n",PETSC_FUNCTION_NAME,"shock_tube");CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  eu->sound = SpeedOfSound_PG;
  phys->maxspeed = 0.; /* will get set in solution */
  ierr = ModelSolutionSetDefault(mod,PhysicsSolution_Euler,phys);CHKERRQ(ierr);
  ierr = ModelFunctionalRegister(mod,"Speed",&eu->monitor.Speed,PhysicsFunctional_Euler,phys);CHKERRQ(ierr);
  ierr = ModelFunctionalRegister(mod,"Energy",&eu->monitor.Energy,PhysicsFunctional_Euler,phys);CHKERRQ(ierr);
  ierr = ModelFunctionalRegister(mod,"Density",&eu->monitor.Density,PhysicsFunctional_Euler,phys);CHKERRQ(ierr);
  ierr = ModelFunctionalRegister(mod,"Momentum",&eu->monitor.Momentum,PhysicsFunctional_Euler,phys);CHKERRQ(ierr);
  ierr = ModelFunctionalRegister(mod,"Pressure",&eu->monitor.Pressure,PhysicsFunctional_Euler,phys);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ErrorIndicator_Simple(PetscInt dim, PetscReal volume, PetscInt numComps, const PetscScalar u[], const PetscScalar grad[], PetscReal *error, void *ctx)
{
  PetscReal      err = 0.;
  PetscInt       i, j;

  PetscFunctionBeginUser;
  for (i = 0; i < numComps; i++) {
    for (j = 0; j < dim; j++) {
      err += PetscSqr(PetscRealPart(grad[i * dim + j]));
    }
  }
  *error = volume * err;
  PetscFunctionReturn(0);
}

static PetscErrorCode ErrorIndicator_Height(PetscInt dim, PetscReal volume, const PetscScalar grad[], PetscReal *error)
{
    PetscReal      err = 0.;
    PetscInt       j;

    PetscFunctionBeginUser;
    for (j = 0; j < dim; j++) {
        err += PetscSqr(PetscRealPart(grad[j]));
    }
    *error = volume * PetscSqrtReal(err);
    PetscFunctionReturn(0);
}

static PetscErrorCode ErrorIndicator_Average(PetscInt dim, PetscReal volume, const PetscScalar grad[], PetscReal *error)
{
    PetscReal      err_height, err_velocity;

    PetscFunctionBeginUser;
    err_height = (PetscSqr(PetscRealPart(grad[0])) + PetscSqr(PetscRealPart(grad[1])))*0.5;
    err_velocity = (PetscSqr(PetscRealPart(grad[2])) + PetscSqr(PetscRealPart(grad[5])))*0.5;
    *error = volume * (err_height + err_velocity);
    PetscFunctionReturn(0);
}

PetscErrorCode ConstructCellBoundary(DM dm, User user)
{
  const char     *name   = "Cell Sets";
  const char     *bdname = "split faces";
  IS             regionIS, innerIS;
  const PetscInt *regions, *cells;
  PetscInt       numRegions, innerRegion, numCells, c;
  PetscInt       cStart, cEnd, cEndInterior, fStart, fEnd;
  PetscBool      hasLabel;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetGhostCellStratum(dm, &cEndInterior, NULL);CHKERRQ(ierr);

  ierr = DMHasLabel(dm, name, &hasLabel);CHKERRQ(ierr);
  if (!hasLabel) PetscFunctionReturn(0);
  ierr = DMGetLabelSize(dm, name, &numRegions);CHKERRQ(ierr);
  if (numRegions != 2) PetscFunctionReturn(0);
  /* Get the inner id */
  ierr = DMGetLabelIdIS(dm, name, &regionIS);CHKERRQ(ierr);
  ierr = ISGetIndices(regionIS, &regions);CHKERRQ(ierr);
  innerRegion = regions[0];
  ierr = ISRestoreIndices(regionIS, &regions);CHKERRQ(ierr);
  ierr = ISDestroy(&regionIS);CHKERRQ(ierr);
  /* Find the faces between cells in different regions, could call DMPlexCreateNeighborCSR() */
  ierr = DMGetStratumIS(dm, name, innerRegion, &innerIS);CHKERRQ(ierr);
  ierr = ISGetLocalSize(innerIS, &numCells);CHKERRQ(ierr);
  ierr = ISGetIndices(innerIS, &cells);CHKERRQ(ierr);
  ierr = DMCreateLabel(dm, bdname);CHKERRQ(ierr);
  for (c = 0; c < numCells; ++c) {
    const PetscInt cell = cells[c];
    const PetscInt *faces;
    PetscInt       numFaces, f;

    if ((cell < cStart) || (cell >= cEnd)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Got invalid point %d which is not a cell", cell);
    ierr = DMPlexGetConeSize(dm, cell, &numFaces);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, cell, &faces);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f) {
      const PetscInt face = faces[f];
      const PetscInt *neighbors;
      PetscInt       nC, regionA, regionB;

      if ((face < fStart) || (face >= fEnd)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Got invalid point %d which is not a face", face);
      ierr = DMPlexGetSupportSize(dm, face, &nC);CHKERRQ(ierr);
      if (nC != 2) continue;
      ierr = DMPlexGetSupport(dm, face, &neighbors);CHKERRQ(ierr);
      if ((neighbors[0] >= cEndInterior) || (neighbors[1] >= cEndInterior)) continue;
      if ((neighbors[0] < cStart) || (neighbors[0] >= cEnd)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Got invalid point %d which is not a cell", neighbors[0]);
      if ((neighbors[1] < cStart) || (neighbors[1] >= cEnd)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Got invalid point %d which is not a cell", neighbors[1]);
      ierr = DMGetLabelValue(dm, name, neighbors[0], &regionA);CHKERRQ(ierr);
      ierr = DMGetLabelValue(dm, name, neighbors[1], &regionB);CHKERRQ(ierr);
      if (regionA < 0) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Invalid label %s: Cell %d has no value", name, neighbors[0]);
      if (regionB < 0) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Invalid label %s: Cell %d has no value", name, neighbors[1]);
      if (regionA != regionB) {
        ierr = DMSetLabelValue(dm, bdname, faces[f], 1);CHKERRQ(ierr);
      }
    }
  }
  ierr = ISRestoreIndices(innerIS, &cells);CHKERRQ(ierr);
  ierr = ISDestroy(&innerIS);CHKERRQ(ierr);
  {
    DMLabel label;

    ierr = DMGetLabel(dm, bdname, &label);CHKERRQ(ierr);
    ierr = DMLabelView(label, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Right now, I have just added duplicate faces, which see both cells. We can
- Add duplicate vertices and decouple the face cones
- Disconnect faces from cells across the rotation gap
*/
PetscErrorCode SplitFaces(DM *dmSplit, const char labelName[], User user)
{
  DM             dm = *dmSplit, sdm;
  PetscSF        sfPoint, gsfPoint;
  PetscSection   coordSection, newCoordSection;
  Vec            coordinates;
  IS             idIS;
  const PetscInt *ids;
  PetscInt       *newpoints;
  PetscInt       dim, depth, maxConeSize, maxSupportSize, numLabels, numGhostCells;
  PetscInt       numFS, fs, pStart, pEnd, p, cEnd, cEndInterior, vStart, vEnd, v, fStart, fEnd, newf, d, l;
  PetscBool      hasLabel;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMHasLabel(dm, labelName, &hasLabel);CHKERRQ(ierr);
  if (!hasLabel) PetscFunctionReturn(0);
  ierr = DMCreate(PetscObjectComm((PetscObject)dm), &sdm);CHKERRQ(ierr);
  ierr = DMSetType(sdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMSetDimension(sdm, dim);CHKERRQ(ierr);

  ierr = DMGetLabelIdIS(dm, labelName, &idIS);CHKERRQ(ierr);
  ierr = ISGetLocalSize(idIS, &numFS);CHKERRQ(ierr);
  ierr = ISGetIndices(idIS, &ids);CHKERRQ(ierr);

  user->numSplitFaces = 0;
  for (fs = 0; fs < numFS; ++fs) {
    PetscInt numBdFaces;

    ierr = DMGetStratumSize(dm, labelName, ids[fs], &numBdFaces);CHKERRQ(ierr);
    user->numSplitFaces += numBdFaces;
  }
  ierr  = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  pEnd += user->numSplitFaces;
  ierr  = DMPlexSetChart(sdm, pStart, pEnd);CHKERRQ(ierr);
  ierr  = DMPlexGetGhostCellStratum(dm, &cEndInterior, NULL);CHKERRQ(ierr);
  ierr  = DMPlexGetHeightStratum(dm, 0, NULL, &cEnd);CHKERRQ(ierr);
  numGhostCells = cEnd - cEndInterior;
  /* Set cone and support sizes */
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt newp = p;
      PetscInt size;

      ierr = DMPlexGetConeSize(dm, p, &size);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(sdm, newp, size);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, p, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(sdm, newp, size);CHKERRQ(ierr);
    }
  }
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  for (fs = 0, newf = fEnd; fs < numFS; ++fs) {
    IS             faceIS;
    const PetscInt *faces;
    PetscInt       numFaces, f;

    ierr = DMGetStratumIS(dm, labelName, ids[fs], &faceIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(faceIS, &numFaces);CHKERRQ(ierr);
    ierr = ISGetIndices(faceIS, &faces);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f, ++newf) {
      PetscInt size;

      /* Right now I think that both faces should see both cells */
      ierr = DMPlexGetConeSize(dm, faces[f], &size);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(sdm, newf, size);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, faces[f], &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(sdm, newf, size);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(faceIS, &faces);CHKERRQ(ierr);
    ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
  }
  ierr = DMSetUp(sdm);CHKERRQ(ierr);
  /* Set cones and supports */
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
  ierr = PetscMalloc1(PetscMax(maxConeSize, maxSupportSize), &newpoints);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    const PetscInt *points, *orientations;
    PetscInt       size, i, newp = p;

    ierr = DMPlexGetConeSize(dm, p, &size);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, p, &points);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dm, p, &orientations);CHKERRQ(ierr);
    for (i = 0; i < size; ++i) newpoints[i] = points[i];
    ierr = DMPlexSetCone(sdm, newp, newpoints);CHKERRQ(ierr);
    ierr = DMPlexSetConeOrientation(sdm, newp, orientations);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dm, p, &size);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, p, &points);CHKERRQ(ierr);
    for (i = 0; i < size; ++i) newpoints[i] = points[i];
    ierr = DMPlexSetSupport(sdm, newp, newpoints);CHKERRQ(ierr);
  }
  ierr = PetscFree(newpoints);CHKERRQ(ierr);
  for (fs = 0, newf = fEnd; fs < numFS; ++fs) {
    IS             faceIS;
    const PetscInt *faces;
    PetscInt       numFaces, f;

    ierr = DMGetStratumIS(dm, labelName, ids[fs], &faceIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(faceIS, &numFaces);CHKERRQ(ierr);
    ierr = ISGetIndices(faceIS, &faces);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f, ++newf) {
      const PetscInt *points;

      ierr = DMPlexGetCone(dm, faces[f], &points);CHKERRQ(ierr);
      ierr = DMPlexSetCone(sdm, newf, points);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, faces[f], &points);CHKERRQ(ierr);
      ierr = DMPlexSetSupport(sdm, newf, points);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(faceIS, &faces);CHKERRQ(ierr);
    ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(idIS, &ids);CHKERRQ(ierr);
  ierr = ISDestroy(&idIS);CHKERRQ(ierr);
  ierr = DMPlexStratify(sdm);CHKERRQ(ierr);
  ierr = DMPlexSetGhostCellStratum(sdm, cEndInterior, PETSC_DETERMINE);CHKERRQ(ierr);
  /* Convert coordinates */
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &newCoordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(newCoordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(newCoordSection, 0, dim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(newCoordSection, vStart, vEnd);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    ierr = PetscSectionSetDof(newCoordSection, v, dim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(newCoordSection, v, 0, dim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(newCoordSection);CHKERRQ(ierr);
  ierr = DMSetCoordinateSection(sdm, PETSC_DETERMINE, newCoordSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&newCoordSection);CHKERRQ(ierr); /* relinquish our reference */
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(sdm, coordinates);CHKERRQ(ierr);
  /* Convert labels */
  ierr = DMGetNumLabels(dm, &numLabels);CHKERRQ(ierr);
  for (l = 0; l < numLabels; ++l) {
    const char *lname;
    PetscBool  isDepth, isDim;

    ierr = DMGetLabelName(dm, l, &lname);CHKERRQ(ierr);
    ierr = PetscStrcmp(lname, "depth", &isDepth);CHKERRQ(ierr);
    if (isDepth) continue;
    ierr = PetscStrcmp(lname, "dim", &isDim);CHKERRQ(ierr);
    if (isDim) continue;
    ierr = DMCreateLabel(sdm, lname);CHKERRQ(ierr);
    ierr = DMGetLabelIdIS(dm, lname, &idIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(idIS, &numFS);CHKERRQ(ierr);
    ierr = ISGetIndices(idIS, &ids);CHKERRQ(ierr);
    for (fs = 0; fs < numFS; ++fs) {
      IS             pointIS;
      const PetscInt *points;
      PetscInt       numPoints;

      ierr = DMGetStratumIS(dm, lname, ids[fs], &pointIS);CHKERRQ(ierr);
      ierr = ISGetLocalSize(pointIS, &numPoints);CHKERRQ(ierr);
      ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
      for (p = 0; p < numPoints; ++p) {
        PetscInt newpoint = points[p];

        ierr = DMSetLabelValue(sdm, lname, newpoint, ids[fs]);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(idIS, &ids);CHKERRQ(ierr);
    ierr = ISDestroy(&idIS);CHKERRQ(ierr);
  }
  {
    /* Convert pointSF */
    const PetscSFNode *remotePoints;
    PetscSFNode       *gremotePoints;
    const PetscInt    *localPoints;
    PetscInt          *glocalPoints,*newLocation,*newRemoteLocation;
    PetscInt          numRoots, numLeaves;
    PetscMPIInt       size;

    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size);CHKERRQ(ierr);
    ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
    ierr = DMGetPointSF(sdm, &gsfPoint);CHKERRQ(ierr);
    ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints);CHKERRQ(ierr);
    if (numRoots >= 0) {
      ierr = PetscMalloc2(numRoots,&newLocation,pEnd-pStart,&newRemoteLocation);CHKERRQ(ierr);
      for (l=0; l<numRoots; l++) newLocation[l] = l; /* + (l >= cEnd ? numGhostCells : 0); */
      ierr = PetscSFBcastBegin(sfPoint, MPIU_INT, newLocation, newRemoteLocation);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sfPoint, MPIU_INT, newLocation, newRemoteLocation);CHKERRQ(ierr);
      ierr = PetscMalloc1(numLeaves,    &glocalPoints);CHKERRQ(ierr);
      ierr = PetscMalloc1(numLeaves, &gremotePoints);CHKERRQ(ierr);
      for (l = 0; l < numLeaves; ++l) {
        glocalPoints[l]        = localPoints[l]; /* localPoints[l] >= cEnd ? localPoints[l] + numGhostCells : localPoints[l]; */
        gremotePoints[l].rank  = remotePoints[l].rank;
        gremotePoints[l].index = newRemoteLocation[localPoints[l]];
      }
      ierr = PetscFree2(newLocation,newRemoteLocation);CHKERRQ(ierr);
      ierr = PetscSFSetGraph(gsfPoint, numRoots+numGhostCells, numLeaves, glocalPoints, PETSC_OWN_POINTER, gremotePoints, PETSC_OWN_POINTER);CHKERRQ(ierr);
    }
    ierr     = DMDestroy(dmSplit);CHKERRQ(ierr);
    *dmSplit = sdm;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreatePartitionVec(DM dm, DM *dmCell, Vec *partition)
{
  /*
   * No need to worry about - used in the vtk routine for visualization purposes.
   *
   * Essentially creates a vector partition containing the rank for the cells.
   * Also create a dmCell object containing the coordinates of the local cells.
   * */
  PetscSF        sfPoint;
  PetscSection   coordSection;
  Vec            coordinates;
  PetscSection   sectionCell;
  PetscScalar    *part;
  PetscInt       cStart, cEnd, c;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMClone(dm, dmCell);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
  ierr = DMSetPointSF(*dmCell, sfPoint);CHKERRQ(ierr);
  ierr = DMSetCoordinateSection(*dmCell, PETSC_DETERMINE, coordSection);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(*dmCell, coordinates);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &sectionCell);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dmCell, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sectionCell, cStart, cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    ierr = PetscSectionSetDof(sectionCell, c, 1);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sectionCell);CHKERRQ(ierr);
  ierr = DMSetLocalSection(*dmCell, sectionCell);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sectionCell);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(*dmCell, partition);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)*partition, "partition");CHKERRQ(ierr);
  ierr = VecGetArray(*partition, &part);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *p;

    ierr = DMPlexPointLocalRef(*dmCell, c, part, &p);CHKERRQ(ierr);
    p[0] = rank;
  }
  ierr = VecRestoreArray(*partition, &part);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMassMatrix(DM dm, Vec *massMatrix, User user)
{
  DM                dmMass, dmFace, dmCell, dmCoord;
  PetscSection      coordSection;
  Vec               coordinates, facegeom, cellgeom;
  PetscSection      sectionMass;
  PetscScalar       *m;
  const PetscScalar *fgeom, *cgeom, *coords;
  PetscInt          vStart, vEnd, v;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMClone(dm, &dmMass);CHKERRQ(ierr);
  ierr = DMSetCoordinateSection(dmMass, PETSC_DETERMINE, coordSection);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dmMass, coordinates);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &sectionMass);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sectionMass, vStart, vEnd);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    PetscInt numFaces;

    ierr = DMPlexGetSupportSize(dmMass, v, &numFaces);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(sectionMass, v, numFaces*numFaces);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sectionMass);CHKERRQ(ierr);
  ierr = DMSetLocalSection(dmMass, sectionMass);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sectionMass);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmMass, massMatrix);CHKERRQ(ierr);
  ierr = VecGetArray(*massMatrix, &m);CHKERRQ(ierr);
  ierr = DMPlexTSGetGeometryFVM(dm, &facegeom, &cellgeom, NULL);CHKERRQ(ierr);
  ierr = VecGetDM(facegeom, &dmFace);CHKERRQ(ierr);
  ierr = VecGetArrayRead(facegeom, &fgeom);CHKERRQ(ierr);
  ierr = VecGetDM(cellgeom, &dmCell);CHKERRQ(ierr);
  ierr = VecGetArrayRead(cellgeom, &cgeom);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &dmCoord);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    const PetscInt        *faces;
    PetscFVFaceGeom       *fgA, *fgB, *cg;
    PetscScalar           *vertex;
    PetscInt               numFaces, sides[2], f, g;

    ierr = DMPlexPointLocalRead(dmCoord, v, coords, &vertex);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dmMass, v, &numFaces);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dmMass, v, &faces);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f) {
      sides[0] = faces[f];
      ierr = DMPlexPointLocalRead(dmFace, faces[f], fgeom, &fgA);CHKERRQ(ierr);
      for (g = 0; g < numFaces; ++g) {
        const PetscInt *cells = NULL;
        PetscReal      area   = 0.0;
        PetscInt       numCells;

        sides[1] = faces[g];
        ierr = DMPlexPointLocalRead(dmFace, faces[g], fgeom, &fgB);CHKERRQ(ierr);
        ierr = DMPlexGetJoin(dmMass, 2, sides, &numCells, &cells);CHKERRQ(ierr);
        if (numCells != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Invalid join for faces");
        ierr = DMPlexPointLocalRead(dmCell, cells[0], cgeom, &cg);CHKERRQ(ierr);
        area += PetscAbsScalar((vertex[0] - cg->centroid[0])*(fgA->centroid[1] - cg->centroid[1]) - (vertex[1] - cg->centroid[1])*(fgA->centroid[0] - cg->centroid[0]));
        area += PetscAbsScalar((vertex[0] - cg->centroid[0])*(fgB->centroid[1] - cg->centroid[1]) - (vertex[1] - cg->centroid[1])*(fgB->centroid[0] - cg->centroid[0]));
        m[f*numFaces+g] = Dot2Real(fgA->normal, fgB->normal)*area*0.5;
        ierr = DMPlexRestoreJoin(dmMass, 2, sides, &numCells, &cells);CHKERRQ(ierr);
      }
    }
  }
  ierr = VecRestoreArrayRead(facegeom, &fgeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(cellgeom, &cgeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecRestoreArray(*massMatrix, &m);CHKERRQ(ierr);
  ierr = DMDestroy(&dmMass);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Behavior will be different for multi-physics or when using non-default boundary conditions */
static PetscErrorCode ModelSolutionSetDefault(Model mod,SolutionFunction func,void *ctx)
{
  PetscFunctionBeginUser;
  mod->solution    = func;
  mod->solutionctx = ctx;
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelFunctionalRegister(Model mod,const char *name,PetscInt *offset,FunctionalFunction func,void *ctx)
{
  PetscErrorCode ierr;
  FunctionalLink link,*ptr;
  PetscInt       lastoffset = -1;

  PetscFunctionBeginUser;
  for (ptr=&mod->functionalRegistry; *ptr; ptr = &(*ptr)->next) lastoffset = (*ptr)->offset;
  ierr         = PetscNew(&link);CHKERRQ(ierr);
  ierr         = PetscStrallocpy(name,&link->name);CHKERRQ(ierr);
  link->offset = lastoffset + 1;
  link->func   = func;
  link->ctx    = ctx;
  link->next   = NULL;
  *ptr         = link;
  *offset      = link->offset;
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelFunctionalSetFromOptions(Model mod,PetscOptionItems *PetscOptionsObject)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  FunctionalLink link;
  char           *names[256];

  PetscFunctionBeginUser;
  mod->numMonitored = ALEN(names);
  ierr = PetscOptionsStringArray("-monitor","list of functionals to monitor","",names,&mod->numMonitored,NULL);CHKERRQ(ierr);
  /* Create list of functionals that will be computed somehow */
  ierr = PetscMalloc1(mod->numMonitored,&mod->functionalMonitored);CHKERRQ(ierr);
  /* Create index of calls that we will have to make to compute these functionals (over-allocation in general). */
  ierr = PetscMalloc1(mod->numMonitored,&mod->functionalCall);CHKERRQ(ierr);
  mod->numCall = 0;
  for (i=0; i<mod->numMonitored; i++) {
    for (link=mod->functionalRegistry; link; link=link->next) {
      PetscBool match;
      ierr = PetscStrcasecmp(names[i],link->name,&match);CHKERRQ(ierr);
      if (match) break;
    }
    if (!link) SETERRQ1(mod->comm,PETSC_ERR_USER,"No known functional '%s'",names[i]);
    mod->functionalMonitored[i] = link;
    for (j=0; j<i; j++) {
      if (mod->functionalCall[j]->func == link->func && mod->functionalCall[j]->ctx == link->ctx) goto next_name;
    }
    mod->functionalCall[mod->numCall++] = link; /* Just points to the first link using the result. There may be more results. */
next_name:
    ierr = PetscFree(names[i]);CHKERRQ(ierr);
  }

  /* Find out the maximum index of any functional computed by a function we will be calling (even if we are not using it) */
  mod->maxComputed = -1;
  for (link=mod->functionalRegistry; link; link=link->next) {
    for (i=0; i<mod->numCall; i++) {
      FunctionalLink call = mod->functionalCall[i];
      if (link->func == call->func && link->ctx == call->ctx) {
        mod->maxComputed = PetscMax(mod->maxComputed,link->offset);
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FunctionalLinkDestroy(FunctionalLink *link)
{
  PetscErrorCode ierr;
  FunctionalLink l,next;

  PetscFunctionBeginUser;
  if (!link) PetscFunctionReturn(0);
  l     = *link;
  *link = NULL;
  for (; l; l=next) {
    next = l->next;
    ierr = PetscFree(l->name);CHKERRQ(ierr);
    ierr = PetscFree(l);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* put the solution callback into a functional callback */
static PetscErrorCode SolutionFunctional(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *modctx)
{
  Model          mod;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  mod  = (Model) modctx;
  ierr = (*mod->solution)(mod, time, x, u, mod->solutionctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetInitialCondition(DM dm, Vec X, User user)
{
  PetscErrorCode     (*func[1]) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  void               *ctx[1];
  Model              mod = user->model;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  func[0] = SolutionFunctional;
  ctx[0]  = (void *) mod;
  ierr    = DMProjectFunction(dm,0.0,func,ctx,INSERT_ALL_VALUES,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode OutputVTK(DM dm, const char *filename, PetscViewer *viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscViewerCreate(PetscObjectComm((PetscObject)dm), viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*viewer, PETSCVIEWERVTK);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(*viewer, filename);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode OutputGrad(Vec X, PetscInt stepnum, DM gradDM)
{
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    PetscViewer viewer2;
    char GradBaseName[256] = "grad_ex11", filename2[PETSC_MAX_PATH_LEN];
    ierr = PetscSNPrintf(filename2, sizeof filename2, "./vtk_output/grad_output/%s-%03D.vtu", GradBaseName, stepnum);
    CHKERRQ(ierr);
    ierr = OutputVTK(gradDM, filename2, &viewer2);
    CHKERRQ(ierr);
    ierr = VecView(X, viewer2);
    CHKERRQ(ierr);
    PetscViewerDestroy(&viewer2);
    PetscFunctionReturn(0);
}

static PetscErrorCode MonitorVTK(TS ts,PetscInt stepnum,PetscReal time,Vec X,void *ctx)
{
  User           user = (User)ctx;
  DM             dm;
  Vec            cellgeom;
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN],*ftable = NULL;
  PetscReal      xnorm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectSetName((PetscObject) X, "u");CHKERRQ(ierr);
  ierr = VecGetDM(X,&dm);CHKERRQ(ierr);
  ierr = DMPlexTSGetGeometryFVM(dm, NULL, &cellgeom, NULL);CHKERRQ(ierr);
  ierr = VecNorm(X,NORM_INFINITY,&xnorm);CHKERRQ(ierr);

  if (stepnum >= 0) {
    stepnum += user->monitorStepOffset;
  }
  if (stepnum >= 0) {           /* No summary for final time */
    Model             mod = user->model;
    PetscInt          c,cStart,cEnd,fcount,i;
    size_t            ftableused,ftablealloc;
    const PetscScalar *cgeom,*x;
    DM                dmCell;
    DMLabel           vtkLabel;
    PetscReal         *fmin,*fmax,*fintegral,*ftmp;
    fcount = mod->maxComputed+1;
    ierr   = PetscMalloc4(fcount,&fmin,fcount,&fmax,fcount,&fintegral,fcount,&ftmp);CHKERRQ(ierr);
    for (i=0; i<fcount; i++) {
      fmin[i]      = PETSC_MAX_REAL;
      fmax[i]      = PETSC_MIN_REAL;
      fintegral[i] = 0;
    }
    ierr = VecGetDM(cellgeom,&dmCell);CHKERRQ(ierr);
    ierr = DMPlexGetInteriorCellStratum(dmCell,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellgeom,&cgeom);CHKERRQ(ierr);
    ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
    ierr = DMGetLabel(dm,"vtk",&vtkLabel);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      PetscFVCellGeom       *cg;
      const PetscScalar     *cx    = NULL;
      PetscInt              vtkVal = 0;

      /* not that these two routines as currently implemented work for any dm with a
       * localSection/globalSection */
      ierr = DMPlexPointLocalRead(dmCell,c,cgeom,&cg);CHKERRQ(ierr);
      ierr = DMPlexPointGlobalRead(dm,c,x,&cx);CHKERRQ(ierr);
      if (vtkLabel) {ierr = DMLabelGetValue(vtkLabel,c,&vtkVal);CHKERRQ(ierr);}
      if (!vtkVal || !cx) continue;        /* ghost, or not a global cell */
      for (i=0; i<mod->numCall; i++) {
        FunctionalLink flink = mod->functionalCall[i];
        ierr = (*flink->func)(mod,time,cg->centroid,cx,ftmp,flink->ctx);CHKERRQ(ierr);
      }
      for (i=0; i<fcount; i++) {
        fmin[i]       = PetscMin(fmin[i],ftmp[i]);
        fmax[i]       = PetscMax(fmax[i],ftmp[i]);
        fintegral[i] += cg->volume * ftmp[i];
      }
    }
    ierr = VecRestoreArrayRead(cellgeom,&cgeom);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
    ierr = MPI_Allreduce(MPI_IN_PLACE,fmin,fcount,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)ts));CHKERRQ(ierr);
    ierr = MPI_Allreduce(MPI_IN_PLACE,fmax,fcount,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)ts));CHKERRQ(ierr);
    ierr = MPI_Allreduce(MPI_IN_PLACE,fintegral,fcount,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)ts));CHKERRQ(ierr);

    ftablealloc = fcount * 100;
    ftableused  = 0;
    ierr        = PetscMalloc1(ftablealloc,&ftable);CHKERRQ(ierr);
    for (i=0; i<mod->numMonitored; i++) {
      size_t         countused;
      char           buffer[256],*p;
      FunctionalLink flink = mod->functionalMonitored[i];
      PetscInt       id    = flink->offset;
      if (i % 3) {
        ierr = PetscArraycpy(buffer,"  ",2);CHKERRQ(ierr);
        p    = buffer + 2;
      } else if (i) {
        char newline[] = "\n";
        ierr = PetscMemcpy(buffer,newline,sizeof(newline)-1);CHKERRQ(ierr);
        p    = buffer + sizeof(newline) - 1;
      } else {
        p = buffer;
      }
      ierr = PetscSNPrintfCount(p,sizeof buffer-(p-buffer),"%12s [%10.7g,%10.7g] int %10.7g",&countused,flink->name,(double)fmin[id],(double)fmax[id],(double)fintegral[id]);CHKERRQ(ierr);
      countused--;
      countused += p - buffer;
      if (countused > ftablealloc-ftableused-1) { /* reallocate */
        char *ftablenew;
        ftablealloc = 2*ftablealloc + countused;
        ierr = PetscMalloc(ftablealloc,&ftablenew);CHKERRQ(ierr);
        ierr = PetscArraycpy(ftablenew,ftable,ftableused);CHKERRQ(ierr);
        ierr = PetscFree(ftable);CHKERRQ(ierr);
        ftable = ftablenew;
      }
      ierr = PetscArraycpy(ftable+ftableused,buffer,countused);CHKERRQ(ierr);
      ftableused += countused;
      ftable[ftableused] = 0;
    }
    ierr = PetscFree4(fmin,fmax,fintegral,ftmp);CHKERRQ(ierr);

    ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"% 3D  time %8.4g  |x| %8.4g  %s\n",stepnum,(double)time,(double)xnorm,ftable ? ftable : "");CHKERRQ(ierr);
    ierr = PetscFree(ftable);CHKERRQ(ierr);
  }
  if (user->vtkInterval < 1) PetscFunctionReturn(0);
  if ((stepnum == -1) ^ (stepnum % user->vtkInterval == 0)) {
    if (stepnum == -1) {        /* Final time is not multiple of normal time interval, write it anyway */
      ierr = TSGetStepNumber(ts,&stepnum);CHKERRQ(ierr);
    }
    ierr = PetscSNPrintf(filename,sizeof filename,"./vtk_output/%s-%03D.vtu",user->outputBasename,stepnum);CHKERRQ(ierr);
    ierr = OutputVTK(dm,filename,&viewer);CHKERRQ(ierr);
    ierr = VecView(X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode initializeTS(DM dm, User user, TS *ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSCreate(PetscObjectComm((PetscObject)dm), ts);CHKERRQ(ierr);
  ierr = TSSetType(*ts, TSSSP);CHKERRQ(ierr);
  ierr = TSSetDM(*ts, dm);CHKERRQ(ierr);
  if (user->vtkmon) {
    ierr = TSMonitorSet(*ts,MonitorVTK,user,NULL);CHKERRQ(ierr);
  }
  ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, user);CHKERRQ(ierr);
  ierr = DMTSSetRHSFunctionLocal(dm, DMPlexTSComputeRHSFunctionFVM, user);CHKERRQ(ierr);
  ierr = TSSetMaxTime(*ts,2.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(*ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode adaptToleranceFVM(PetscFV fvm, TS ts, Vec sol, VecTagger refineTag, VecTagger coarsenTag, User user, TS *tsNew, Vec *solNew, FILE *cellfile)
{
  DM                dm, gradDM, cellDM;
  Vec               cellGeom, faceGeom;
  PetscBool         isForest, computeGradient;
  Vec               grad, locGrad, locX, errVec, grad_plot;
  PetscInt          cStart, cEnd, c, dim, nRefine, nCoarsen;
  PetscReal         minMaxInd[2] = {PETSC_MAX_REAL, PETSC_MIN_REAL}, minMaxIndGlobal[2], minInd, maxInd, time;
  PetscScalar       *errArray;
  const PetscScalar *pointVals;
  const PetscScalar *pointGrads;
  const PetscScalar *pointGeom;
  DMLabel           adaptLabel = NULL;
  IS                refineIS, coarsenIS;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = TSGetTime(ts,&time);CHKERRQ(ierr);
  ierr = VecGetDM(sol, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = PetscFVGetComputeGradients(fvm,&computeGradient);CHKERRQ(ierr);
  ierr = PetscFVSetComputeGradients(fvm,PETSC_TRUE);CHKERRQ(ierr);

  // Setting dm as the base DM
  DM preForest;
  ierr = DMConvert(dm, DMPLEX, &dm);CHKERRQ(ierr);
  ierr = DMCreate(PETSC_COMM_SELF, &preForest);CHKERRQ(ierr);
  ierr = DMSetType(preForest,(dim == 2) ? DMP4EST : DMP8EST);CHKERRQ(ierr);
  ierr = DMCopyDisc(dm,preForest);CHKERRQ(ierr);
  ierr = DMForestSetBaseDM(preForest,dm);CHKERRQ(ierr);
  ierr = DMForestSetMinimumRefinement(preForest,0);CHKERRQ(ierr);
  ierr = DMForestSetInitialRefinement(preForest,1);CHKERRQ(ierr);
  ierr = DMSetUp(preForest);CHKERRQ(ierr);
  // Transferring the vector from base dm to preForest
  Vec baseX, preX;
  ierr = DMGetGlobalVector(dm,&baseX);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(preForest, &preX);CHKERRQ(ierr);
  ierr = DMForestTransferVecFromBase(preForest, baseX, preX);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &baseX);CHKERRQ(ierr);

  // Adaption stage
  ierr = DMIsForest(preForest, &isForest);CHKERRQ(ierr);
  ierr = DMPlexGetDataFVM(preForest, fvm, &cellGeom, &faceGeom, &gradDM);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(preForest,&locX);CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(preForest, PETSC_TRUE, locX, 0.0, faceGeom, cellGeom, NULL);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(preForest, preX, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (preForest, preX, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(preForest, &preX);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(gradDM, &grad);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(preForest, &grad_plot);CHKERRQ(ierr);
  /* Reconstructing Gradients using dm, local vector and gradient vector */
  ierr = DMPlexReconstructGradientsFVM(preForest, locX, grad);CHKERRQ(ierr);
  ierr = DMPlexReconstructGradientsFVM(preForest, locX, grad_plot);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(gradDM, &locGrad);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(gradDM, grad, INSERT_VALUES, locGrad);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(gradDM, grad, INSERT_VALUES, locGrad);CHKERRQ(ierr);
  PetscInt stepnum;
  TSGetStepNumber(ts, &stepnum);
  OutputGrad(grad_plot, stepnum, preForest);
  VecDestroy(&grad_plot);
  VecDestroy(&grad);
  /* Only using the local gradients and local vector. */
  /* Obtaining the normal cell ranges for each processor */
  ierr = DMPlexGetInteriorCellStratum(preForest,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = VecGetArrayRead(locGrad,&pointGrads);CHKERRQ(ierr);
  ierr = VecGetArrayRead(cellGeom,&pointGeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(locX,&pointVals);CHKERRQ(ierr);
  /* Getting the cell dm from the vec obtained from the FV object */
  ierr = VecGetDM(cellGeom,&cellDM);CHKERRQ(ierr);
  ierr = DMLabelCreate(PETSC_COMM_SELF,"adapt",&adaptLabel);CHKERRQ(ierr);
  ierr = VecCreateMPI(PetscObjectComm((PetscObject)preForest),cEnd-cStart,PETSC_DETERMINE,&errVec);CHKERRQ(ierr);
  ierr = VecSetUp(errVec);CHKERRQ(ierr);
  ierr = VecGetArray(errVec,&errArray);CHKERRQ(ierr);

  /* Finding the maximum gradient */
  PetscReal MaxGrad = -1000.0;
  PetscReal errInd = 0.0;
  PetscBool criteria2 = PETSC_FALSE, criteria3 = PETSC_FALSE;
  PetscOptionsGetBool(NULL, NULL, "-criteria_height", &criteria2, NULL);
  PetscOptionsGetBool(NULL, NULL, "-criteria_average", &criteria3, NULL);

  for (c = cStart; c < cEnd; c++){
      PetscScalar           *pointGrad;
      PetscFVCellGeom       *cg;
      PetscScalar           *pointVal;

      ierr = DMPlexPointLocalRead(cellDM,c,pointGeom,&cg);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(gradDM,c,pointGrads,&pointGrad);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(preForest,c,pointVals,&pointVal);CHKERRQ(ierr);

      if(criteria2) {
          ierr = ErrorIndicator_Height(dim, cg->volume, pointGrad, &errInd);
          CHKERRQ(ierr);
      }
      else if(criteria3){
          ierr = ErrorIndicator_Average(dim, cg->volume, pointGrad, &errInd);
          CHKERRQ(ierr);
      }
      else {
          ierr = (user->model->errorIndicator)(dim, cg->volume, user->model->physics->dof, pointVal, pointGrad, &errInd,
                                               user->model->errorCtx);
          CHKERRQ(ierr);
      }
      if (errInd >= MaxGrad)   MaxGrad = errInd;
  }
  if (MaxGrad == -1000.0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Maximum gradient of %d has not been changed.", c);
//  printf("Max grad is %f\n", MaxGrad);

  for (c = cStart; c < cEnd; c++) {
    PetscReal             errInd = 0.;
    PetscScalar           *pointGrad;
    PetscScalar           *pointVal;
    PetscFVCellGeom       *cg;

    /* Obtaining the gradient, geometry and value at each cell */
    ierr = DMPlexPointLocalRead(gradDM,c,pointGrads,&pointGrad);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(cellDM,c,pointGeom,&cg);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(preForest,c,pointVals,&pointVal);CHKERRQ(ierr);

    /* Getting the adapting criteria as the product of volume and norm of the gradients */
    if(criteria2) {
          ierr = ErrorIndicator_Height(dim, cg->volume, pointGrad, &errInd);
          CHKERRQ(ierr);
    }
    else if(criteria3){
        ierr = ErrorIndicator_Average(dim, cg->volume, pointGrad, &errInd);
        CHKERRQ(ierr);
    }
    else {
        ierr = (user->model->errorIndicator)(dim, cg->volume, user->model->physics->dof, pointVal, pointGrad, &errInd,
                                               user->model->errorCtx);
        CHKERRQ(ierr);
    }
    errArray[c-cStart] = errInd/MaxGrad;
    /* To prevent from obtaining NaN values */
    minMaxInd[0] = PetscMin(minMaxInd[0],errInd);
    minMaxInd[1] = PetscMax(minMaxInd[1],errInd);
  }
  ierr = VecRestoreArray(errVec,&errArray);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(locX,&pointVals);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(cellGeom,&pointGeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(locGrad,&pointGrads);CHKERRQ(ierr);
  ierr = VecDestroy(&locGrad);CHKERRQ(ierr);
  ierr = VecDestroy(&locX);CHKERRQ(ierr);

  /* Provide the function with the Tag, the norms in the cells, and resulting tags */
  ierr = VecViewFromOptions(errVec, NULL, "-error_vec");CHKERRQ(ierr);
  ierr = VecTaggerComputeIS(refineTag,errVec,&refineIS);CHKERRQ(ierr);
  ierr = VecTaggerComputeIS(coarsenTag,errVec,&coarsenIS);CHKERRQ(ierr);
  ierr = ISGetSize(refineIS,&nRefine);CHKERRQ(ierr);
  ierr = ISGetSize(coarsenIS,&nCoarsen);CHKERRQ(ierr);
  if (nRefine) {ierr = DMLabelSetStratumIS(adaptLabel,DM_ADAPT_REFINE,refineIS);CHKERRQ(ierr);}
  if (nCoarsen) {ierr = DMLabelSetStratumIS(adaptLabel,DM_ADAPT_COARSEN,coarsenIS);CHKERRQ(ierr);}
  ierr = ISDestroy(&coarsenIS);CHKERRQ(ierr);
  ierr = ISDestroy(&refineIS);CHKERRQ(ierr);
  ierr = VecDestroy(&errVec);CHKERRQ(ierr);

  ierr = PetscFVSetComputeGradients(fvm,computeGradient);CHKERRQ(ierr);
  minMaxInd[1] = -minMaxInd[1];
  ierr = MPI_Allreduce(minMaxInd,minMaxIndGlobal,2,MPIU_REAL,MPI_MIN,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  minInd = minMaxIndGlobal[0];
  maxInd = -minMaxIndGlobal[1];
  ierr = PetscInfo2(ts, "error indicator range (%E, %E)\n", minInd, maxInd);CHKERRQ(ierr);

  DM postForest=NULL;

  if (nRefine || nCoarsen) { /* at least one cell is over the refinement threshold */
      /* Converting/adapting the DM from the old mesh to the new mesh */
    ierr = DMForestTemplate(preForest, PETSC_COMM_SELF, &postForest);CHKERRQ(ierr);
    ierr = DMForestSetAdaptivityLabel(postForest, adaptLabel);CHKERRQ(ierr);
    ierr = DMSetUp(postForest);CHKERRQ(ierr);
  }
  ierr = DMLabelDestroy(&adaptLabel);CHKERRQ(ierr);
  if (postForest) {
    ierr = PetscInfo2(ts, "Adapted mesh, marking %D cells for refinement, and %D cells for coarsening\n", nRefine, nCoarsen);CHKERRQ(ierr);
    if (tsNew) {ierr = initializeTS(postForest, user, tsNew);CHKERRQ(ierr);}
    if (solNew) {
      /* Creating an empty solution vector with the same size as the adoptedDM */
      ierr = DMCreateGlobalVector(postForest, solNew);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) *solNew, "solution");CHKERRQ(ierr);
        /* Transferring/Interpolating the solution to the new dm (persumably cell centers) */
      ierr = DMGetGlobalVector(preForest, &preX);
      ierr = DMForestTransferVec(preForest, preX, postForest, *solNew, PETSC_TRUE, time);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(preForest, &preX);CHKERRQ(ierr);
    }
    /* clear internal references to the previous dm */
    ierr = DMDestroy(&preForest);CHKERRQ(ierr);
    if (isForest) {
        ierr = DMForestSetAdaptivityForest(postForest,NULL);CHKERRQ(ierr);
    }
    ierr = DMDestroy(&postForest);CHKERRQ(ierr);
  } else {
    if (tsNew)  *tsNew  = NULL;
    if (solNew) *solNew = NULL;
  }
  fprintf(cellfile, "%d\n", (cEnd-cStart));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm          comm;
  PetscDS           prob;
  PetscFV           fvm;
  PetscLimiter      limiter = NULL, noneLimiter = NULL;
  User              user;
  Model             mod;
  Physics           phys;
  DM                dm;
  PetscReal         ftime, cfl, dt, minRadius;
  PetscInt          dim, nsteps;
  TS                ts;
  TSConvergedReason reason;
  Vec               X;
  PetscViewer       viewer;
  PetscBool         simplex = PETSC_FALSE, vtkCellGeom, splitFaces, useAMR;
  PetscInt          overlap, adaptInterval;
  char              filename[PETSC_MAX_PATH_LEN] = "";
  char              physname[256]  = "advect";
  VecTagger         refineTag = NULL, coarsenTag = NULL;
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc, &argv, (char*) 0, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  ierr          = PetscNew(&user);CHKERRQ(ierr);
  ierr          = PetscNew(&user->model);CHKERRQ(ierr);
  ierr          = PetscNew(&user->model->physics);CHKERRQ(ierr);
  mod           = user->model;
  phys          = mod->physics;
  mod->comm     = comm;
  useAMR        = PETSC_FALSE;
  adaptInterval = 1;

  /* Register physical models to be available on the command line */
  ierr = PetscFunctionListAdd(&PhysicsList,"advect"          ,PhysicsCreate_Advect);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PhysicsList,"sw"              ,PhysicsCreate_SW);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PhysicsList,"euler"           ,PhysicsCreate_Euler);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm,NULL,"Unstructured Finite Volume Mesh Options","");CHKERRQ(ierr);
  {
    cfl  = 0.9 * 4; /* default SSPRKS2 with s=5 stages is stable for CFL number s-1 */
    ierr = PetscOptionsReal("-ufv_cfl","CFL number per step","",cfl,&cfl,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-f","Exodus.II filename to read","",filename,filename,sizeof(filename),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-simplex","Flag to use a simplex mesh","",simplex,&simplex,NULL);CHKERRQ(ierr);
    splitFaces = PETSC_FALSE;
    ierr = PetscOptionsBool("-ufv_split_faces","Split faces between cell sets","",splitFaces,&splitFaces,NULL);CHKERRQ(ierr);
    overlap = 1;
    ierr = PetscOptionsInt("-ufv_mesh_overlap","Number of cells to overlap partitions","",overlap,&overlap,NULL);CHKERRQ(ierr);
    user->vtkInterval = 1;
    ierr = PetscOptionsInt("-ufv_vtk_interval","VTK output interval (0 to disable)","",user->vtkInterval,&user->vtkInterval,NULL);CHKERRQ(ierr);
    user->vtkmon = PETSC_TRUE;
    ierr = PetscOptionsBool("-ufv_vtk_monitor","Use VTKMonitor routine","",user->vtkmon,&user->vtkmon,NULL);CHKERRQ(ierr);
    vtkCellGeom = PETSC_FALSE;
    ierr = PetscStrcpy(user->outputBasename, "ex11");CHKERRQ(ierr);
    ierr = PetscOptionsString("-ufv_vtk_basename","VTK output basename","",user->outputBasename,user->outputBasename,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-ufv_vtk_cellgeom","Write cell geometry (for debugging)","",vtkCellGeom,&vtkCellGeom,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-ufv_use_amr","use local adaptive mesh refinement","",useAMR,&useAMR,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ufv_adapt_interval","time steps between AMR","",adaptInterval,&adaptInterval,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (useAMR) {
    VecTaggerBox refineBox, coarsenBox;

    refineBox.min  = refineBox.max  = PETSC_MAX_REAL;
    coarsenBox.min = coarsenBox.max = PETSC_MIN_REAL;

    // coarsen: -inf to 0.1 and refine: 0.9 to inf
    ierr = VecTaggerCreate(comm,&refineTag);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)refineTag,"refine_");CHKERRQ(ierr);
    ierr = VecTaggerSetType(refineTag,VECTAGGERABSOLUTE);CHKERRQ(ierr);
    ierr = VecTaggerAbsoluteSetBox(refineTag,&refineBox);CHKERRQ(ierr);
    ierr = VecTaggerSetFromOptions(refineTag);CHKERRQ(ierr);
    ierr = VecTaggerSetUp(refineTag);CHKERRQ(ierr);
    ierr = PetscObjectViewFromOptions((PetscObject)refineTag,NULL,"-tag_view");CHKERRQ(ierr);

    ierr = VecTaggerCreate(comm,&coarsenTag);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)coarsenTag,"coarsen_");CHKERRQ(ierr);
    ierr = VecTaggerSetType(coarsenTag,VECTAGGERABSOLUTE);CHKERRQ(ierr);
    ierr = VecTaggerAbsoluteSetBox(coarsenTag,&coarsenBox);CHKERRQ(ierr);
    ierr = VecTaggerSetFromOptions(coarsenTag);CHKERRQ(ierr);
    ierr = VecTaggerSetUp(coarsenTag);CHKERRQ(ierr);
    ierr = PetscObjectViewFromOptions((PetscObject)coarsenTag,NULL,"-tag_view");CHKERRQ(ierr);
  }

  ierr = PetscOptionsBegin(comm,NULL,"Unstructured Finite Volume Physics Options","");CHKERRQ(ierr);
  {
    PetscErrorCode (*physcreate)(Model,Physics,PetscOptionItems*);
    ierr = PetscOptionsFList("-physics","Physics module to solve","",PhysicsList,physname,physname,sizeof physname,NULL);CHKERRQ(ierr);
    ierr = PetscFunctionListFind(PhysicsList,physname,&physcreate);CHKERRQ(ierr);
    ierr = PetscMemzero(phys,sizeof(struct _n_Physics));CHKERRQ(ierr);
    ierr = (*physcreate)(mod,phys,PetscOptionsObject);CHKERRQ(ierr);
    /* Count number of fields and dofs */
    for (phys->nfields=0,phys->dof=0; phys->field_desc[phys->nfields].name; phys->nfields++) phys->dof += phys->field_desc[phys->nfields].dof;
    if (phys->dof <= 0) SETERRQ1(comm,PETSC_ERR_ARG_WRONGSTATE,"Physics '%s' did not set dof",physname);
    ierr = ModelFunctionalSetFromOptions(mod,PetscOptionsObject);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Create mesh */
  {
    size_t len,i;
    for (i = 0; i < DIM; i++) { mod->bounds[2*i] = 0.; mod->bounds[2*i+1] = 1.;};
    ierr = PetscStrlen(filename,&len);CHKERRQ(ierr);
    dim = DIM;
    if (!len) { /* a null name means just do a hex box */
      PetscInt cells[3] = {1, 1, 1}; /* coarse mesh is one cell; refine from there */
      PetscBool flg1, flg2, skew = PETSC_FALSE;
      PetscInt nret1 = DIM;
      PetscInt nret2 = 2*DIM;
      ierr = PetscOptionsBegin(comm,NULL,"Rectangular mesh options","");CHKERRQ(ierr);
      ierr = PetscOptionsIntArray("-grid_size","number of cells in each direction","",cells,&nret1,&flg1);CHKERRQ(ierr);
      ierr = PetscOptionsRealArray("-grid_bounds","bounds of the mesh in each direction (i.e., x_min,x_max,y_min,y_max","",mod->bounds,&nret2,&flg2);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-grid_skew_60","Skew grid for 60 degree shock mesh","",skew,&skew,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsEnd();CHKERRQ(ierr);
      if (flg1) {
        dim = nret1;
        if (dim != DIM) SETERRQ1(comm,PETSC_ERR_ARG_SIZ,"Dim wrong size %D in -grid_size",dim);
      }
      ierr = DMPlexCreateBoxMesh(comm, dim, simplex, cells, NULL, NULL, mod->bcs, PETSC_TRUE, &dm);CHKERRQ(ierr);
      if (flg2) {
        PetscInt dimEmbed, i;
        PetscInt nCoords;
        PetscScalar *coords;
        Vec coordinates;

        ierr = DMGetCoordinatesLocal(dm,&coordinates);CHKERRQ(ierr);
        ierr = DMGetCoordinateDim(dm,&dimEmbed);CHKERRQ(ierr);
        ierr = VecGetLocalSize(coordinates,&nCoords);CHKERRQ(ierr);
        if (nCoords % dimEmbed) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Coordinate vector the wrong size");
        ierr = VecGetArray(coordinates,&coords);CHKERRQ(ierr);
        for (i = 0; i < nCoords; i += dimEmbed) {
          PetscInt j;

          PetscScalar *coord = &coords[i];
          for (j = 0; j < dimEmbed; j++) {
            coord[j] = mod->bounds[2 * j] + coord[j] * (mod->bounds[2 * j + 1] - mod->bounds[2 * j]);
            if (dim==2 && cells[1]==1 && j==0 && skew) {
              if (cells[0]==2 && i==8) {
                coord[j] = .57735026918963; /* hack to get 60 deg skewed mesh */
              }
              else if (cells[0]==3) {
                if(i==2 || i==10) coord[j] = mod->bounds[1]/4.;
                else if (i==4) coord[j] = mod->bounds[1]/2.;
                else if (i==12) coord[j] = 1.57735026918963*mod->bounds[1]/2.;
              }
            }
          }
        }
        ierr = VecRestoreArray(coordinates,&coords);CHKERRQ(ierr);
        ierr = DMSetCoordinatesLocal(dm,coordinates);CHKERRQ(ierr);
      }
    } else {
      ierr = DMPlexCreateFromFile(comm, filename, PETSC_TRUE, &dm);CHKERRQ(ierr);
  }
  }
  ierr = DMViewFromOptions(dm, NULL, "-orig_dm_view");CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

  /* set up BCs, functions, tags */
  ierr = DMCreateLabel(dm, "Face Sets");CHKERRQ(ierr);
  mod->errorIndicator = ErrorIndicator_Simple;

  {
    DM dmDist;

    ierr = DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMPlexDistribute(dm, overlap, NULL, &dmDist);CHKERRQ(ierr);
    if (dmDist) {
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
      dm   = dmDist;
    }
  }

  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);

  {
    DM gdm;

    ierr = DMPlexConstructGhostCells(dm, NULL, NULL, &gdm);CHKERRQ(ierr);
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm   = gdm;
    ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  }

  if (splitFaces) {
      ierr = ConstructCellBoundary(dm, user);CHKERRQ(ierr);
  }

  ierr = SplitFaces(&dm, "split faces", user);CHKERRQ(ierr);

  ierr = PetscFVCreate(comm, &fvm);CHKERRQ(ierr);
  ierr = PetscFVSetFromOptions(fvm);CHKERRQ(ierr);
  ierr = PetscFVSetNumComponents(fvm, phys->dof);CHKERRQ(ierr);
  ierr = PetscFVSetSpatialDimension(fvm, dim);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fvm,"");CHKERRQ(ierr);
  {
    PetscInt f, dof;
    for (f=0,dof=0; f < phys->nfields; f++) {
      PetscInt newDof = phys->field_desc[f].dof;

      if (newDof == 1) {
        ierr = PetscFVSetComponentName(fvm,dof,phys->field_desc[f].name);CHKERRQ(ierr);
      }
      else {
        PetscInt j;

        for (j = 0; j < newDof; j++) {
          char     compName[256]  = "Unknown";

          ierr = PetscSNPrintf(compName,sizeof(compName),"%s_%d",phys->field_desc[f].name,j);CHKERRQ(ierr);
          ierr = PetscFVSetComponentName(fvm,dof+j,compName);CHKERRQ(ierr);
        }
      }
      dof += newDof;
    }
  }
  /* FV is now structured with one field having all physics as components
   * Adding the field using the FVM object which has number of components as dof*/
  ierr = DMAddField(dm, NULL, (PetscObject) fvm);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetRiemannSolver(prob, 0, user->model->physics->riemann);CHKERRQ(ierr);
  ierr = PetscDSSetContext(prob, 0, user->model->physics);CHKERRQ(ierr);
  ierr = (*mod->setupbc)(prob,phys);CHKERRQ(ierr);
  ierr = PetscDSSetFromOptions(prob);CHKERRQ(ierr);
  {
    char      convType[256];
    PetscBool flg;

    ierr = PetscOptionsBegin(comm, "", "Mesh conversion options", "DMPLEX");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-dm_type","Convert DMPlex to another format","ex12",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (flg) {
      DM dmConv;

      ierr = DMConvert(dm,convType,&dmConv);CHKERRQ(ierr);
      if (dmConv) {
        ierr = DMViewFromOptions(dmConv, NULL, "-dm_conv_view");CHKERRQ(ierr);
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        dm   = dmConv;
        ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
      }
    }
  }

  ierr = initializeTS(dm, user, &ts);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &X);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) X, "solution");CHKERRQ(ierr);
  ierr = SetInitialCondition(dm, X, user);CHKERRQ(ierr);

  PetscInt adaptLimit;
  PetscOptionsGetInt(NULL, NULL, "-adapt_limit", &adaptLimit, NULL);

  char textfilename[256];
  ierr = PetscSNPrintf(textfilename,sizeof textfilename,"./cell_count/adaptIter_%d.dat",adaptLimit);CHKERRQ(ierr);
  FILE *cellfile = fopen(textfilename, "w");

  if (useAMR) {
    PetscInt adaptIter;

    /* use no limiting when reconstructing gradients for adaptivity */
    ierr = PetscFVGetLimiter(fvm, &limiter);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject) limiter);CHKERRQ(ierr);
    ierr = PetscLimiterCreate(PetscObjectComm((PetscObject) fvm), &noneLimiter);CHKERRQ(ierr);
    ierr = PetscLimiterSetType(noneLimiter, PETSCLIMITERNONE);CHKERRQ(ierr);

    ierr = PetscFVSetLimiter(fvm, noneLimiter);CHKERRQ(ierr);
    for (adaptIter = 0; adaptIter<adaptLimit; ++adaptIter) {
      PetscLogDouble bytes;
      TS             tsNew = NULL;
      printf("Pre-adapt iter: %d \n", adaptIter);
      ierr = PetscMemoryGetCurrentUsage(&bytes);CHKERRQ(ierr);
      ierr = PetscInfo2(ts, "refinement loop %D: memory used %g\n", adaptIter, bytes);CHKERRQ(ierr);
      ierr = DMViewFromOptions(dm, NULL, "-initial_dm_view");CHKERRQ(ierr);
      ierr = VecViewFromOptions(X, NULL, "-initial_vec_view");CHKERRQ(ierr);
#if 0
      if (viewInitial) {
        PetscViewer viewer;
        char        buf[256];
        PetscBool   isHDF5, isVTK;

        ierr = PetscViewerCreate(comm,&viewer);CHKERRQ(ierr);
        ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
        ierr = PetscViewerSetOptionsPrefix(viewer,"initial_");CHKERRQ(ierr);
        ierr = PetscViewerSetFromOptions(viewer);CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&isHDF5);CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERVTK,&isVTK);CHKERRQ(ierr);
        if (isHDF5) {
          ierr = PetscSNPrintf(buf, 256, "ex11-initial-%d.h5", adaptIter);CHKERRQ(ierr);
        } else if (isVTK) {
          ierr = PetscSNPrintf(buf, 256, "ex11-initial-%d.vtu", adaptIter);CHKERRQ(ierr);
          ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_VTK_VTU);CHKERRQ(ierr);
        }
        ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
        ierr = PetscViewerFileSetName(viewer,buf);CHKERRQ(ierr);
        if (isHDF5) {
          ierr = DMView(dm,viewer);CHKERRQ(ierr);
          ierr = PetscViewerFileSetMode(viewer,FILE_MODE_UPDATE);CHKERRQ(ierr);
        }
        ierr = VecView(X,viewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      }
#endif
      fprintf(cellfile, "0.0,");
      ierr = adaptToleranceFVM(fvm, ts, X, refineTag, coarsenTag, user, &tsNew, NULL, cellfile);CHKERRQ(ierr);
      if (!tsNew) {
        break;
      } else {
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        ierr = VecDestroy(&X);CHKERRQ(ierr);
        ierr = TSDestroy(&ts);CHKERRQ(ierr);
        ts   = tsNew;
        ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)dm);CHKERRQ(ierr);
        ierr = DMCreateGlobalVector(dm,&X);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject) X, "solution");CHKERRQ(ierr);
        ierr = SetInitialCondition(dm, X, user);CHKERRQ(ierr);
      }
    }
    /* restore original limiter */
    ierr = PetscFVSetLimiter(fvm, limiter);CHKERRQ(ierr);
  }

  if (vtkCellGeom) {
    DM  dmCell;
    Vec cellgeom, partition;
    ierr = DMPlexTSGetGeometryFVM(dm, NULL, &cellgeom, NULL);CHKERRQ(ierr);
    ierr = OutputVTK(dm, "ex11-cellgeom.vtk", &viewer);CHKERRQ(ierr);
    ierr = VecView(cellgeom, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = CreatePartitionVec(dm, &dmCell, &partition);CHKERRQ(ierr);
    ierr = OutputVTK(dmCell, "ex11-partition.vtk", &viewer);CHKERRQ(ierr);
    ierr = VecView(partition, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = VecDestroy(&partition);CHKERRQ(ierr);
    ierr = DMDestroy(&dmCell);CHKERRQ(ierr);
  }

  /* collect max maxspeed from all processes -- todo */
  /* Provides the mesh width of the smallest cell */
  ierr = DMPlexTSGetGeometryFVM(dm, NULL, NULL, &minRadius);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&phys->maxspeed,&mod->maxspeed,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)ts));CHKERRQ(ierr);
  if (mod->maxspeed <= 0) SETERRQ1(comm,PETSC_ERR_ARG_WRONGSTATE,"Physics '%s' did not set maxspeed",physname);
  dt   = cfl * minRadius / mod->maxspeed;
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  if (!useAMR) {
    ierr = TSSolve(ts,X);CHKERRQ(ierr);
    ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
    ierr = TSGetStepNumber(ts,&nsteps);CHKERRQ(ierr);
  } else {
    PetscReal finalTime;
    PetscInt  adaptIter;
    TS        tsNew = NULL;
    Vec       solNew = NULL;

    ierr   = TSGetMaxTime(ts,&finalTime);CHKERRQ(ierr);
    ierr   = TSSetMaxSteps(ts,adaptInterval);CHKERRQ(ierr);
    ierr   = TSSolve(ts,X);CHKERRQ(ierr);
    ierr   = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
    ierr   = TSGetStepNumber(ts,&nsteps);CHKERRQ(ierr);
    for (adaptIter = 0;ftime < finalTime;adaptIter++) {
      PetscLogDouble bytes;
      ierr = PetscMemoryGetCurrentUsage(&bytes);CHKERRQ(ierr);
      ierr = PetscInfo2(ts, "AMR time step loop %D: memory used %g\n", adaptIter, bytes);CHKERRQ(ierr);
      ierr = PetscFVSetLimiter(fvm,noneLimiter);CHKERRQ(ierr);
      fprintf(cellfile, "%f,", ftime);
      ierr = adaptToleranceFVM(fvm,ts,X,refineTag,coarsenTag,user,&tsNew,&solNew, cellfile);CHKERRQ(ierr);
      ierr = PetscFVSetLimiter(fvm,limiter);CHKERRQ(ierr);
      if (tsNew) {
        ierr = PetscInfo(ts, "AMR used\n");CHKERRQ(ierr);
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        ierr = VecDestroy(&X);CHKERRQ(ierr);
        ierr = TSDestroy(&ts);CHKERRQ(ierr);
        ts   = tsNew;
        X    = solNew;
        ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
        ierr = VecGetDM(X,&dm);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)dm);CHKERRQ(ierr);
        ierr = DMPlexTSGetGeometryFVM(dm, NULL, NULL, &minRadius);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&phys->maxspeed,&mod->maxspeed,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)ts));CHKERRQ(ierr);
        if (mod->maxspeed <= 0) SETERRQ1(comm,PETSC_ERR_ARG_WRONGSTATE,"Physics '%s' did not set maxspeed",physname);
        dt   = cfl * minRadius / mod->maxspeed;
        ierr = TSSetStepNumber(ts,nsteps);CHKERRQ(ierr);
        ierr = TSSetTime(ts,ftime);CHKERRQ(ierr);
        ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
      } else {
        ierr = PetscInfo(ts, "AMR not used\n");CHKERRQ(ierr);
      }
      user->monitorStepOffset = nsteps;
      ierr = TSSetMaxSteps(ts,nsteps+adaptInterval);CHKERRQ(ierr);
      ierr = TSSolve(ts,X);CHKERRQ(ierr);
      ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
      ierr = TSGetStepNumber(ts,&nsteps);CHKERRQ(ierr);
    }
  }

  fclose(cellfile);

  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s at time %g after %D steps\n",TSConvergedReasons[reason],(double)ftime,nsteps);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = VecTaggerDestroy(&refineTag);CHKERRQ(ierr);
  ierr = VecTaggerDestroy(&coarsenTag);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&PhysicsList);CHKERRQ(ierr);
  ierr = FunctionalLinkDestroy(&user->model->functionalRegistry);CHKERRQ(ierr);
  ierr = PetscFree(user->model->functionalMonitored);CHKERRQ(ierr);
  ierr = PetscFree(user->model->functionalCall);CHKERRQ(ierr);
  ierr = PetscFree(user->model->physics->data);CHKERRQ(ierr);
  ierr = PetscFree(user->model->physics);CHKERRQ(ierr);
  ierr = PetscFree(user->model);CHKERRQ(ierr);
  ierr = PetscFree(user);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = PetscLimiterDestroy(&limiter);CHKERRQ(ierr);
  ierr = PetscLimiterDestroy(&noneLimiter);CHKERRQ(ierr);
  ierr = PetscFVDestroy(&fvm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/* Godunov fluxs */
PetscScalar cvmgp_(PetscScalar *a, PetscScalar *b, PetscScalar *test)
{
    /* System generated locals */
    PetscScalar ret_val;

    if (PetscRealPart(*test) > 0.) {
	goto L10;
    }
    ret_val = *b;
    return ret_val;
L10:
    ret_val = *a;
    return ret_val;
} /* cvmgp_ */

PetscScalar cvmgm_(PetscScalar *a, PetscScalar *b, PetscScalar *test)
{
    /* System generated locals */
    PetscScalar ret_val;

    if (PetscRealPart(*test) < 0.) {
	goto L10;
    }
    ret_val = *b;
    return ret_val;
L10:
    ret_val = *a;
    return ret_val;
} /* cvmgm_ */

int riem1mdt( PetscScalar *gaml, PetscScalar *gamr, PetscScalar *rl, PetscScalar *pl,
              PetscScalar *uxl, PetscScalar *rr, PetscScalar *pr,
              PetscScalar *uxr, PetscScalar *rstarl, PetscScalar *rstarr, PetscScalar *
              pstar, PetscScalar *ustar)
{
    /* Initialized data */

    static PetscScalar smallp = 1e-8;

    /* System generated locals */
    int i__1;
    PetscScalar d__1, d__2;

    /* Local variables */
    static int i0;
    static PetscScalar cl, cr, wl, zl, wr, zr, pst, durl, skpr1, skpr2;
    static int iwave;
    static PetscScalar gascl4, gascr4, cstarl, dpstar, cstarr;
    /* static PetscScalar csqrl, csqrr, gascl1, gascl2, gascl3, gascr1, gascr2, gascr3; */
    static int iterno;
    static PetscScalar ustarl, ustarr, rarepr1, rarepr2;



    /* gascl1 = *gaml - 1.; */
    /* gascl2 = (*gaml + 1.) * .5; */
    /* gascl3 = gascl2 / *gaml; */
    gascl4 = 1. / (*gaml - 1.);

    /* gascr1 = *gamr - 1.; */
    /* gascr2 = (*gamr + 1.) * .5; */
    /* gascr3 = gascr2 / *gamr; */
    gascr4 = 1. / (*gamr - 1.);
    iterno = 10;
/*        find pstar: */
    cl = PetscSqrtScalar(*gaml * *pl / *rl);
    cr = PetscSqrtScalar(*gamr * *pr / *rr);
    wl = *rl * cl;
    wr = *rr * cr;
    /* csqrl = wl * wl; */
    /* csqrr = wr * wr; */
    *pstar = (wl * *pr + wr * *pl) / (wl + wr);
    *pstar = PetscMax(PetscRealPart(*pstar),PetscRealPart(smallp));
    pst = *pl / *pr;
    skpr1 = cr * (pst - 1.) * PetscSqrtScalar(2. / (*gamr * (*gamr - 1. + (*gamr + 1.) * pst)));
    d__1 = (*gamr - 1.) / (*gamr * 2.);
    rarepr2 = gascr4 * 2. * cr * (1. - PetscPowScalar(pst, d__1));
    pst = *pr / *pl;
    skpr2 = cl * (pst - 1.) * PetscSqrtScalar(2. / (*gaml * (*gaml - 1. + (*gaml + 1.) * pst)));
    d__1 = (*gaml - 1.) / (*gaml * 2.);
    rarepr1 = gascl4 * 2. * cl * (1. - PetscPowScalar(pst, d__1));
    durl = *uxr - *uxl;
    if (PetscRealPart(*pr) < PetscRealPart(*pl)) {
	if (PetscRealPart(durl) >= PetscRealPart(rarepr1)) {
	    iwave = 100;
	} else if (PetscRealPart(durl) <= PetscRealPart(-skpr1)) {
	    iwave = 300;
	} else {
	    iwave = 400;
	}
    } else {
	if (PetscRealPart(durl) >= PetscRealPart(rarepr2)) {
	    iwave = 100;
	} else if (PetscRealPart(durl) <= PetscRealPart(-skpr2)) {
	    iwave = 300;
	} else {
	    iwave = 200;
	}
    }
    if (iwave == 100) {
/*     1-wave: rarefaction wave, 3-wave: rarefaction wave */
/*     case (100) */
	i__1 = iterno;
	for (i0 = 1; i0 <= i__1; ++i0) {
	    d__1 = *pstar / *pl;
	    d__2 = 1. / *gaml;
	    *rstarl = *rl * PetscPowScalar(d__1, d__2);
	    cstarl = PetscSqrtScalar(*gaml * *pstar / *rstarl);
	    ustarl = *uxl - gascl4 * 2. * (cstarl - cl);
	    zl = *rstarl * cstarl;
	    d__1 = *pstar / *pr;
	    d__2 = 1. / *gamr;
	    *rstarr = *rr * PetscPowScalar(d__1, d__2);
	    cstarr = PetscSqrtScalar(*gamr * *pstar / *rstarr);
	    ustarr = *uxr + gascr4 * 2. * (cstarr - cr);
	    zr = *rstarr * cstarr;
	    dpstar = zl * zr * (ustarr - ustarl) / (zl + zr);
	    *pstar -= dpstar;
	    *pstar = PetscMax(PetscRealPart(*pstar),PetscRealPart(smallp));
	    if (PetscAbsScalar(dpstar) / PetscRealPart(*pstar) <= 1e-8) {
#if 0
        break;
#endif
	    }
	}
/*     1-wave: shock wave, 3-wave: rarefaction wave */
    } else if (iwave == 200) {
/*     case (200) */
	i__1 = iterno;
	for (i0 = 1; i0 <= i__1; ++i0) {
	    pst = *pstar / *pl;
	    ustarl = *uxl - (pst - 1.) * cl * PetscSqrtScalar(2. / (*gaml * (*gaml - 1. + (*gaml + 1.) * pst)));
	    zl = *pl / cl * PetscSqrtScalar(*gaml * 2. * (*gaml - 1. + (*gaml + 1.) * pst)) * (*gaml - 1. + (*gaml + 1.) * pst) / (*gaml * 3. - 1. + (*gaml + 1.) * pst);
	    d__1 = *pstar / *pr;
	    d__2 = 1. / *gamr;
	    *rstarr = *rr * PetscPowScalar(d__1, d__2);
	    cstarr = PetscSqrtScalar(*gamr * *pstar / *rstarr);
	    zr = *rstarr * cstarr;
	    ustarr = *uxr + gascr4 * 2. * (cstarr - cr);
	    dpstar = zl * zr * (ustarr - ustarl) / (zl + zr);
	    *pstar -= dpstar;
	    *pstar = PetscMax(PetscRealPart(*pstar),PetscRealPart(smallp));
	    if (PetscAbsScalar(dpstar) / PetscRealPart(*pstar) <= 1e-8) {
#if 0
        break;
#endif
	    }
	}
/*     1-wave: shock wave, 3-wave: shock */
    } else if (iwave == 300) {
/*     case (300) */
	i__1 = iterno;
	for (i0 = 1; i0 <= i__1; ++i0) {
	    pst = *pstar / *pl;
	    ustarl = *uxl - (pst - 1.) * cl * PetscSqrtScalar(2. / (*gaml * (*gaml - 1. + (*gaml + 1.) * pst)));
	    zl = *pl / cl * PetscSqrtScalar(*gaml * 2. * (*gaml - 1. + (*gaml + 1.) * pst)) * (*gaml - 1. + (*gaml + 1.) * pst) / (*gaml * 3. - 1. + (*gaml + 1.) * pst);
	    pst = *pstar / *pr;
	    ustarr = *uxr + (pst - 1.) * cr * PetscSqrtScalar(2. / (*gamr * (*gamr - 1. + (*gamr + 1.) * pst)));
	    zr = *pr / cr * PetscSqrtScalar(*gamr * 2. * (*gamr - 1. + (*gamr + 1.) * pst)) * (*gamr - 1. + (*gamr + 1.) * pst) / (*gamr * 3. - 1. + (*gamr + 1.) * pst);
	    dpstar = zl * zr * (ustarr - ustarl) / (zl + zr);
	    *pstar -= dpstar;
	    *pstar = PetscMax(PetscRealPart(*pstar),PetscRealPart(smallp));
	    if (PetscAbsScalar(dpstar) / PetscRealPart(*pstar) <= 1e-8) {
#if 0
        break;
#endif
	    }
	}
/*     1-wave: rarefaction wave, 3-wave: shock */
    } else if (iwave == 400) {
/*     case (400) */
	i__1 = iterno;
	for (i0 = 1; i0 <= i__1; ++i0) {
	    d__1 = *pstar / *pl;
	    d__2 = 1. / *gaml;
	    *rstarl = *rl * PetscPowScalar(d__1, d__2);
	    cstarl = PetscSqrtScalar(*gaml * *pstar / *rstarl);
	    ustarl = *uxl - gascl4 * 2. * (cstarl - cl);
	    zl = *rstarl * cstarl;
	    pst = *pstar / *pr;
	    ustarr = *uxr + (pst - 1.) * cr * PetscSqrtScalar(2. / (*gamr * (*gamr - 1. + (*gamr + 1.) * pst)));
	    zr = *pr / cr * PetscSqrtScalar(*gamr * 2. * (*gamr - 1. + (*gamr + 1.) * pst)) * (*gamr - 1. + (*gamr + 1.) * pst) / (*gamr * 3. - 1. + (*gamr + 1.) * pst);
	    dpstar = zl * zr * (ustarr - ustarl) / (zl + zr);
	    *pstar -= dpstar;
	    *pstar = PetscMax(PetscRealPart(*pstar),PetscRealPart(smallp));
	    if (PetscAbsScalar(dpstar) / PetscRealPart(*pstar) <= 1e-8) {
#if 0
	      break;
#endif
	    }
	}
    }

    *ustar = (zl * ustarr + zr * ustarl) / (zl + zr);
    if (PetscRealPart(*pstar) > PetscRealPart(*pl)) {
	pst = *pstar / *pl;
	*rstarl = ((*gaml + 1.) * pst + *gaml - 1.) / ((*gaml - 1.) * pst + *
		gaml + 1.) * *rl;
    }
    if (PetscRealPart(*pstar) > PetscRealPart(*pr)) {
	pst = *pstar / *pr;
	*rstarr = ((*gamr + 1.) * pst + *gamr - 1.) / ((*gamr - 1.) * pst + *
		gamr + 1.) * *rr;
    }
    return iwave;
}

PetscScalar sign(PetscScalar x)
{
    if(PetscRealPart(x) > 0) return 1.0;
    if(PetscRealPart(x) < 0) return -1.0;
    return 0.0;
}
/*        Riemann Solver */
/* -------------------------------------------------------------------- */
int riemannsolver(PetscScalar *xcen, PetscScalar *xp,
                   PetscScalar *dtt, PetscScalar *rl, PetscScalar *uxl, PetscScalar *pl,
                   PetscScalar *utl, PetscScalar *ubl, PetscScalar *gaml, PetscScalar *rho1l,
                   PetscScalar *rr, PetscScalar *uxr, PetscScalar *pr, PetscScalar *utr,
                   PetscScalar *ubr, PetscScalar *gamr, PetscScalar *rho1r, PetscScalar *rx,
                   PetscScalar *uxm, PetscScalar *px, PetscScalar *utx, PetscScalar *ubx,
                   PetscScalar *gam, PetscScalar *rho1)
{
    /* System generated locals */
    PetscScalar d__1, d__2;

    /* Local variables */
    static PetscScalar s, c0, p0, r0, u0, w0, x0, x2, ri, cx, sgn0, wsp0, gasc1, gasc2, gasc3, gasc4;
    static PetscScalar cstar, pstar, rstar, ustar, xstar, wspst, ushock, streng, rstarl, rstarr, rstars;
    int iwave;

    if (*rl == *rr && *pr == *pl && *uxl == *uxr && *gaml == *gamr) {
	*rx = *rl;
	*px = *pl;
	*uxm = *uxl;
	*gam = *gaml;
	x2 = *xcen + *uxm * *dtt;

	if (PetscRealPart(*xp) >= PetscRealPart(x2)) {
	    *utx = *utr;
	    *ubx = *ubr;
	    *rho1 = *rho1r;
	} else {
	    *utx = *utl;
	    *ubx = *ubl;
	    *rho1 = *rho1l;
	}
	return 0;
    }
    iwave = riem1mdt(gaml, gamr, rl, pl, uxl, rr, pr, uxr, &rstarl, &rstarr, &pstar, &ustar);

    x2 = *xcen + ustar * *dtt;
    d__1 = *xp - x2;
    sgn0 = sign(d__1);
/*            x is in 3-wave if sgn0 = 1 */
/*            x is in 1-wave if sgn0 = -1 */
    r0 = cvmgm_(rl, rr, &sgn0);
    p0 = cvmgm_(pl, pr, &sgn0);
    u0 = cvmgm_(uxl, uxr, &sgn0);
    *gam = cvmgm_(gaml, gamr, &sgn0);
    gasc1 = *gam - 1.;
    gasc2 = (*gam + 1.) * .5;
    gasc3 = gasc2 / *gam;
    gasc4 = 1. / (*gam - 1.);
    c0 = PetscSqrtScalar(*gam * p0 / r0);
    streng = pstar - p0;
    w0 = *gam * r0 * p0 * (gasc3 * streng / p0 + 1.);
    rstars = r0 / (1. - r0 * streng / w0);
    d__1 = p0 / pstar;
    d__2 = -1. / *gam;
    rstarr = r0 * PetscPowScalar(d__1, d__2);
    rstar = cvmgm_(&rstarr, &rstars, &streng);
    w0 = PetscSqrtScalar(w0);
    cstar = PetscSqrtScalar(*gam * pstar / rstar);
    wsp0 = u0 + sgn0 * c0;
    wspst = ustar + sgn0 * cstar;
    ushock = ustar + sgn0 * w0 / rstar;
    wspst = cvmgp_(&ushock, &wspst, &streng);
    wsp0 = cvmgp_(&ushock, &wsp0, &streng);
    x0 = *xcen + wsp0 * *dtt;
    xstar = *xcen + wspst * *dtt;
/*           using gas formula to evaluate rarefaction wave */
/*            ri : reiman invariant */
    ri = u0 - sgn0 * 2. * gasc4 * c0;
    cx = sgn0 * .5 * gasc1 / gasc2 * ((*xp - *xcen) / *dtt - ri);
    *uxm = ri + sgn0 * 2. * gasc4 * cx;
    s = p0 / PetscPowScalar(r0, *gam);
    d__1 = cx * cx / (*gam * s);
    *rx = PetscPowScalar(d__1, gasc4);
    *px = cx * cx * *rx / *gam;
    d__1 = sgn0 * (x0 - *xp);
    *rx = cvmgp_(rx, &r0, &d__1);
    d__1 = sgn0 * (x0 - *xp);
    *px = cvmgp_(px, &p0, &d__1);
    d__1 = sgn0 * (x0 - *xp);
    *uxm = cvmgp_(uxm, &u0, &d__1);
    d__1 = sgn0 * (xstar - *xp);
    *rx = cvmgm_(rx, &rstar, &d__1);
    d__1 = sgn0 * (xstar - *xp);
    *px = cvmgm_(px, &pstar, &d__1);
    d__1 = sgn0 * (xstar - *xp);
    *uxm = cvmgm_(uxm, &ustar, &d__1);
    if (PetscRealPart(*xp) >= PetscRealPart(x2)) {
	*utx = *utr;
	*ubx = *ubr;
	*rho1 = *rho1r;
    } else {
	*utx = *utl;
	*ubx = *ubl;
	*rho1 = *rho1l;
    }
    return iwave;
}
int godunovflux( const PetscScalar *ul, const PetscScalar *ur,
                 PetscScalar *flux, const PetscReal *nn, const int *ndim,
                 const PetscReal *gamma)
{
    /* System generated locals */
  int i__1,iwave;
    PetscScalar d__1, d__2, d__3;

    /* Local variables */
    static int k;
    static PetscScalar bn[3], fn, ft, tg[3], pl, rl, pm, pr, rr, xp, ubl, ubm,
	    ubr, dtt, unm, tmp, utl, utm, uxl, utr, uxr, gaml, gamm, gamr,
	    xcen, rhom, rho1l, rho1m, rho1r;
    /* Parameter adjustments */
    --nn;
    --flux;
    --ur;
    --ul;

    /* Function Body */
    xcen = 0.;
    xp = 0.;
    i__1 = *ndim;
    for (k = 1; k <= i__1; ++k) {
	tg[k - 1] = 0.;
	bn[k - 1] = 0.;
    }
    dtt = 1.;
    if (*ndim == 3) {
	if (nn[1] == 0. && nn[2] == 0.) {
	    tg[0] = 1.;
	} else {
	    tg[0] = -nn[2];
	    tg[1] = nn[1];
	}
/*           tmp=dsqrt(tg(1)**2+tg(2)**2) */
/*           tg=tg/tmp */
	bn[0] = -nn[3] * tg[1];
	bn[1] = nn[3] * tg[0];
	bn[2] = nn[1] * tg[1] - nn[2] * tg[0];
/* Computing 2nd power */
	d__1 = bn[0];
/* Computing 2nd power */
	d__2 = bn[1];
/* Computing 2nd power */
	d__3 = bn[2];
	tmp = PetscSqrtScalar(d__1 * d__1 + d__2 * d__2 + d__3 * d__3);
	i__1 = *ndim;
	for (k = 1; k <= i__1; ++k) {
	    bn[k - 1] /= tmp;
	}
    } else if (*ndim == 2) {
	tg[0] = -nn[2];
	tg[1] = nn[1];
/*           tmp=dsqrt(tg(1)**2+tg(2)**2) */
/*           tg=tg/tmp */
	bn[0] = 0.;
	bn[1] = 0.;
	bn[2] = 1.;
    }
    rl = ul[1];
    rr = ur[1];
    uxl = 0.;
    uxr = 0.;
    utl = 0.;
    utr = 0.;
    ubl = 0.;
    ubr = 0.;
    i__1 = *ndim;
    for (k = 1; k <= i__1; ++k) {
	uxl += ul[k + 1] * nn[k];
	uxr += ur[k + 1] * nn[k];
	utl += ul[k + 1] * tg[k - 1];
	utr += ur[k + 1] * tg[k - 1];
	ubl += ul[k + 1] * bn[k - 1];
	ubr += ur[k + 1] * bn[k - 1];
    }
    uxl /= rl;
    uxr /= rr;
    utl /= rl;
    utr /= rr;
    ubl /= rl;
    ubr /= rr;

    gaml = *gamma;
    gamr = *gamma;
/* Computing 2nd power */
    d__1 = uxl;
/* Computing 2nd power */
    d__2 = utl;
/* Computing 2nd power */
    d__3 = ubl;
    pl = (*gamma - 1.) * (ul[*ndim + 2] - rl * .5 * (d__1 * d__1 + d__2 * d__2 + d__3 * d__3));
/* Computing 2nd power */
    d__1 = uxr;
/* Computing 2nd power */
    d__2 = utr;
/* Computing 2nd power */
    d__3 = ubr;
    pr = (*gamma - 1.) * (ur[*ndim + 2] - rr * .5 * (d__1 * d__1 + d__2 * d__2 + d__3 * d__3));
    rho1l = rl;
    rho1r = rr;

    iwave = riemannsolver(&xcen, &xp, &dtt, &rl, &uxl, &pl, &utl, &ubl, &gaml, &
                          rho1l, &rr, &uxr, &pr, &utr, &ubr, &gamr, &rho1r, &rhom, &unm, &
                          pm, &utm, &ubm, &gamm, &rho1m);

    flux[1] = rhom * unm;
    fn = rhom * unm * unm + pm;
    ft = rhom * unm * utm;
/*           flux(2)=fn*nn(1)+ft*nn(2) */
/*           flux(3)=fn*tg(1)+ft*tg(2) */
    flux[2] = fn * nn[1] + ft * tg[0];
    flux[3] = fn * nn[2] + ft * tg[1];
/*           flux(2)=rhom*unm*(unm)+pm */
/*           flux(3)=rhom*(unm)*utm */
    if (*ndim == 3) {
	flux[4] = rhom * unm * ubm;
    }
    flux[*ndim + 2] = (rhom * .5 * (unm * unm + utm * utm + ubm * ubm) + gamm / (gamm - 1.) * pm) * unm;
    return iwave;
} /* godunovflux_ */

/* Subroutine to set up the initial conditions for the */
/* Shock Interface interaction or linear wave (Ravi Samtaney,Mark Adams). */
/* ----------------------------------------------------------------------- */
int projecteqstate(PetscReal wc[], const PetscReal ueq[], PetscReal lv[][3])
{
  int j,k;
/*      Wc=matmul(lv,Ueq) 3 vars */
  for (k = 0; k < 3; ++k) {
    wc[k] = 0.;
    for (j = 0; j < 3; ++j) {
      wc[k] += lv[k][j]*ueq[j];
    }
  }
  return 0;
}
/* ----------------------------------------------------------------------- */
int projecttoprim(PetscReal v[], const PetscReal wc[], PetscReal rv[][3])
{
  int k,j;
  /*      V=matmul(rv,WC) 3 vars */
  for (k = 0; k < 3; ++k) {
    v[k] = 0.;
    for (j = 0; j < 3; ++j) {
      v[k] += rv[k][j]*wc[j];
    }
  }
  return 0;
}
/* ---------------------------------------------------------------------- */
int eigenvectors(PetscReal rv[][3], PetscReal lv[][3], const PetscReal ueq[], PetscReal gamma)
{
  int j,k;
  PetscReal rho,csnd,p0;
  /* PetscScalar u; */

  for (k = 0; k < 3; ++k) for (j = 0; j < 3; ++j) { lv[k][j] = 0.; rv[k][j] = 0.; }
  rho = ueq[0];
  /* u = ueq[1]; */
  p0 = ueq[2];
  csnd = PetscSqrtReal(gamma * p0 / rho);
  lv[0][1] = rho * .5;
  lv[0][2] = -.5 / csnd;
  lv[1][0] = csnd;
  lv[1][2] = -1. / csnd;
  lv[2][1] = rho * .5;
  lv[2][2] = .5 / csnd;
  rv[0][0] = -1. / csnd;
  rv[1][0] = 1. / rho;
  rv[2][0] = -csnd;
  rv[0][1] = 1. / csnd;
  rv[0][2] = 1. / csnd;
  rv[1][2] = 1. / rho;
  rv[2][2] = csnd;
  return 0;
}

int initLinearWave(EulerNode *ux, const PetscReal gamma, const PetscReal coord[], const PetscReal Lx)
{
  PetscReal p0,u0,wcp[3],wc[3];
  PetscReal lv[3][3];
  PetscReal vp[3];
  PetscReal rv[3][3];
  PetscReal eps, ueq[3], rho0, twopi;

  /* Function Body */
  twopi = 2.*PETSC_PI;
  eps = 1e-4; /* perturbation */
  rho0 = 1e3;   /* density of water */
  p0 = 101325.; /* init pressure of 1 atm (?) */
  u0 = 0.;
  ueq[0] = rho0;
  ueq[1] = u0;
  ueq[2] = p0;
  /* Project initial state to characteristic variables */
  eigenvectors(rv, lv, ueq, gamma);
  projecteqstate(wc, ueq, lv);
  wcp[0] = wc[0];
  wcp[1] = wc[1];
  wcp[2] = wc[2] + eps * PetscCosReal(coord[0] * 2. * twopi / Lx);
  projecttoprim(vp, wcp, rv);
  ux->r = vp[0]; /* density */
  ux->ru[0] = vp[0] * vp[1]; /* x momentum */
  ux->ru[1] = 0.;
#if defined DIM > 2
  if (dim>2) ux->ru[2] = 0.;
#endif
  /* E = rho * e + rho * v^2/2 = p/(gam-1) + rho*v^2/2 */
  ux->E = vp[2]/(gamma - 1.) + 0.5*vp[0]*vp[1]*vp[1];
  return 0;
}

/*TEST

  # 2D Advection 0-10
  test:
    suffix: 0
    requires: exodusii
    args: -ufv_vtk_interval 0 -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside.exo

  test:
    suffix: 1
    requires: exodusii
    args: -ufv_vtk_interval 0 -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad-15.exo

  test:
    suffix: 2
    requires: exodusii
    nsize: 2
    args: -ufv_vtk_interval 0 -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside.exo

  test:
    suffix: 3
    requires: exodusii
    nsize: 2
    args: -ufv_vtk_interval 0 -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad-15.exo

  test:
    suffix: 4
    requires: exodusii
    nsize: 8
    args: -ufv_vtk_interval 0 -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad.exo

  test:
    suffix: 5
    requires: exodusii
    args: -ufv_vtk_interval 0 -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside.exo -ts_type rosw -ts_adapt_reject_safety 1

  test:
    suffix: 6
    requires: exodusii
    args: -ufv_vtk_interval 0 -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/squaremotor-30.exo -ufv_split_faces

  test:
    suffix: 7
    requires: exodusii
    args: -ufv_vtk_interval 0 -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad-15.exo -dm_refine 1

  test:
    suffix: 8
    requires: exodusii
    nsize: 2
    args: -ufv_vtk_interval 0 -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad-15.exo -dm_refine 1

  test:
    suffix: 9
    requires: exodusii
    nsize: 8
    args: -ufv_vtk_interval 0 -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad-15.exo -dm_refine 1

  test:
    suffix: 10
    requires: exodusii
    args: -ufv_vtk_interval 0 -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad.exo

  # 2D Shallow water
  test:
    suffix: sw_0
    requires: exodusii
    args: -ufv_vtk_interval 0 -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/annulus-20.exo -bc_wall 100,101 -physics sw -ufv_cfl 5 -petscfv_type leastsquares -petsclimiter_type sin -ts_max_time 1 -ts_ssp_type rks2 -ts_ssp_nstages 10 -monitor height,energy

  # 2D Advection: p4est
  test:
    suffix: p4est_advec_2d
    requires: p4est
    args: -ufv_vtk_interval 0 -f -dm_type p4est -dm_forest_minimum_refinement 1 -dm_forest_initial_refinement 2 -dm_p4est_refine_pattern hash -dm_forest_maximum_refinement 5

  # Advection in a box
  test:
    suffix: adv_2d_quad_0
    args: -ufv_vtk_interval 0 -dm_refine 3 -dm_plex_separate_marker -bc_inflow 1,2,4 -bc_outflow 3

  test:
    suffix: adv_2d_quad_1
    args: -ufv_vtk_interval 0 -dm_refine 3 -dm_plex_separate_marker -grid_bounds -0.5,0.5,-0.5,0.5 -bc_inflow 1,2,4 -bc_outflow 3 -advect_sol_type bump -advect_bump_center 0.25,0 -advect_bump_radius 0.1
    timeoutfactor: 3

  test:
    suffix: adv_2d_quad_p4est_0
    requires: p4est
    args: -ufv_vtk_interval 0 -dm_refine 5 -dm_type p4est -dm_plex_separate_marker -bc_inflow 1,2,4 -bc_outflow 3

  test:
    suffix: adv_2d_quad_p4est_1
    requires: p4est
    args: -ufv_vtk_interval 0 -dm_refine 5 -dm_type p4est -dm_plex_separate_marker -grid_bounds -0.5,0.5,-0.5,0.5 -bc_inflow 1,2,4 -bc_outflow 3 -advect_sol_type bump -advect_bump_center 0.25,0 -advect_bump_radius 0.1
    timeoutfactor: 3

  test:
    suffix: adv_2d_quad_p4est_adapt_0
    requires: p4est !__float128 #broken for quad precision
    args: -ufv_vtk_interval 0 -dm_refine 3 -dm_type p4est -dm_plex_separate_marker -grid_bounds -0.5,0.5,-0.5,0.5 -bc_inflow 1,2,4 -bc_outflow 3 -advect_sol_type bump -advect_bump_center 0.25,0 -advect_bump_radius 0.1 -ufv_use_amr -refine_vec_tagger_box 0.005,inf -coarsen_vec_tagger_box 0,1.e-5 -petscfv_type leastsquares -ts_max_time 0.01
    timeoutfactor: 3

  test:
    suffix: adv_2d_tri_0
    requires: triangle
    TODO: how did this ever get in master when there is no support for this
    args: -ufv_vtk_interval 0 -simplex -dm_refine 3 -dm_plex_separate_marker -bc_inflow 1,2,4 -bc_outflow 3

  test:
    suffix: adv_2d_tri_1
    requires: triangle
    TODO: how did this ever get in master when there is no support for this
    args: -ufv_vtk_interval 0 -simplex -dm_refine 5 -dm_plex_separate_marker -grid_bounds -0.5,0.5,-0.5,0.5 -bc_inflow 1,2,4 -bc_outflow 3 -advect_sol_type bump -advect_bump_center 0.25,0 -advect_bump_radius 0.1

  test:
    suffix: adv_0
    requires: exodusii
    args: -ufv_vtk_interval 0 -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo -bc_inflow 100,101,200 -bc_outflow 201

  test:
    suffix: shock_0
    requires: p4est !single !complex
    args: -ufv_vtk_interval 0 -monitor density,energy -f -grid_size 2,1 -grid_bounds -1,1.,0.,1 -bc_wall 1,2,3,4 -dm_type p4est -dm_forest_partition_overlap 1 -dm_forest_maximum_refinement 6 -dm_forest_minimum_refinement 2 -dm_forest_initial_refinement 2 -ufv_use_amr -refine_vec_tagger_box 0.5,inf -coarsen_vec_tagger_box 0,1.e-2 -refine_tag_view -coarsen_tag_view -physics euler -eu_type iv_shock -ufv_cfl 10 -eu_alpha 60. -grid_skew_60 -eu_gamma 1.4 -eu_amach 2.02 -eu_rho2 3. -petscfv_type leastsquares -petsclimiter_type minmod -petscfv_compute_gradients 0 -ts_max_time 0.5 -ts_ssp_type rks2 -ts_ssp_nstages 10 -ufv_vtk_basename ${wPETSC_DIR}/ex11
    timeoutfactor: 3

  # Test GLVis visualization of PetscFV fields
  test:
    suffix: glvis_adv_2d_tet
    args: -ufv_vtk_interval 0 -ts_monitor_solution glvis: -ts_max_steps 0 -ufv_vtk_monitor 0 -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/square_periodic.msh -dm_plex_gmsh_periodic 0

  test:
    suffix: glvis_adv_2d_quad
    args: -ufv_vtk_interval 0 -ts_monitor_solution glvis: -ts_max_steps 0 -ufv_vtk_monitor 0 -dm_refine 5 -dm_plex_separate_marker -bc_inflow 1,2,4 -bc_outflow 3

  test:
    suffix: tut_1
    requires: exodusii
    nsize: 1
    args: -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside.exo

  test:
    suffix: tut_2
    requires: exodusii
    nsize: 1
    args: -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside.exo -ts_type rosw

  test:
    suffix: tut_3
    requires: exodusii
    nsize: 4
    args: -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/annulus-20.exo -monitor Error -advect_sol_type bump -petscfv_type leastsquares -petsclimiter_type sin

  test:
    suffix: tut_4
    requires: exodusii
    nsize: 4
    args: -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/annulus-20.exo -physics sw -monitor Height,Energy -petscfv_type leastsquares -petsclimiter_type minmod

TEST*/
