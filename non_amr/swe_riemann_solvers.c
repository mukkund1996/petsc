static char help[] = "Nonlinear, time-dependent PDE in 2d.\n";

#include <petscdmplex.h>
#include <petscds.h>
#include <petscts.h>
#include <petscsf.h> /* For SplitFaces() */
#include <petscviewerhdf5.h>

#define DIM 2                   /* Geometric dimension */
#define ALEN(a) (sizeof(a)/sizeof((a)[0])) /* Just returns the size of the array */

typedef struct _n_FunctionalLink *FunctionalLink;
/* 'User' implements a discretization of a continuous model. */
typedef struct _n_User *User;
/* Represents continuum physical equations. */
typedef struct _n_Physics *Physics;
/* Physical model includes boundary conditions, initial conditions, and functionals of interest. It is
 * discretization-independent, but its members depend on the scenario being solved. */
typedef struct _n_Model *Model;
/* 'User' implements a discretization of a continuous model. */
typedef struct _n_User *User;

/* Declaration of all functions */
extern PetscErrorCode MyTSMonitor(TS, PetscInt, PetscReal, Vec, void *);

typedef PetscErrorCode (*SolutionFunction)(Model, PetscReal, const PetscReal *, PetscScalar *, void *);

typedef PetscErrorCode (*SetUpBCFunction)(PetscDS, Physics);

PETSC_STATIC_INLINE PetscReal Dot2Real(const PetscReal *x, const PetscReal *y) { return x[0] * y[0] + x[1] * y[1]; }

PETSC_STATIC_INLINE PetscReal Norm2Real(const PetscReal *x) { return PetscSqrtReal(PetscAbsReal(Dot2Real(x, x))); }

PETSC_STATIC_INLINE void Normalize2Real(PetscReal *x) {
    PetscReal a = 1. / Norm2Real(x);
    x[0] *= a;
    x[1] *= a;
}

PETSC_STATIC_INLINE void Scale2Real(PetscReal a, const PetscReal *x, PetscReal *y) {
    y[0] = a * x[0];
    y[1] = a * x[1];
}

/* Defining super and sub classes of user context structs */
struct _n_FunctionalLink {
    char *name;
    void *ctx;
    PetscInt offset;
    FunctionalLink next;
};

struct _n_User {
    Model model;
};

struct _n_Physics {
    PetscRiemannFunc riemann;
    PetscInt dof;          /* number of degrees of freedom per cell */
    PetscReal maxspeed;     /* kludge to pick initial time step, need to add monitoring and step control */
    void *data;
    PetscInt nfields;
    PetscFV fvm;
    const struct FieldDescription *field_desc;
};

struct _n_Model {
    MPI_Comm comm;        /* Does not do collective communicaton, but some error conditions can be collective */
    Physics physics;
    PetscInt maxComputed;
    PetscInt numMonitored;
    FunctionalLink *functionalMonitored;
    FunctionalLink *functionalCall;
    SolutionFunction solution;
    SetUpBCFunction setupbc;
    void *solutionctx;
    PetscReal maxspeed;    /* estimate of global maximum speed (for CFL calculation) */
    PetscReal bounds[2 * DIM];
    DMBoundaryType bcs[3];
    PetscReal cfl;
};

/* Defining the context specific for the shallow water model */
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
    PetscReal uh[2];
} SWNode;
typedef union {
    SWNode swnode;
    PetscReal vals[3];
} SWNodeUnion;

/* Function returning flux given the context and the conserved variables */
static PetscErrorCode SWFlux(Physics phys, const PetscReal *n, const SWNode *x, SWNode *f) {
    Physics_SW *sw = (Physics_SW *) phys->data;
    PetscReal uhn, u[DIM];
    PetscInt i;

    PetscFunctionBeginUser;
    Scale2Real(1. / x->h, x->uh, u);
    uhn = x->uh[0] * n[0] + x->uh[1] * n[1];
    f->h = uhn;
    for (i = 0; i < DIM; i++) f->uh[i] = u[i] * uhn + 0.5 * sw->gravity * PetscSqr(x->h) * n[i];
    PetscFunctionReturn(0);
}

/* Functions responsible for defining the boundary cells (elements) */
static PetscErrorCode
PhysicsBoundary_SW_Wall(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *xI, PetscScalar *xG,
                        void *ctx) {
    PetscFunctionBeginUser;
    xG[0] = xI[0];
    xG[1] = -xI[1];
    xG[2] = -xI[2];
    PetscFunctionReturn(0);
}

/* Defining the function responsible for the setting up the BCs */
static PetscErrorCode SetUpBC_SW(PetscDS prob, Physics phys) {
    PetscErrorCode ierr;
    const PetscInt wallids[] = {1, 2, 3, 4};
    PetscFunctionBeginUser;
    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "wall", "Face Sets", 0, 0, NULL,
                              (void (*)(void)) PhysicsBoundary_SW_Wall, ALEN(wallids), wallids, phys);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/* Specifying the Riemann function for the shallow water model model */
static void PhysicsRiemann_SW(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n, const PetscScalar *xL,
                              const PetscScalar *xR, PetscInt numConstants, const PetscScalar constants[],
                              PetscScalar *flux, Physics phys) {
    Physics_SW *sw = (Physics_SW *) phys->data;
    PetscReal cL, cR, speed;
    PetscReal nn[DIM];
    char riemann[64] = "rusanov";
    PetscOptionsGetString(NULL, NULL, "-riemann", riemann, sizeof(riemann), NULL);

    const SWNode *uL = (const SWNode *) xL, *uR = (const SWNode *) xR;
    SWNodeUnion fL, fR;
    PetscInt i;
    PetscReal zero = 0.;

    if (uL->h < 0 || uR->h < 0) {
        for (i = 0; i < 1 + dim; i++) flux[i] = zero / zero;
        return;
    } /* SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Reconstructed thickness is negative"); */

    nn[0] = n[0];
    nn[1] = n[1];
    Normalize2Real(nn);
    SWFlux(phys, nn, uL, &(fL.swnode));
    SWFlux(phys, nn, uR, &(fR.swnode));

    PetscReal U_L[2], U_R[2];
    cR = PetscSqrtReal(sw->gravity * uR->h);
    cL = PetscSqrtReal(sw->gravity * uL->h);

    PetscBool flg = PETSC_FALSE, flg1 = PETSC_FALSE;
    PetscStrcmp(riemann, "roe", &flg);
    PetscStrcmp(riemann, "hllc", &flg1);

    /* APPROXIMATE RIEMANN SOLVER : ROE */
    if (flg) {
        PetscInt j;
        PetscReal duml, dumr, hl = uL->h, hr = uR->h, ul, ur, vl, vr;
        ul = PetscAbsReal((uL->uh[0]) / uL->h);
        vl = PetscAbsReal((uL->uh[1]) / uL->h);
        ur = PetscAbsReal((uR->uh[0]) / uR->h);
        vr = PetscAbsReal((uR->uh[1]) / uR->h);
        duml = PetscSqrtReal(hl);
        dumr = PetscSqrtReal(hr);

        PetscReal hhat, uhat, vhat, chat, uperp, dh, du, dv, cn = n[0], sn = n[1];
        hhat = duml * dumr;
        uhat = (duml * ul + dumr * ur) / (duml + dumr);
        vhat = (duml * vl + dumr * vr) / (duml + dumr);
        chat = PetscSqrtReal(0.5 * sw->gravity * (hl + hr));
        uperp = uhat * cn + vhat * sn;
        dh = hr - hl;
        du = ur - ul;
        dv = vr - vl;

        PetscReal dupar, duperp, dW[3];
        dupar = -du * sn + dv * cn;
        duperp = du * cn + dv * sn;
        dW[0] = 0.5 * (dh - hhat * duperp / chat);
        dW[1] = hhat * dupar;
        dW[2] = 0.5 * (dh + hhat * duperp / chat);

        PetscReal uperpl, uperpr, al1, al3, ar1, ar3;
        uperpl = ul * cn + vl * sn;
        uperpr = ur * cn + vr * sn;
        al1 = uperpl - cL;
        al3 = uperpl + cL;
        ar1 = uperpr - cR;
        ar3 = uperpr + cR;

        /* Definition of R */
        PetscReal R[3][3];
        R[0][0] = 1;
        R[0][1] = 0;
        R[0][2] = 1;
        R[1][0] = uhat - chat * cn;
        R[1][1] = -sn;
        R[1][2] = uhat + chat * cn;
        R[2][0] = vhat - chat * sn;
        R[2][1] = cn;
        R[2][2] = vhat + chat * sn;

        /* Finding the characteristic speeds and applying the fixes */
        PetscReal da1, da3, a[3];
        da1 = PetscMax(0, (2 * (ar1 - al1)));
        da3 = PetscMax(0, (2 * (ar3 - al3)));
        a[0] = PetscAbsReal(uperp - chat);
        a[1] = PetscAbsReal(uperp);
        a[2] = PetscAbsReal(uperp + chat);
        if (a[0] < da1)
            a[0] = 0.5 * (a[0] * a[0] / da1 + da1);
        if (a[2] < da3)
            a[2] = 0.5 * (a[2] * a[2] / da3 + da3);

        PetscReal A[3][3];
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                if (i == j)
                    A[i][j] = a[i];
                else
                    A[i][j] = 0;
            }
        }
        /* Finding the products  */
        PetscInt k;
        PetscReal C[3][3], speed[3], sum;
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                sum = 0.0;
                for (k = 0; k < 3; k++) {
                    sum += R[i][k] * A[k][j];
                }
                C[i][j] = sum;
            }
        }
        for (i = 0; i < 3; i++) {
            sum = 0;
            for (j = 0; j < 3; j++) {
                sum += C[i][j] * dW[j];
            }
            speed[i] = sum;
        }
        /* Computing the fluxes */
        for (i = 0; i < 3; i++)
            flux[i] = 0.5 * (fL.vals[i] + fR.vals[i] - speed[i]);
    }

    /* APPROXIMATE RIEMANN SOLVER : HLLC */
    else if (flg1) {
        PetscReal hL, hR, u_L, u_R, hStar, uStar;
        hL = uL->h;
        hR = uR->h;
        u_L = PetscAbsReal(Dot2Real(uL->uh, nn) / uL->h);
        u_R = PetscAbsReal(Dot2Real(uR->uh, nn) / uR->h);
        for (i = 0; i < 2; i++) {
            U_L[i] = PetscAbsReal((uL->uh[i]) / uL->h);
            U_R[i] = PetscAbsReal((uR->uh[i]) / uR->h);
        }
        /* Computing uStar */
        uStar = 0.5 * (u_L + u_R) + cL - cR;
        /* Computing hStar */
        hStar = PetscSqr(0.5 * (cL + cR) + 0.25 * (u_L - u_R)) / sw->gravity;
        /* Computing sL */
        PetscReal sL, sR;
        if (hL > 0)
            sL = PetscMin((u_L - cL), uStar - PetscSqrtReal((sw->gravity * hStar)));
        else
            sL = u_R - 2 * cR;
        /* Computing sR */
        if (hR > 0)
            sR = PetscMax((u_R + cR), uStar + PetscSqrtReal((sw->gravity * hStar)));
        else
            sR = u_L + 2 * cL;
        /* Computing sStar */
        PetscReal sStar;
        sStar = (sL * hR * (u_R - sR) - sR * hL * (u_L - sL)) / (hR * (u_R - sR) - hL * (u_L - sL));

        PetscReal uLbar, uRbar;
        uLbar = U_L[1] * nn[0] - U_L[0] * nn[1];
        uRbar = U_R[1] * nn[0] - U_R[0] * nn[1];

        PetscReal xbarL[3], xbarR[3];
        xbarL[0] = hL;
        xbarL[1] = hL * u_L;
        xbarL[2] = hL * uLbar;

        xbarR[0] = hR;
        xbarR[1] = hR * u_R;
        xbarR[2] = hR * uRbar;

        PetscReal fStar[3], fStar_L[3], fStar_R[3];
        for (i = 0; i < 1 + dim; i++)
            fStar[i] = (sR * fL.vals[i] - sL * fR.vals[i] + sL * sR * (xbarL[i] - xbarR[i])) / (sR - sL);
        /* Defining the flux by the intermediate wave */
        for (i = 0; i < 2; i++) {
            fStar_L[i] = fStar[i];
            fStar_R[i] = fStar[i];
        }
        fStar_L[2] = fStar[0] * uLbar;
        fStar_R[2] = fStar[0] * uRbar;

        /* Picking the final flux */
        if (sL >= 0) {
            for (i = 0; i < 3; i++)
                flux[i] = fL.vals[i];
        } else if (sStar >= 0 && sL < 0) {
            for (i = 0; i < 3; i++)
                flux[i] = fStar_L[i];
        } else if (sStar < 0 && sR >= 0) {
            for (i = 0; i < 3; i++)
                flux[i] = fStar_R[i];
        } else {
            for (i = 0; i < 3; i++)
                flux[i] = fR.vals[i];
        }
    }
    /* APPROXIMATE RIEMANN SOLVER : RUSANOV */
    else {
        /* gravity wave speed */
        speed = PetscMax(PetscAbsReal(Dot2Real(uL->uh, nn) / uL->h) + cL,
                         PetscAbsReal(Dot2Real(uR->uh, nn) / uR->h) + cR);
        for (i = 0; i < 1 + dim; i++)
            flux[i] = (0.5 * (fL.vals[i] + fR.vals[i]) + 0.5 * speed * (xL[i] - xR[i])) * Norm2Real(n);
    }
}

/* Routine for defining the initial solution
 * Calling sequence of the function :
 * func(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx);
 * dim	- The spatial dimension
 * x	- The coordinates
 * Nf	- The number of fields
 * u	- The output field values
 * ctx	- optional user-defined function context
*/
static PetscErrorCode PhysicsSolution_SW(Model mod, PetscReal time, const PetscReal *x, PetscScalar *u, void *ctx) {
    PetscErrorCode ierr;
    PetscReal dx[2], r, sigma;
    PetscBool bc1 = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL, NULL, "-bc", &bc1, NULL);
    CHKERRQ(ierr);

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
            u[0] = 1.1;
        else
            u[0] = 1.0;
        u[1] = 0.0;
        u[2] = 0.0;
    }

    PetscFunctionReturn(0);
}

/* Defining the functional parameters defined with the SW model context */
static PetscErrorCode
PhysicsFunctional_SW(Model mod, PetscReal time, const PetscReal *coord, const PetscScalar *xx, PetscReal *f,
                     void *ctx) {
    Physics phys = (Physics) ctx;
    Physics_SW *sw = (Physics_SW *) phys->data;
    const SWNode *x = (const SWNode *) xx;
    PetscReal u[2];
    PetscReal h;

    PetscFunctionBeginUser;
    h = x->h;
    Scale2Real(1. / x->h, x->uh, u);
    f[sw->functional.Height] = h;
    f[sw->functional.Speed] = Norm2Real(u) + PetscSqrtReal(sw->gravity * h);
    f[sw->functional.Energy] = 0.5 * (Dot2Real(x->uh, u) + sw->gravity * PetscSqr(h));
    PetscFunctionReturn(0);
}

/* Initializing all of the structs related to the shallow water model */
static PetscErrorCode PhysicsCreate_SW(Model mod, Physics phys) {
    Physics_SW *sw;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    phys->riemann = (PetscRiemannFunc) PhysicsRiemann_SW;
    ierr = PetscNew(&sw);
    CHKERRQ(ierr);
    phys->data = sw;
    mod->setupbc = SetUpBC_SW;
    sw->gravity = 9.81;
    /* Mach 1 for depth of 2 */
    phys->maxspeed = PetscSqrtReal(2.0 * sw->gravity);
    /* Initial/transient solution with default boundary conditions */
    mod->solution = PhysicsSolution_SW;
    mod->solutionctx = phys;
    mod->bcs[0] = mod->bcs[1] = mod->bcs[2] = DM_BOUNDARY_GHOSTED;
    PetscFunctionReturn(0);

}
/* End of model specific structure definition */

/* Defining the routine to set the initial value of the solution */
static PetscErrorCode
SolutionFunctional(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *modctx) {
    Model mod;
    PetscErrorCode ierr;
    PetscFunctionBegin;
    mod = (Model) modctx;
    ierr = (*mod->solution)(mod, time, x, u, mod->solutionctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode Compute_exact_solution(DM dm, Vec X, User user, PetscReal time) {
    PetscErrorCode(*func[1])(PetscInt
    dim, PetscReal
    time,
    const PetscReal x[], PetscInt
    Nf, PetscScalar * u,
    void *ctx);
    void *ctx[1];
    Model mod = user->model;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    func[0] = SolutionFunctional;
    ctx[0] = (void *) mod;
    ierr = DMProjectFunction(dm, time, func, ctx, INSERT_ALL_VALUES, X);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/* Initializing the TS object and defining the parameters */
static PetscErrorCode initializeTS(DM dm, User user, TS *ts) {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = TSCreate(PetscObjectComm((PetscObject) dm), ts);
    CHKERRQ(ierr);
    ierr = TSSetType(*ts, TSSSP);
    CHKERRQ(ierr);
    ierr = TSSetDM(*ts, dm);
    CHKERRQ(ierr);
    ierr = TSMonitorSet(*ts, MyTSMonitor, PETSC_VIEWER_STDOUT_WORLD, NULL);
    CHKERRQ(ierr);
    ierr = DMTSSetRHSFunctionLocal(dm, DMPlexTSComputeRHSFunctionFVM, user);
    CHKERRQ(ierr);
    ierr = TSSetMaxTime(*ts, 2.0);
    CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(*ts, TS_EXACTFINALTIME_STEPOVER);
    CHKERRQ(ierr);
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

PetscErrorCode MyTSMonitor(TS ts, PetscInt step, PetscReal ptime, Vec v, void *ctx) {
    PetscErrorCode ierr;
    PetscReal norm;
    MPI_Comm comm;

    PetscFunctionBeginUser;
    if (step < 0) PetscFunctionReturn(0); /* step of -1 indicates an interpolated solution */
    ierr = VecNorm(v, NORM_2, &norm);
    CHKERRQ(ierr);
    ierr = PetscObjectGetComm((PetscObject) ts, &comm);
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "timestep %D time %g norm %g\n", step, (double) ptime, (double) norm);
    CHKERRQ(ierr);
    PetscInt vtkInterval = 5;
    char outputBasename[256] = "swe_fixed", filename[PETSC_MAX_PATH_LEN];
    DM dm;
    PetscViewer viewer;
    ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    if ((step == -1) ^ (step % vtkInterval == 0)) {
        if (step == -1) {        /* Final time is not multiple of normal time interval, write it anyway */
            ierr = TSGetStepNumber(ts, &step);
            CHKERRQ(ierr);
        }
        ierr = PetscSNPrintf(filename, sizeof filename, "./vtk_output/%s-%03D.vtu", outputBasename, step);
        CHKERRQ(ierr);
        ierr = OutputVTK(dm, filename, &viewer);
        CHKERRQ(ierr);
        ierr = VecView(v, viewer);
        CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer);
        CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

/*static PetscErrorCode CreateMesh(MPI_Comm comm, DM dm, Model mod, PetscInt dim) {

    PetscErrorCode ierr;
    *//* Defining the dimension of the domain *//*
    PetscBool simplex = PETSC_FALSE;
    PetscInt n = 10, overlap = 1;
    ierr = PetscOptionsGetInt(NULL, NULL, "-mesh", &n, NULL);
    CHKERRQ(ierr);
    *//* Setting a default cfl value *//*
    mod->cfl = 1.0;
    ierr = PetscOptionsGetReal(NULL, NULL, "-cfl", &mod->cfl, NULL);
    CHKERRQ(ierr);

    PetscFunctionBeginUser;
    *//* Defining the number of faces in each dimension *//*
    PetscInt cells[3] = {n, n, 1};
    ierr = DMPlexCreateBoxMesh(comm, dim, simplex, cells, NULL, NULL, mod->bcs, PETSC_TRUE, &dm);
    CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);
    CHKERRQ(ierr);

    *//* set up BCs, functions, tags *//*
    ierr = DMCreateLabel(dm, "Face Sets");
    CHKERRQ(ierr);

    *//* Configuring the DMPLEX object for FVM and distributing over all procs *//*
    DM dmDist;

    ierr = DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE);
    CHKERRQ(ierr);
    ierr = DMPlexDistribute(dm, overlap, NULL, &dmDist);
    CHKERRQ(ierr);
    if (dmDist) {
        ierr = DMDestroy(&dm);
        CHKERRQ(ierr);
        dm = dmDist;
    }
    PetscFunctionReturn(0);

    *//* Constructing the ghost cells for DMPLEX object *//*
    DM gdm;
    ierr = DMPlexConstructGhostCells(dm, NULL, NULL, &gdm);
    CHKERRQ(ierr);
    ierr = DMDestroy(&dm);
    CHKERRQ(ierr);
    dm = gdm;
    ierr = DMViewFromOptions(dm, NULL, "-dm_view");
    CHKERRQ(ierr);

}

static PetscErrorCode CreateFV(MPI_Comm comm, PetscInt dim, PetscFV fvm, Physics phys) {

    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    *//* Creating and configuring the PetscFV object *//*
    ierr = PetscFVCreate(comm, &fvm);
    CHKERRQ(ierr);
    ierr = PetscFVSetNumComponents(fvm, phys->dof);
    CHKERRQ(ierr);
    ierr = PetscFVSetSpatialDimension(fvm, dim);
    CHKERRQ(ierr);

    *//*....Setting the FV limiter....*//*
    PetscLimiter limiter = NULL, LimiterType = NULL;
    ierr = PetscFVGetLimiter(fvm, &limiter);
    CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject) limiter);
    CHKERRQ(ierr);
    ierr = PetscLimiterCreate(PetscObjectComm((PetscObject) fvm), &LimiterType);
    CHKERRQ(ierr);
    ierr = PetscLimiterSetType(LimiterType, PETSCLIMITERSUPERBEE);
    CHKERRQ(ierr);
    ierr = PetscFVSetLimiter(fvm, LimiterType);
    CHKERRQ(ierr);

    ierr = PetscObjectSetName((PetscObject) fvm, "");
    CHKERRQ(ierr);
    *//* Defining the component name for the PetscFV object *//*
    ierr = PetscFVSetComponentName(fvm, 0, "h");
    CHKERRQ(ierr);
    ierr = PetscFVSetComponentName(fvm, 1, "uh");
    CHKERRQ(ierr);
    ierr = PetscLimiterDestroy(&limiter);
    CHKERRQ(ierr);
    ierr = PetscLimiterDestroy(&LimiterType);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);

}

static PetscErrorCode CreateDS(DM dm, PetscFV fvm, User user, PetscDS prob) {

    PetscErrorCode ierr;
    Model mod;
    Physics phys;
    PetscFunctionBeginUser;
    mod = user->model;
    phys = mod->physics;

    *//* Adding the field and specifying the dof (no. of components) *//*
    ierr = DMAddField(dm, NULL, (PetscObject) fvm);
    CHKERRQ(ierr);
    ierr = DMCreateDS(dm);
    CHKERRQ(ierr);
    ierr = DMGetDS(dm, &prob);
    CHKERRQ(ierr);
    ierr = PetscDSSetRiemannSolver(prob, 0, user->model->physics->riemann);
    CHKERRQ(ierr);
    ierr = PetscDSSetContext(prob, 0, user->model->physics);
    CHKERRQ(ierr);
    ierr = (*mod->setupbc)(prob, phys);
    CHKERRQ(ierr);
    ierr = PetscDSSetFromOptions(prob);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}*/

int main(int argc, char **argv) {
    MPI_Comm comm;
    PetscDS prob = NULL;
    User user;
    Model mod;
    Physics phys = NULL;
//    PetscFV fvm;
    DM dm = NULL;
    PetscReal ftime, dt, minRadius;
    PetscInt dim = 2, nsteps;
    TS ts;
    Vec X;
    PetscErrorCode ierr;
    PetscMPIInt rank;

    ierr = PetscInitialize(&argc, &argv, (char *) 0, help);
    if (ierr) return ierr;
    comm = PETSC_COMM_WORLD;
    ierr = MPI_Comm_rank(comm, &rank);
    CHKERRQ(ierr);


    ierr = PetscNew(&user);
    CHKERRQ(ierr);
    ierr = PetscNew(&user->model);
    CHKERRQ(ierr);
    ierr = PetscNew(&user->model->physics);
    CHKERRQ(ierr);
    mod = user->model;
    phys = mod->physics;
    mod->comm = comm;

    /* Initializing the structures for the shallow water model */
    ierr = PhysicsCreate_SW(mod, phys);
    CHKERRQ(ierr);

    /* Setting the number of fields and dof for the model
     * No. of fields and DOF (No. of components)
     * 0. h     - DOF = 1
     * 1. uh/vh - DOF = 2
     *
     * Total no. of fields = 2
     * Total no. of DOF    = 3
     * */
    phys->nfields = 2;
    phys->dof = 3;

//    /* Mesh creation routines using DMPLEXCreateBoxMesh */
//    ierr = CreateMesh(comm, dm, mod, dim);
//    CHKERRQ(ierr);
//
//    /* Creation and configuration of the PetscFV object */
//    ierr = CreateFV(comm, dim, fvm, phys);
//    CHKERRQ(ierr);
//
//    ierr = CreateDS(dm, fvm, user, prob);
//    CHKERRQ(ierr);

    PetscBool simplex = PETSC_FALSE;
    PetscInt n = 10, overlap = 1;
    ierr = PetscOptionsGetInt(NULL, NULL, "-mesh", &n, NULL);
    CHKERRQ(ierr);
    /* Setting a default cfl value */
    mod->cfl = 1.0;
    ierr = PetscOptionsGetReal(NULL, NULL, "-cfl", &mod->cfl, NULL);
    CHKERRQ(ierr);

    /* Defining the number of faces in each dimension */
    PetscInt cells[3] = {n, n, 1};
    ierr = DMPlexCreateBoxMesh(comm, dim, simplex, cells, NULL, NULL, mod->bcs, PETSC_TRUE, &dm);
    CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);
    CHKERRQ(ierr);

    /* set up BCs, functions, tags */
    ierr = DMCreateLabel(dm, "Face Sets");
    CHKERRQ(ierr);

    /* Configuring the DMPLEX object for FVM and distributing over all procs */
    DM dmDist;

    ierr = DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE);
    CHKERRQ(ierr);
    ierr = DMPlexDistribute(dm, overlap, NULL, &dmDist);
    CHKERRQ(ierr);
    if (dmDist) {
        ierr = DMDestroy(&dm);
        CHKERRQ(ierr);
        dm = dmDist;
    }
    /* Constructing the ghost cells for DMPLEX object */
    DM gdm;
    ierr = DMPlexConstructGhostCells(dm, NULL, NULL, &gdm);
    CHKERRQ(ierr);
    ierr = DMDestroy(&dm);
    CHKERRQ(ierr);
    dm = gdm;
    ierr = DMViewFromOptions(dm, NULL, "-dm_view");
    CHKERRQ(ierr);

    /* Creating and configuring the PetscFV object */
    ierr = PetscFVCreate(comm, &phys->fvm);
    CHKERRQ(ierr);
    ierr = PetscFVSetType(phys->fvm, PETSCFVLEASTSQUARES);
    CHKERRQ(ierr);
    ierr = PetscFVSetNumComponents(phys->fvm, phys->dof);
    CHKERRQ(ierr);
    ierr = PetscFVSetSpatialDimension(phys->fvm, dim);
    CHKERRQ(ierr);

    /*....Setting the FV limiter....*/
    PetscLimiter LimiterType = NULL;
    ierr = PetscLimiterCreate(PetscObjectComm((PetscObject) phys->fvm), &LimiterType);
    CHKERRQ(ierr);
    ierr = PetscLimiterSetType(LimiterType, PETSCLIMITERMINMOD);
    CHKERRQ(ierr);
    ierr = PetscLimiterSetFromOptions(LimiterType);
    CHKERRQ(ierr);
    ierr = PetscFVSetLimiter(phys->fvm, LimiterType);
    CHKERRQ(ierr);

    /* Computing gradients */
    PetscBool isgradients = PETSC_FALSE;
    ierr = PetscFVSetComputeGradients(phys->fvm, isgradients);
    CHKERRQ(ierr);

    ierr = PetscObjectSetName((PetscObject) phys->fvm, "");
    CHKERRQ(ierr);
    /* Defining the component name for the PetscFV object */
    ierr = PetscFVSetComponentName(phys->fvm, 0, "h");
    CHKERRQ(ierr);
    ierr = PetscFVSetComponentName(phys->fvm, 1, "uh");
    CHKERRQ(ierr);
    ierr = PetscFVSetFromOptions(phys->fvm);
    CHKERRQ(ierr);

    ierr = PetscLimiterDestroy(&LimiterType);
    CHKERRQ(ierr);

    /* Adding the field and specifying the dof (no. of components) */
    ierr = DMAddField(dm, NULL, (PetscObject) phys->fvm);
    CHKERRQ(ierr);
    ierr = DMCreateDS(dm);
    CHKERRQ(ierr);
    ierr = DMGetDS(dm, &prob);
    CHKERRQ(ierr);
    ierr = PetscDSSetRiemannSolver(prob, 0, user->model->physics->riemann);
    CHKERRQ(ierr);
    ierr = PetscDSSetContext(prob, 0, user->model->physics);
    CHKERRQ(ierr);
    ierr = (*mod->setupbc)(prob, phys);
    CHKERRQ(ierr);
    ierr = PetscDSSetFromOptions(prob);
    CHKERRQ(ierr);

    /* Initializing the solution vector */
    ierr = DMCreateGlobalVector(dm, &X);
    CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) X, "numerical solution");
    CHKERRQ(ierr);

    /* Initializing TS (Time Stepping object) */
    ierr = initializeTS(dm, user, &ts);
    CHKERRQ(ierr);

    /* Setting the initial condition for X */
    ierr = Compute_exact_solution(dm, X, user, 0.0);
    CHKERRQ(ierr);

    /* Setting the dt according to the speed and the smallest mesh width */
    ierr = DMPlexTSGetGeometryFVM(dm, NULL, NULL, &minRadius);
    CHKERRQ(ierr);
    ierr = MPI_Allreduce(&phys->maxspeed, &mod->maxspeed, 1, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject) ts));
    CHKERRQ(ierr);
    dt = mod->cfl * minRadius / mod->maxspeed;

    ierr = TSSetTimeStep(ts, dt);
    CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);
    CHKERRQ(ierr);

    /* March the solution to the required final time */
    ierr = TSSolve(ts, X);
    CHKERRQ(ierr);
    ierr = TSGetSolveTime(ts, &ftime);
    CHKERRQ(ierr);
    ierr = TSGetStepNumber(ts, &nsteps);
    CHKERRQ(ierr);
    /* Getting the current time from the TS object */
    PetscReal sim_time;
    ierr = TSGetTime(ts, &sim_time);
    CHKERRQ(ierr);

    PetscBool view = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL, NULL, "-vecview", &view, NULL);
    CHKERRQ(ierr);
    if (view) {
        ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_MATLAB);
        CHKERRQ(ierr);
        ierr = VecView(X, PETSC_VIEWER_STDOUT_WORLD);
        CHKERRQ(ierr);
    }

    /* Clean up routine */
    ierr = TSDestroy(&ts);
    CHKERRQ(ierr);
    ierr = PetscFree(user->model->physics->data);
    CHKERRQ(ierr);
    ierr = PetscFree(user->model->physics);
    CHKERRQ(ierr);
    ierr = PetscFree(user->model);
    CHKERRQ(ierr);
    ierr = PetscFree(user);
    CHKERRQ(ierr);
    ierr = VecDestroy(&X);
    CHKERRQ(ierr);
    ierr = PetscFVDestroy(&phys->fvm);
    CHKERRQ(ierr);
    ierr = DMDestroy(&dm);
    CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
/* End of main() */

/*TEST

    test:
      args: -ts_max_steps 10

    test:
      suffix: 2
      args: -ts_max_time 5.0

TEST*/
