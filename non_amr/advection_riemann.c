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

PETSC_STATIC_INLINE void Waxpy2Real(PetscReal a, const PetscReal *x, const PetscReal *y, PetscReal *w) {
    w[0] = a * x[0] + y[0];
    w[1] = a * x[1] + y[1];
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
};

/* Defining the context specific for the advection model */
typedef struct {
    PetscReal center[DIM];
    PetscReal radius;
} Physics_Advect_Bump;

typedef struct {
    PetscReal inflowState;
    union {
        Physics_Advect_Bump bump;
    } sol;
    struct {
        PetscInt Solution;
        PetscInt Error;
    } functional;
} Physics_Advect;

/* Functions responsible for defining the boundary cells (elements) */
static PetscErrorCode
PhysicsBoundary_Advect_Inflow(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *xI,
                              PetscScalar *xG, void *ctx) {
    Physics phys = (Physics) ctx;
    Physics_Advect *advect = (Physics_Advect *) phys->data;

    PetscFunctionBeginUser;
    xG[0] = advect->inflowState;
    PetscFunctionReturn(0);
}

static PetscErrorCode
PhysicsBoundary_Advect_Dirichlet(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *xI,
                                 PetscScalar *xG, void *ctx) {
    PetscFunctionBeginUser;
    xG[0] = xI[0];
    PetscFunctionReturn(0);

}

static PetscErrorCode
PhysicsBoundary_Advect_Outflow(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *xI,
                               PetscScalar *xG, void *ctx) {
    PetscFunctionBeginUser;
    xG[0] = xI[0];
    PetscFunctionReturn(0);
}

/* Defining the function responsible for the setting up the BCs */
static PetscErrorCode SetUpBC_Advect(PetscDS prob, Physics phys) {
    PetscErrorCode ierr;
    PetscFunctionBeginUser;

//    const PetscInt inflowids[] = {100, 200, 300}, outflowids[] = {101};
//    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "inflow", "Face Sets", 0, 0, NULL,
//                              (void (*)(void)) PhysicsBoundary_Advect_Inflow, ALEN(inflowids), inflowids, phys);
//    CHKERRQ(ierr);
//    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "outflow", "Face Sets", 0, 0, NULL,
//                              (void (*)(void)) PhysicsBoundary_Advect_Outflow, ALEN(outflowids), outflowids, phys);
//    CHKERRQ(ierr);

    /* Zero dirichlet boundary conditions */
    const PetscInt wallids[] = {1, 2, 4}, inflowids[] = {3};

    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "wall", "Face Sets", 0, 0, NULL,
                              (void (*)(void)) PhysicsBoundary_Advect_Dirichlet, ALEN(wallids), wallids, phys);
    CHKERRQ(ierr);
    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "inflow", "Face Sets", 0, 0, NULL,
                              (void (*)(void)) PhysicsBoundary_Advect_Dirichlet, ALEN(inflowids), inflowids, phys);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

/* Specifying the Riemann function for the advection model */
static void
PhysicsRiemann_Advect(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n, const PetscScalar *xL,
                      const PetscScalar *xR, PetscInt numConstants, const PetscScalar constants[], PetscScalar *flux,
                      Physics phys) {
    PetscReal wind[DIM], wn;

    /* Try to find out why qp (x coodinates at the faces) when defining
     * the wind. */

    wind[0] = 0.1;
    wind[1] = 0.0;

/*
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
        PetscInt maxI = 0;
        PetscReal maxComp2 = comp2[0];

        for (i = 1; i < dim; i++) {
            if (comp2[i] > maxComp2) {
                maxI = i;
                maxComp2 = comp2[i];
            }
        }
        wind[maxI] = 0.;
    }
*/

    wn = Dot2Real(wind, n);
    flux[0] = (wn > 0 ? xL[0] : xR[0]) * wn;
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
static PetscErrorCode PhysicsSolution_Advect(Model mod, PetscReal time, const PetscReal *x, PetscScalar *u, void *ctx) {
    Physics phys = (Physics) ctx;
    Physics_Advect *advect = (Physics_Advect *) phys->data;
    PetscBool bc1 = PETSC_FALSE;

    PetscFunctionBeginUser;
    if (bc1) {
        Physics_Advect_Bump *bump = &advect->sol.bump;
        PetscReal x0[DIM], v[DIM], r, cost, sint;
        cost = PetscCosReal(time);
        sint = PetscSinReal(time);
        x0[0] = cost * x[0] + sint * x[1];
        x0[1] = -sint * x[0] + cost * x[1];
        Waxpy2Real(-1, bump->center, x0, v);
        r = Norm2Real(v);
        u[0] = 0.5 + 0.5 * PetscCosReal(PetscMin(r / bump->radius, 1) * PETSC_PI);
    } else {
        /* Assingning a discontinuous initial condition */
        if (x[0] <= (0.25 + 0.1 * time)) {
            u[0] = 0.5;
        } else u[0] = 0.0;
    }
    PetscFunctionReturn(0);
}

/* Initializing all of the structs related to the advection model */
static PetscErrorCode PhysicsCreate_Advect(Model mod, Physics phys) {
    Physics_Advect *advect;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    phys->riemann = (PetscRiemannFunc) PhysicsRiemann_Advect;
    ierr = PetscNew(&advect);
    CHKERRQ(ierr);
    phys->data = advect;
    mod->setupbc = SetUpBC_Advect;

    Physics_Advect_Bump *bump = &advect->sol.bump;

    bump->center[0] = 0.4;
    bump->center[1] = 0.2;

    bump->radius = 0.1;
    phys->maxspeed = 3.;       /* radius of mesh, kludge */

    /* Initial/transient solution with default boundary conditions */
    mod->solution = PhysicsSolution_Advect;
    mod->solutionctx = phys;

    mod->bcs[0] = mod->bcs[1] = mod->bcs[2] = DM_BOUNDARY_MIRROR;
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
    PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
    MPI_Comm comm;
    PetscDS prob;
    PetscFV fvm;
    PetscLimiter limiter = NULL, LimiterType = NULL;
    User user;
    Model mod;
    Physics phys;
    DM dm;
    PetscReal ftime, cfl, dt, minRadius;
    PetscInt dim, nsteps;
    TS ts;
    Vec X;
    PetscBool simplex = PETSC_FALSE;
    PetscInt overlap;
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


    /* Number of cells to overlap partitions */
    overlap = 1;

    /* Initializing the structures for the advection model */
    ierr = PhysicsCreate_Advect(mod, phys);
    CHKERRQ(ierr);

    /* Setting the number of fields and dof for the model */
    phys->dof = 1;

    /* Mesh creation routine */
    size_t i;
    for (i = 0; i < DIM; i++) {
        mod->bounds[2 * i] = 0.;
        mod->bounds[2 * i + 1] = 1.;
    }

    /* Defining the dimension of the domain */
    dim = 2;
    PetscInt n = 10;
    ierr = PetscOptionsGetInt(NULL, NULL, "-mesh", &n, NULL);
    CHKERRQ(ierr);
    /* Setting a default cfl value */
    cfl = 1.0;
    ierr = PetscOptionsGetReal(NULL, NULL, "-cfl", &cfl, NULL);
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
    ierr = PetscFVCreate(comm, &fvm);
    CHKERRQ(ierr);
    ierr = PetscFVSetFromOptions(fvm);
    CHKERRQ(ierr);
    ierr = PetscFVSetNumComponents(fvm, phys->dof);
    CHKERRQ(ierr);
    ierr = PetscFVSetSpatialDimension(fvm, dim);
    CHKERRQ(ierr);

    /*....Setting the FV limiter....*/
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
    /* Defining the component name for the PetscFV object */
    ierr = PetscFVSetComponentName(fvm, phys->dof, "q");
    CHKERRQ(ierr);

    /* Adding the field and specifying the dof (no. of components) */
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

    /* Initializing TS (Time Stepping object) */
    ierr = initializeTS(dm, user, &ts);
    CHKERRQ(ierr);


    /* Initializing the solution vector */
    ierr = DMCreateGlobalVector(dm, &X);
    CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) X, "numerical solution");
    CHKERRQ(ierr);

    /* Creating the vector to contain the exact solution */
    Vec X_exact;
    ierr = VecDuplicate(X, &X_exact);
    CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) X_exact, "exact solution");
    CHKERRQ(ierr);

    /* Setting the initial condition for X */
    ierr = Compute_exact_solution(dm, X, user, 0.0);
    CHKERRQ(ierr);

    /* Setting the dt according to the speed and the smallest mesh width */
    ierr = DMPlexTSGetGeometryFVM(dm, NULL, NULL, &minRadius);
    CHKERRQ(ierr);
    ierr = MPI_Allreduce(&phys->maxspeed, &mod->maxspeed, 1, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject) ts));
    CHKERRQ(ierr);
    dt = cfl * minRadius / mod->maxspeed;
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

    /* Computing the exact solution at time sim_time */
    ierr = Compute_exact_solution(dm, X_exact, user, sim_time);
    CHKERRQ(ierr);

    /* View the vector */
//    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_MATLAB);
//    CHKERRQ(ierr);
//    ierr = VecView(X_exact, PETSC_VIEWER_STDOUT_WORLD);
//    CHKERRQ(ierr);
//
//    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_MATLAB);
//    CHKERRQ(ierr);
//    ierr = VecView(X, PETSC_VIEWER_STDOUT_WORLD);
//    CHKERRQ(ierr);

    /* Compute the norm of the error */
    ierr = VecAXPY(X_exact, -1.0, X);
    CHKERRQ(ierr);
    PetscReal err_norm;
    ierr = VecNorm(X_exact, NORM_INFINITY, &err_norm);
    CHKERRQ(ierr);
    printf("The error norm is %f \n", err_norm);

    /* Clean up routine */
//    ierr = PetscViewerPopFormat(viewer);
//    CHKERRQ(ierr);
//    ierr = PetscViewerDestroy(&viewer);
//    CHKERRQ(ierr);
    ierr = TSDestroy(&ts);
    CHKERRQ(ierr);

    ierr = PetscFree(user->model->functionalMonitored);
    CHKERRQ(ierr);
    ierr = PetscFree(user->model->functionalCall);
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
    ierr = PetscLimiterDestroy(&limiter);
    CHKERRQ(ierr);
    ierr = PetscLimiterDestroy(&LimiterType);
    CHKERRQ(ierr);
    ierr = PetscFVDestroy(&fvm);
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
