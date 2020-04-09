static char help[] = "Nonlinear, time-dependent PDE in 2d.\n";

#include <petscdmplex.h>
#include <petscds.h>
#include <petscts.h>
#include <petscsf.h> /* For SplitFaces() */
#include <petscviewerhdf5.h>
#include <petscdmforest.h>

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
    PetscInt numSplitFaces;
    PetscInt monitorStepOffset;
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
    void *errorCtx;

    PetscErrorCode
    (*errorIndicator)(PetscInt, PetscReal, PetscInt, const PetscScalar[], const PetscScalar[], PetscReal *, void *);
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
    cR = PetscSqrtReal(sw->gravity * uR->h);
    cL = PetscSqrtReal(sw->gravity * uL->h);
    /* gravity wave speed */
    speed = PetscMax(PetscAbsReal(Dot2Real(uL->uh, nn) / uL->h) + cL,
                     PetscAbsReal(Dot2Real(uR->uh, nn) / uR->h) + cR);
    for (i = 0; i < 1 + dim; i++)
        flux[i] = (0.5 * (fL.vals[i] + fR.vals[i]) + 0.5 * speed * (xL[i] - xR[i])) * Norm2Real(n);
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

    PetscInt vtkInterval = 2;
    char outputBasename[256] = "swe_amr", filename[PETSC_MAX_PATH_LEN];
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

static PetscErrorCode ErrorIndicator_Simple(PetscInt dim, PetscReal volume, PetscInt numComps, const PetscScalar u[],
                                            const PetscScalar grad[], PetscReal *error, void *ctx) {
    PetscReal err = 0.;
    PetscInt i, j;

    PetscFunctionBeginUser;
    for (i = 0; i < numComps; i++) {
        for (j = 0; j < dim; j++) {
            err += PetscSqr(PetscRealPart(grad[i * dim + j]));
        }
    }
    *error = volume * err;
    PetscFunctionReturn(0);
}

/* Right now, I have just added duplicate faces, which see both cells. We can
- Add duplicate vertices and decouple the face cones
- Disconnect faces from cells across the rotation gap
*/
PetscErrorCode SplitFaces(DM *dmSplit, const char labelName[], User user) {
    DM dm = *dmSplit, sdm;
    PetscSF sfPoint, gsfPoint;
    PetscSection coordSection, newCoordSection;
    Vec coordinates;
    IS idIS;
    const PetscInt *ids;
    PetscInt *newpoints;
    PetscInt dim, depth, maxConeSize, maxSupportSize, numLabels, numGhostCells;
    PetscInt numFS, fs, pStart, pEnd, p, cEnd, cEndInterior, vStart, vEnd, v, fStart, fEnd, newf, d, l;
    PetscBool hasLabel;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = DMHasLabel(dm, labelName, &hasLabel);
    CHKERRQ(ierr);
    if (!hasLabel) PetscFunctionReturn(0);
    ierr = DMCreate(PetscObjectComm((PetscObject) dm), &sdm);
    CHKERRQ(ierr);
    ierr = DMSetType(sdm, DMPLEX);
    CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);
    CHKERRQ(ierr);
    ierr = DMSetDimension(sdm, dim);
    CHKERRQ(ierr);

    ierr = DMGetLabelIdIS(dm, labelName, &idIS);
    CHKERRQ(ierr);
    ierr = ISGetLocalSize(idIS, &numFS);
    CHKERRQ(ierr);
    ierr = ISGetIndices(idIS, &ids);
    CHKERRQ(ierr);

    user->numSplitFaces = 0;
    for (fs = 0; fs < numFS; ++fs) {
        PetscInt numBdFaces;
        ierr = DMGetStratumSize(dm, labelName, ids[fs], &numBdFaces);
        CHKERRQ(ierr);
        user->numSplitFaces += numBdFaces;
    }
    ierr = DMPlexGetChart(dm, &pStart, &pEnd);
    CHKERRQ(ierr);
    pEnd += user->numSplitFaces;
    ierr = DMPlexSetChart(sdm, pStart, pEnd);
    CHKERRQ(ierr);
    ierr = DMPlexGetGhostCellStratum(dm, &cEndInterior, NULL);
    CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, NULL, &cEnd);
    CHKERRQ(ierr);
    numGhostCells = cEnd - cEndInterior;
    /* Set cone and support sizes */
    ierr = DMPlexGetDepth(dm, &depth);
    CHKERRQ(ierr);
    for (d = 0; d <= depth; ++d) {
        ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);
        CHKERRQ(ierr);
        for (p = pStart; p < pEnd; ++p) {
            PetscInt newp = p;
            PetscInt size;

            ierr = DMPlexGetConeSize(dm, p, &size);
            CHKERRQ(ierr);
            ierr = DMPlexSetConeSize(sdm, newp, size);
            CHKERRQ(ierr);
            ierr = DMPlexGetSupportSize(dm, p, &size);
            CHKERRQ(ierr);
            ierr = DMPlexSetSupportSize(sdm, newp, size);
            CHKERRQ(ierr);
        }
    }
    ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);
    CHKERRQ(ierr);
    for (fs = 0, newf = fEnd; fs < numFS; ++fs) {
        IS faceIS;
        const PetscInt *faces;
        PetscInt numFaces, f;

        ierr = DMGetStratumIS(dm, labelName, ids[fs], &faceIS);
        CHKERRQ(ierr);
        ierr = ISGetLocalSize(faceIS, &numFaces);
        CHKERRQ(ierr);
        ierr = ISGetIndices(faceIS, &faces);
        CHKERRQ(ierr);
        for (f = 0; f < numFaces; ++f, ++newf) {
            PetscInt size;

            /* Right now I think that both faces should see both cells */
            ierr = DMPlexGetConeSize(dm, faces[f], &size);
            CHKERRQ(ierr);
            ierr = DMPlexSetConeSize(sdm, newf, size);
            CHKERRQ(ierr);
            ierr = DMPlexGetSupportSize(dm, faces[f], &size);
            CHKERRQ(ierr);
            ierr = DMPlexSetSupportSize(sdm, newf, size);
            CHKERRQ(ierr);
        }
        ierr = ISRestoreIndices(faceIS, &faces);
        CHKERRQ(ierr);
        ierr = ISDestroy(&faceIS);
        CHKERRQ(ierr);
    }
    ierr = DMSetUp(sdm);
    CHKERRQ(ierr);
    /* Set cones and supports */
    ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);
    CHKERRQ(ierr);
    ierr = PetscMalloc1(PetscMax(maxConeSize, maxSupportSize), &newpoints);
    CHKERRQ(ierr);
    ierr = DMPlexGetChart(dm, &pStart, &pEnd);
    CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
        const PetscInt *points, *orientations;
        PetscInt size, i, newp = p;

        ierr = DMPlexGetConeSize(dm, p, &size);
        CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, p, &points);
        CHKERRQ(ierr);
        ierr = DMPlexGetConeOrientation(dm, p, &orientations);
        CHKERRQ(ierr);
        for (i = 0; i < size; ++i) newpoints[i] = points[i];
        ierr = DMPlexSetCone(sdm, newp, newpoints);
        CHKERRQ(ierr);
        ierr = DMPlexSetConeOrientation(sdm, newp, orientations);
        CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, p, &size);
        CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, p, &points);
        CHKERRQ(ierr);
        for (i = 0; i < size; ++i) newpoints[i] = points[i];
        ierr = DMPlexSetSupport(sdm, newp, newpoints);
        CHKERRQ(ierr);
    }
    ierr = PetscFree(newpoints);
    CHKERRQ(ierr);
    for (fs = 0, newf = fEnd; fs < numFS; ++fs) {
        IS faceIS;
        const PetscInt *faces;
        PetscInt numFaces, f;

        ierr = DMGetStratumIS(dm, labelName, ids[fs], &faceIS);
        CHKERRQ(ierr);
        ierr = ISGetLocalSize(faceIS, &numFaces);
        CHKERRQ(ierr);
        ierr = ISGetIndices(faceIS, &faces);
        CHKERRQ(ierr);
        for (f = 0; f < numFaces; ++f, ++newf) {
            const PetscInt *points;

            ierr = DMPlexGetCone(dm, faces[f], &points);
            CHKERRQ(ierr);
            ierr = DMPlexSetCone(sdm, newf, points);
            CHKERRQ(ierr);
            ierr = DMPlexGetSupport(dm, faces[f], &points);
            CHKERRQ(ierr);
            ierr = DMPlexSetSupport(sdm, newf, points);
            CHKERRQ(ierr);
        }
        ierr = ISRestoreIndices(faceIS, &faces);
        CHKERRQ(ierr);
        ierr = ISDestroy(&faceIS);
        CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(idIS, &ids);
    CHKERRQ(ierr);
    ierr = ISDestroy(&idIS);
    CHKERRQ(ierr);
    ierr = DMPlexStratify(sdm);
    CHKERRQ(ierr);
    ierr = DMPlexSetGhostCellStratum(sdm, cEndInterior, PETSC_DETERMINE);
    CHKERRQ(ierr);
    /* Convert coordinates */
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);
    CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(dm, &coordSection);
    CHKERRQ(ierr);
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &newCoordSection);
    CHKERRQ(ierr);
    ierr = PetscSectionSetNumFields(newCoordSection, 1);
    CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(newCoordSection, 0, dim);
    CHKERRQ(ierr);
    ierr = PetscSectionSetChart(newCoordSection, vStart, vEnd);
    CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
        ierr = PetscSectionSetDof(newCoordSection, v, dim);
        CHKERRQ(ierr);
        ierr = PetscSectionSetFieldDof(newCoordSection, v, 0, dim);
        CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(newCoordSection);
    CHKERRQ(ierr);
    ierr = DMSetCoordinateSection(sdm, PETSC_DETERMINE, newCoordSection);
    CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&newCoordSection);
    CHKERRQ(ierr); /* relinquish our reference */
    ierr = DMGetCoordinatesLocal(dm, &coordinates);
    CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(sdm, coordinates);
    CHKERRQ(ierr);
    /* Convert labels */
    ierr = DMGetNumLabels(dm, &numLabels);
    CHKERRQ(ierr);
    for (l = 0; l < numLabels; ++l) {
        const char *lname;
        PetscBool isDepth, isDim;

        ierr = DMGetLabelName(dm, l, &lname);
        CHKERRQ(ierr);
        ierr = PetscStrcmp(lname, "depth", &isDepth);
        CHKERRQ(ierr);
        if (isDepth) continue;
        ierr = PetscStrcmp(lname, "dim", &isDim);
        CHKERRQ(ierr);
        if (isDim) continue;
        ierr = DMCreateLabel(sdm, lname);
        CHKERRQ(ierr);
        ierr = DMGetLabelIdIS(dm, lname, &idIS);
        CHKERRQ(ierr);
        ierr = ISGetLocalSize(idIS, &numFS);
        CHKERRQ(ierr);
        ierr = ISGetIndices(idIS, &ids);
        CHKERRQ(ierr);
        for (fs = 0; fs < numFS; ++fs) {
            IS pointIS;
            const PetscInt *points;
            PetscInt numPoints;

            ierr = DMGetStratumIS(dm, lname, ids[fs], &pointIS);
            CHKERRQ(ierr);
            ierr = ISGetLocalSize(pointIS, &numPoints);
            CHKERRQ(ierr);
            ierr = ISGetIndices(pointIS, &points);
            CHKERRQ(ierr);
            for (p = 0; p < numPoints; ++p) {
                PetscInt newpoint = points[p];

                ierr = DMSetLabelValue(sdm, lname, newpoint, ids[fs]);
                CHKERRQ(ierr);
            }
            ierr = ISRestoreIndices(pointIS, &points);
            CHKERRQ(ierr);
            ierr = ISDestroy(&pointIS);
            CHKERRQ(ierr);
        }
        ierr = ISRestoreIndices(idIS, &ids);
        CHKERRQ(ierr);
        ierr = ISDestroy(&idIS);
        CHKERRQ(ierr);
    }
    {
        /* Convert pointSF */
        const PetscSFNode *remotePoints;
        PetscSFNode *gremotePoints;
        const PetscInt *localPoints;
        PetscInt *glocalPoints, *newLocation, *newRemoteLocation;
        PetscInt numRoots, numLeaves;
        PetscMPIInt size;

        ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);
        CHKERRQ(ierr);
        ierr = DMGetPointSF(dm, &sfPoint);
        CHKERRQ(ierr);
        ierr = DMGetPointSF(sdm, &gsfPoint);
        CHKERRQ(ierr);
        ierr = DMPlexGetChart(dm, &pStart, &pEnd);
        CHKERRQ(ierr);
        ierr = PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints);
        CHKERRQ(ierr);
        if (numRoots >= 0) {
            ierr = PetscMalloc2(numRoots, &newLocation, pEnd - pStart, &newRemoteLocation);
            CHKERRQ(ierr);
            for (l = 0; l < numRoots; l++) newLocation[l] = l; /* + (l >= cEnd ? numGhostCells : 0); */
            ierr = PetscSFBcastBegin(sfPoint, MPIU_INT, newLocation, newRemoteLocation);
            CHKERRQ(ierr);
            ierr = PetscSFBcastEnd(sfPoint, MPIU_INT, newLocation, newRemoteLocation);
            CHKERRQ(ierr);
            ierr = PetscMalloc1(numLeaves, &glocalPoints);
            CHKERRQ(ierr);
            ierr = PetscMalloc1(numLeaves, &gremotePoints);
            CHKERRQ(ierr);
            for (l = 0; l < numLeaves; ++l) {
                glocalPoints[l] = localPoints[l]; /* localPoints[l] >= cEnd ? localPoints[l] + numGhostCells : localPoints[l]; */
                gremotePoints[l].rank = remotePoints[l].rank;
                gremotePoints[l].index = newRemoteLocation[localPoints[l]];
            }
            ierr = PetscFree2(newLocation, newRemoteLocation);
            CHKERRQ(ierr);
            ierr = PetscSFSetGraph(gsfPoint, numRoots + numGhostCells, numLeaves, glocalPoints, PETSC_OWN_POINTER,
                                   gremotePoints, PETSC_OWN_POINTER);
            CHKERRQ(ierr);
        }
        ierr = DMDestroy(dmSplit);
        CHKERRQ(ierr);
        *dmSplit = sdm;
    }
    PetscFunctionReturn(0);
}

static PetscErrorCode
adaptToleranceFVM(PetscFV fvm, TS ts, Vec sol, VecTagger refineTag, VecTagger coarsenTag, User user, TS *tsNew,
                  Vec *solNew) {
    DM dm, gradDM, plex, cellDM, adaptedDM = NULL;
    Vec cellGeom, faceGeom;
    PetscBool isForest, computeGradient;
    Vec grad, locGrad, locX, errVec;
    PetscInt cStart, cEnd, c, dim, nRefine, nCoarsen;
    PetscReal minMaxInd[2] = {PETSC_MAX_REAL, PETSC_MIN_REAL}, minMaxIndGlobal[2], minInd, maxInd, time;
    PetscScalar *errArray;
    const PetscScalar *pointVals;
    const PetscScalar *pointGrads;
    const PetscScalar *pointGeom;
    DMLabel adaptLabel = NULL;
    IS refineIS, coarsenIS;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = TSGetTime(ts, &time);
    CHKERRQ(ierr);
    ierr = VecGetDM(sol, &dm);
    CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);
    CHKERRQ(ierr);
    ierr = PetscFVGetComputeGradients(fvm, &computeGradient);
    CHKERRQ(ierr);
    ierr = PetscFVSetComputeGradients(fvm, PETSC_TRUE);
    CHKERRQ(ierr);
    ierr = DMIsForest(dm, &isForest);
    CHKERRQ(ierr);
    ierr = DMConvert(dm, DMPLEX, &plex);
    CHKERRQ(ierr);
    ierr = DMPlexGetDataFVM(plex, fvm, &cellGeom, &faceGeom, &gradDM);
    CHKERRQ(ierr);
    ierr = DMCreateLocalVector(plex, &locX);
    CHKERRQ(ierr);
    ierr = DMPlexInsertBoundaryValues(plex, PETSC_TRUE, locX, 0.0, faceGeom, cellGeom, NULL);
    CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(plex, sol, INSERT_VALUES, locX);
    CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(plex, sol, INSERT_VALUES, locX);
    CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(gradDM, &grad);
    CHKERRQ(ierr);
    /* Reconstructing Gradients using dm, local vector and gradient vector */
    ierr = DMPlexReconstructGradientsFVM(plex, locX, grad);
    CHKERRQ(ierr);
    ierr = DMCreateLocalVector(gradDM, &locGrad);
    CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(gradDM, grad, INSERT_VALUES, locGrad);
    CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(gradDM, grad, INSERT_VALUES, locGrad);
    CHKERRQ(ierr);
    ierr = VecDestroy(&grad);
    CHKERRQ(ierr);
    /* Only using the local gradients and local vector. */
    /* Obtaining the normal cell ranges for each processor */
    ierr = DMPlexGetInteriorCellStratum(plex, &cStart, &cEnd);
    CHKERRQ(ierr);
    ierr = VecGetArrayRead(locGrad, &pointGrads);
    CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellGeom, &pointGeom);
    CHKERRQ(ierr);
    ierr = VecGetArrayRead(locX, &pointVals);
    CHKERRQ(ierr);
    /* Getting the cell dm from the vec obtained from the FV object */
    ierr = VecGetDM(cellGeom, &cellDM);
    CHKERRQ(ierr);
    ierr = DMLabelCreate(PETSC_COMM_SELF, "adapt", &adaptLabel);
    CHKERRQ(ierr);
    ierr = VecCreateMPI(PetscObjectComm((PetscObject) plex), cEnd - cStart, PETSC_DETERMINE, &errVec);
    CHKERRQ(ierr);
    ierr = VecSetUp(errVec);
    CHKERRQ(ierr);
    ierr = VecGetArray(errVec, &errArray);
    CHKERRQ(ierr);
    for (c = cStart; c < cEnd; c++) {
        PetscReal errInd = 0.;
        PetscScalar *pointGrad;
        PetscScalar *pointVal;
        PetscFVCellGeom *cg;

        /* Obtaining the gradient, geometry and value at each cell */
        ierr = DMPlexPointLocalRead(gradDM, c, pointGrads, &pointGrad);
        CHKERRQ(ierr);
        ierr = DMPlexPointLocalRead(cellDM, c, pointGeom, &cg);
        CHKERRQ(ierr);
        ierr = DMPlexPointLocalRead(plex, c, pointVals, &pointVal);
        CHKERRQ(ierr);

        /* Getting the adapting criteria as the product of volume and norm of the gradients */
        ierr = (user->model->errorIndicator)(dim, cg->volume, user->model->physics->dof, pointVal, pointGrad, &errInd,
                                             user->model->errorCtx);
        CHKERRQ(ierr);
        errArray[c - cStart] = errInd;
        /* To prevent from obtaining NaN values */
        minMaxInd[0] = PetscMin(minMaxInd[0], errInd);
        minMaxInd[1] = PetscMax(minMaxInd[1], errInd);
    }
    ierr = VecRestoreArray(errVec, &errArray);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(locX, &pointVals);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(cellGeom, &pointGeom);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(locGrad, &pointGrads);
    CHKERRQ(ierr);
    ierr = VecDestroy(&locGrad);
    CHKERRQ(ierr);
    ierr = VecDestroy(&locX);
    CHKERRQ(ierr);
    ierr = DMDestroy(&plex);
    CHKERRQ(ierr);

    /* Provide the function with the Tag, the norms in the cells, and resulting tags */
    ierr = VecTaggerComputeIS(refineTag, errVec, &refineIS);
    CHKERRQ(ierr);
    ierr = VecTaggerComputeIS(coarsenTag, errVec, &coarsenIS);
    CHKERRQ(ierr);
    ierr = ISGetSize(refineIS, &nRefine);
    CHKERRQ(ierr);
    ierr = ISGetSize(coarsenIS, &nCoarsen);
    CHKERRQ(ierr);
    /* DM_ADAPT_REFINE - 1, DM_ADAPT_COARSE - 0 */
    if (nRefine) {
        ierr = DMLabelSetStratumIS(adaptLabel, DM_ADAPT_REFINE, refineIS);
        CHKERRQ(ierr);
    }
    if (nCoarsen) {
        ierr = DMLabelSetStratumIS(adaptLabel, DM_ADAPT_COARSEN, coarsenIS);
        CHKERRQ(ierr);
    }
    ierr = ISDestroy(&coarsenIS);
    CHKERRQ(ierr);
    ierr = ISDestroy(&refineIS);
    CHKERRQ(ierr);
    ierr = VecDestroy(&errVec);
    CHKERRQ(ierr);

    ierr = PetscFVSetComputeGradients(fvm, computeGradient);
    CHKERRQ(ierr);
    minMaxInd[1] = -minMaxInd[1];
    ierr = MPI_Allreduce(minMaxInd, minMaxIndGlobal, 2, MPIU_REAL, MPI_MIN, PetscObjectComm((PetscObject) dm));
    CHKERRQ(ierr);
    minInd = minMaxIndGlobal[0];
    maxInd = -minMaxIndGlobal[1];
    ierr = PetscInfo2(ts, "error indicator range (%E, %E)\n", minInd, maxInd);
    CHKERRQ(ierr);
    if (nRefine || nCoarsen) { /* at least one cell is over the refinement threshold */
        /* Converting/adapting the DM from the old mesh to the new mesh */
        ierr = DMAdaptLabel(dm, adaptLabel, &adaptedDM);
        CHKERRQ(ierr);
    }
    ierr = DMLabelDestroy(&adaptLabel);
    CHKERRQ(ierr);
    if (adaptedDM) {
        ierr = PetscInfo2(ts, "Adapted mesh, marking %D cells for refinement, and %D cells for coarsening\n", nRefine,
                          nCoarsen);
        CHKERRQ(ierr);
        if (tsNew) {
            ierr = initializeTS(adaptedDM, user, tsNew);
            CHKERRQ(ierr);
        }
        if (solNew) {
            /* Creating an empty solution vector with the same size as the adoptedDM */
            ierr = DMCreateGlobalVector(adaptedDM, solNew);
            CHKERRQ(ierr);
            ierr = PetscObjectSetName((PetscObject) *solNew, "solution");
            CHKERRQ(ierr);
            /* Transferring/Interpolating the solution to the new dm (persumably cell centers) */
            ierr = DMForestTransferVec(dm, sol, adaptedDM, *solNew, PETSC_TRUE, time);
            CHKERRQ(ierr);
        }
        /* clear internal references to the previous dm */
        if (isForest) {
            ierr = DMForestSetAdaptivityForest(adaptedDM, NULL);
            CHKERRQ(ierr);
        }
        ierr = DMDestroy(&adaptedDM);
        CHKERRQ(ierr);
    } else {
        if (tsNew) *tsNew = NULL;
        if (solNew) *solNew = NULL;
    }
    PetscFunctionReturn(0);
}

static PetscErrorCode PostStep(TS ts) {
    PetscErrorCode ierr;
    PetscInt stepi;
    Vec X;
    PetscReal time;
    DM dm;
    PetscFunctionBegin;
    ierr = TSGetSolution(ts, &X);
    CHKERRQ(ierr);
    ierr = VecGetDM(X, &dm);
    CHKERRQ(ierr);
    ierr = DMGetOutputSequenceNumber(dm, &stepi, NULL);
    CHKERRQ(ierr);
    ierr = TSGetTime(ts, &time);
    CHKERRQ(ierr);
    ierr = DMSetOutputSequenceNumber(dm, stepi + 1, time);
    CHKERRQ(ierr); // stay ahead of initial solution
    ierr = VecViewFromOptions(X, NULL, "-vec_view");
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
    MPI_Comm comm;
    PetscDS prob = NULL;
    User user;
    Model mod;
    Physics phys = NULL;
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

    PetscBool useAMR = PETSC_TRUE;
    PetscInt adaptInterval = 4;
    ierr = PetscOptionsBegin(comm, NULL, "Unstructured Finite Volume Mesh Options", "");
            CHKERRQ(ierr);
            {
                ierr = PetscOptionsBool("-ufv_use_amr", "use local adaptive mesh refinement", "", useAMR, &useAMR,
                                        NULL);
                CHKERRQ(ierr);
                ierr = PetscOptionsInt("-ufv_adapt_interval", "time steps between AMR", "", adaptInterval,
                                       &adaptInterval, NULL);
                CHKERRQ(ierr);
            }
            ierr = PetscOptionsEnd();
    CHKERRQ(ierr);

    VecTaggerBox refineBox, coarsenBox;
    VecTagger refineTag = NULL, coarsenTag = NULL;
    if (useAMR) {
        refineBox.min = refineBox.max = PETSC_MAX_REAL;
        coarsenBox.min = coarsenBox.max = PETSC_MIN_REAL;

        ierr = VecTaggerCreate(comm, &refineTag);
        CHKERRQ(ierr);
        ierr = PetscObjectSetOptionsPrefix((PetscObject) refineTag, "refine_");
        CHKERRQ(ierr);
        ierr = VecTaggerSetType(refineTag, VECTAGGERABSOLUTE);
        CHKERRQ(ierr);
        ierr = VecTaggerAbsoluteSetBox(refineTag, &refineBox);
        CHKERRQ(ierr);
        ierr = VecTaggerSetFromOptions(refineTag);
        CHKERRQ(ierr);
        ierr = VecTaggerSetUp(refineTag);
        CHKERRQ(ierr);
        ierr = PetscObjectViewFromOptions((PetscObject) refineTag, NULL, "-tag_view");
        CHKERRQ(ierr);

        ierr = VecTaggerCreate(comm, &coarsenTag);
        CHKERRQ(ierr);
        ierr = PetscObjectSetOptionsPrefix((PetscObject) coarsenTag, "coarsen_");
        CHKERRQ(ierr);
        ierr = VecTaggerSetType(coarsenTag, VECTAGGERABSOLUTE);
        CHKERRQ(ierr);
        ierr = VecTaggerAbsoluteSetBox(coarsenTag, &coarsenBox);
        CHKERRQ(ierr);
        ierr = VecTaggerSetFromOptions(coarsenTag);
        CHKERRQ(ierr);
        ierr = VecTaggerSetUp(coarsenTag);
        CHKERRQ(ierr);
        ierr = PetscObjectViewFromOptions((PetscObject) coarsenTag, NULL, "-tag_view");
        CHKERRQ(ierr);
    }

    /* Initializing the structures for the shallow water model */
    ierr = PhysicsCreate_SW(mod, phys);
    CHKERRQ(ierr);

    phys->nfields = 2;
    phys->dof = 3;

    PetscBool simplex = PETSC_FALSE;
    PetscInt n = 5, overlap = 1;
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

    /* set up BCs, functions, tags */
    ierr = DMCreateLabel(dm, "Face Sets");
    CHKERRQ(ierr);

    ierr = DMViewFromOptions(dm, NULL, "-orig_dm_view");CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    mod->errorIndicator = ErrorIndicator_Simple;

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

    ierr = DMSetFromOptions(dm);
    CHKERRQ(ierr);
    /* Constructing the ghost cells for DMPLEX object */
    DM gdm;
    ierr = DMPlexConstructGhostCells(dm, NULL, NULL, &gdm);
    CHKERRQ(ierr);
    ierr = DMDestroy(&dm);
    CHKERRQ(ierr);
    dm = gdm;
    ierr = DMViewFromOptions(dm, NULL, "-dm_view");
    CHKERRQ(ierr);

    ierr = SplitFaces(&dm, "split faces", user);
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
    PetscLimiter LimiterType = NULL, noneLimiter = NULL;
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

    /* Initializing TS (Time Stepping object) */
    ierr = initializeTS(dm, user, &ts);
    CHKERRQ(ierr);

    /* Initializing the solution vector */
    ierr = DMCreateGlobalVector(dm, &X);
    CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) X, "numerical solution");
    CHKERRQ(ierr);

    /* Setting the initial condition for X */
    ierr = Compute_exact_solution(dm, X, user, 0.0);
    CHKERRQ(ierr);

    if (useAMR) {
        PetscInt adaptIter;

        /* use no limiting when reconstructing gradients for adaptivity */
        ierr = PetscFVGetLimiter(phys->fvm, &LimiterType);
        CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject) LimiterType);
        CHKERRQ(ierr);
        ierr = PetscLimiterCreate(PetscObjectComm((PetscObject) phys->fvm), &noneLimiter);
        CHKERRQ(ierr);
        ierr = PetscLimiterSetType(noneLimiter, PETSCLIMITERNONE);
        CHKERRQ(ierr);

        ierr = PetscFVSetLimiter(phys->fvm, noneLimiter);
        CHKERRQ(ierr);
        for (adaptIter = 0;; ++adaptIter) {
            PetscLogDouble bytes;
            TS tsNew = NULL;

            ierr = PetscMemoryGetCurrentUsage(&bytes);
            CHKERRQ(ierr);
            ierr = PetscInfo2(ts, "refinement loop %D: memory used %g\n", adaptIter, bytes);
            CHKERRQ(ierr);
            ierr = adaptToleranceFVM(phys->fvm, ts, X, refineTag, coarsenTag, user, &tsNew, NULL);
            CHKERRQ(ierr);
            if (!tsNew) {
                break;
            } else {
                ierr = DMDestroy(&dm);
                CHKERRQ(ierr);
                ierr = VecDestroy(&X);
                CHKERRQ(ierr);
                ierr = TSDestroy(&ts);
                CHKERRQ(ierr);
                ts = tsNew;
                ierr = TSGetDM(ts, &dm);
                CHKERRQ(ierr);
                ierr = PetscObjectReference((PetscObject) dm);
                CHKERRQ(ierr);
                ierr = DMCreateGlobalVector(dm, &X);
                CHKERRQ(ierr);
                ierr = PetscObjectSetName((PetscObject) X, "solution");
                CHKERRQ(ierr);
                ierr = Compute_exact_solution(dm, X, user, 0.0);
                CHKERRQ(ierr);
            }
        }
        /* restore original limiter */
        ierr = PetscFVSetLimiter(phys->fvm, LimiterType);
        CHKERRQ(ierr);
    }

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
    if (!useAMR) {
        ierr = TSSolve(ts, X);
        CHKERRQ(ierr);
        ierr = TSGetSolveTime(ts, &ftime);
        CHKERRQ(ierr);
        ierr = TSGetStepNumber(ts, &nsteps);
        CHKERRQ(ierr);
    } else {
        PetscReal finalTime;
        PetscInt adaptIter;
        TS tsNew = NULL;
        Vec solNew = NULL;

        ierr = TSGetMaxTime(ts, &finalTime);
        CHKERRQ(ierr);
        ierr = TSSetMaxSteps(ts, adaptInterval);
        CHKERRQ(ierr);
        ierr = TSSolve(ts, X);
        CHKERRQ(ierr);
        ierr = TSGetSolveTime(ts, &ftime);
        CHKERRQ(ierr);
        ierr = TSGetStepNumber(ts, &nsteps);
        CHKERRQ(ierr);
        for (adaptIter = 0; ftime < finalTime; adaptIter++) {
            PetscLogDouble bytes;

            ierr = PetscMemoryGetCurrentUsage(&bytes);
            CHKERRQ(ierr);
            ierr = PetscInfo2(ts, "AMR time step loop %D: memory used %g\n", adaptIter, bytes);
            CHKERRQ(ierr);
            ierr = PetscFVSetLimiter(phys->fvm, noneLimiter);
            CHKERRQ(ierr);
            ierr = adaptToleranceFVM(phys->fvm, ts, X, refineTag, coarsenTag, user, &tsNew, &solNew);
            CHKERRQ(ierr);
            ierr = PetscFVSetLimiter(phys->fvm, LimiterType);
            CHKERRQ(ierr);
            if (tsNew) {
                ierr = PetscInfo(ts, "AMR used\n");
                CHKERRQ(ierr);
                ierr = DMDestroy(&dm);
                CHKERRQ(ierr);
                ierr = VecDestroy(&X);
                CHKERRQ(ierr);
                ierr = TSDestroy(&ts);
                CHKERRQ(ierr);
                ts = tsNew;
                X = solNew;
                ierr = TSSetFromOptions(ts);
                CHKERRQ(ierr);
                ierr = VecGetDM(X, &dm);
                CHKERRQ(ierr);
                ierr = PetscObjectReference((PetscObject) dm);
                CHKERRQ(ierr);
                ierr = DMPlexTSGetGeometryFVM(dm, NULL, NULL, &minRadius);
                CHKERRQ(ierr);
                ierr = MPI_Allreduce(&phys->maxspeed, &mod->maxspeed, 1, MPIU_REAL, MPIU_MAX,
                                     PetscObjectComm((PetscObject) ts));
                CHKERRQ(ierr);
                dt = mod->cfl * minRadius / mod->maxspeed;
                ierr = TSSetStepNumber(ts, nsteps);
                CHKERRQ(ierr);
                ierr = TSSetTime(ts, ftime);
                CHKERRQ(ierr);
                ierr = TSSetTimeStep(ts, dt);
                CHKERRQ(ierr);
            } else {
                ierr = PetscInfo(ts, "AMR not used\n");
                CHKERRQ(ierr);
            }
            user->monitorStepOffset = nsteps;
            ierr = TSSetMaxSteps(ts, nsteps + adaptInterval);
            CHKERRQ(ierr);
            ierr = TSSolve(ts, X);
            CHKERRQ(ierr);
            ierr = TSGetSolveTime(ts, &ftime);
            CHKERRQ(ierr);
            ierr = TSGetStepNumber(ts, &nsteps);
            CHKERRQ(ierr);
        }
    }

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
    ierr = VecTaggerDestroy(&refineTag);
    CHKERRQ(ierr);
    ierr = VecTaggerDestroy(&coarsenTag);
    CHKERRQ(ierr);
    ierr = PetscLimiterDestroy(&LimiterType);
    CHKERRQ(ierr);
    ierr = PetscLimiterDestroy(&noneLimiter);
    CHKERRQ(ierr);
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
