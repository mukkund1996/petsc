
static char help[] = "Nonlinear, time-dependent PDE in 2d.\n";


/*
   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
   Include "petscts.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
#include <petscdmplex.h>
#include <petscds.h>
#include <petscts.h>
#include <petscblaslapack.h>


#if defined(PETSC_HAVE_CGNS)
#undef I
#include <cgnslib.h>
#endif

/*
   User-defined routines
*/
extern PetscErrorCode FormFunction(TS, PetscReal, Vec, Vec, void *), FormInitialSolution(DM, Vec);

extern PetscErrorCode MyTSMonitor(TS, PetscInt, PetscReal, Vec, void *);

extern PetscErrorCode MySNESMonitor(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);

/* ========================================================================== */
typedef struct {
    DM da;
    PetscBool interpolate;                  /* Generate intermediate mesh elements */
    char filename[PETSC_MAX_PATH_LEN]; /* Mesh filename */
    PetscInt dim;

    PetscErrorCode (**bcFuncs)(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);

    PetscScalar u, v, diffusion, C;
    PetscScalar delta_x, delta_y;
    PetscInt cells[2];
} AppCtx;

//static PetscErrorCode zero(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
//{
//    PetscInt i;
//    for (i = 0; i < dim; ++i) u[i] = 0.0;
//    return 0;
//}

/* ========================================================================== */
static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options) {
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    options->interpolate = PETSC_TRUE;
    options->filename[0] = '\0';
    options->dim = 2;
    options->bcFuncs = NULL;
    options->u = 2.5;
    options->v = 0.0;
    options->cells[0] = 2;
    options->cells[1] = 2;
    options->diffusion = 0.0;
    options->C = 0.01;

    ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");
            CHKERRQ(ierr);
            ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "advection_DMPLEX.c",
                                    options->interpolate, &options->interpolate, NULL);
            CHKERRQ(ierr);
            ierr = PetscOptionsString("-filename", "The mesh file", "advection_DMPLEX.c", options->filename,
                                      options->filename, PETSC_MAX_PATH_LEN, NULL);
            CHKERRQ(ierr);
            ierr = PetscOptionsInt("-dim", "The dimension of problem used for non-file mesh", "advection_DMPLEX.c",
                                   options->dim, &options->dim, NULL);
            CHKERRQ(ierr);
            ierr = PetscOptionsScalar("-u", "The x component of the convective coefficient", "advection_DMPLEX.c",
                                      options->u, &options->u, NULL);
            CHKERRQ(ierr);
            ierr = PetscOptionsScalar("-v", "The y component of the convective coefficient", "advection_DMPLEX.c",
                                      options->v, &options->v, NULL);
            CHKERRQ(ierr);
            ierr = PetscOptionsScalar("-diffus", "The diffusive coefficient", "advection_DMPLEX.c", options->diffusion,
                                      &options->diffusion, NULL);
            CHKERRQ(ierr);
            ierr = PetscOptionsScalar("-c", "The Non-linear coefficient", "advection_DMPLEX.c", options->C, &options->C,
                                      NULL);
            CHKERRQ(ierr);

            ierr = PetscOptionsEnd();
    PetscFunctionReturn(0);
}
/* ========================================================================== */
// Routine for Creating the Mesh
static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm) {
    size_t len;
    PetscErrorCode ierr;
    DMBoundaryType bcs[3];
    bcs[0] = bcs[1] = bcs[2] = DM_BOUNDARY_GHOSTED;

    PetscFunctionBeginUser;
    ierr = PetscStrlen(user->filename, &len);
    CHKERRQ(ierr);
    // If you dont specify a file_name/location, run this routine
    if (!len) {
        DMLabel label;
//        PetscInt id = 1;

        ierr = DMPlexCreateBoxMesh(comm, user->dim, PETSC_FALSE, user->cells, NULL, NULL, bcs, user->interpolate, dm);
        CHKERRQ(ierr);
        /* Mark boundary and set BC */
        ierr = DMCreateLabel(*dm, "boundary");
        CHKERRQ(ierr);
        ierr = DMGetLabel(*dm, "boundary", &label);
        CHKERRQ(ierr);
        ierr = DMPlexMarkBoundaryFaces(*dm, 1, label);
        CHKERRQ(ierr);
        ierr = DMPlexLabelComplete(*dm, label);
        CHKERRQ(ierr);
//        ierr = PetscMalloc1(1, &user->bcFuncs);
//        CHKERRQ(ierr);
//        user->bcFuncs[0] = zero;
//        ierr = DMAddBoundary(*dm, DM_BC_ESSENTIAL, "wall", "boundary", 0, 0, NULL, (void (*)(void)) user->bcFuncs[0], 1, &id, user);CHKERRQ(ierr);
    } else {
        ierr = DMPlexCreateFromFile(comm, user->filename, user->interpolate, dm);
        CHKERRQ(ierr);
    }
    ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");
    CHKERRQ(ierr);
    ierr = DMSetFromOptions(*dm);
    CHKERRQ(ierr);
    ierr = DMViewFromOptions(*dm, NULL, "-dm_view");
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
/* ========================================================================== */
/* Checking the Mesh Structure (Simplex or Tensor) */
static PetscErrorCode CheckMeshTopology(DM dm) {
    PetscInt dim, coneSize, cStart;
    PetscBool isSimplex;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = DMGetDimension(dm, &dim);
    CHKERRQ(ierr);
    // Get starting and ending point (here NULL) for needed height
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);
    CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, cStart, &coneSize);
    CHKERRQ(ierr);
    // Possibly for a triangle for a 2D mesh?
    isSimplex = coneSize == dim + 1 ? PETSC_TRUE : PETSC_FALSE;
    ierr = DMPlexCheckSymmetry(dm);
    CHKERRQ(ierr);
    ierr = DMPlexCheckSkeleton(dm, 0);
    CHKERRQ(ierr);
    ierr = DMPlexCheckFaces(dm, 0);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
/* ========================================================================== */
/* Subroutine to define faces information and corresponding neighbors */
static PetscErrorCode CheckMeshGeometry(DM dm) {
    PetscInt dim, coneSize, cStart, cEnd, c; //cStart, cEnd - cells
    PetscReal *v0, *J, *invJ, detJ;
    PetscInt conesize;
    const PetscInt *cone;
    PetscInt nC;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = DMGetDimension(dm, &dim);
    CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
    CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, cStart, &coneSize);
    CHKERRQ(ierr);
    ierr = PetscMalloc3(dim, &v0, dim * dim, &J, dim * dim, &invJ);
    CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
        // conesize - no. of nodes supporting the cell
        ierr = DMPlexGetConeSize(dm, c, &conesize);
        CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, c, &cone);
        CHKERRQ(ierr);
        /* printf("  element = %4d, cone size for this element = %4d \n", c, conesize); */
        /* for (i = 0; i<conesize;i++) printf("    element[%2d] = %4d\n",i,cone[i]); */

        // Possibly a check for an invalid Jacobian
        ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);
        CHKERRQ(ierr);
        if (detJ <= 0.0)
            SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for cell %D", (double) detJ, c);
    }
    ierr = PetscFree3(v0, J, invJ);
    CHKERRQ(ierr);
    /* ierr = DMPlexGetTransitiveClosure(dm, 1, PETSC_TRUE, &numPoints, &points); CHKERRQ(ierr); */
    /* ierr = DMPlexRestoreTransitiveClosure(dm, 1, PETSC_TRUE, &numPoints, &points); CHKERRQ(ierr); */

    for (c = cStart; c < cEnd; ++c) {
        const PetscInt *faces;
        PetscInt numFaces, f;

        if ((c < cStart) || (c >= cEnd))
            SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Got invalid point %d which is not a cell", c);
        DMPlexGetConeSize(dm, c, &numFaces);
        DMPlexGetCone(dm, c, &faces);
        for (f = 0; f < numFaces; ++f) {
            const PetscInt face = faces[f];
            const PetscInt *neighbors;

            DMPlexGetSupportSize(dm, face, &nC);
            // Check for the boundary faces possibly
            if (nC != 2) continue;
            DMPlexGetSupport(dm, face, &neighbors);
        }
    }

    PetscFunctionReturn(0);
}
/* ========================================================================== */

int main(int argc, char **argv) {
    TS ts;                         /* time integrator */
    SNES snes;
    Vec x, r;                        /* solution, residual vectors */
    PetscErrorCode ierr;
    DM da;
    PetscMPIInt rank;
    PetscViewer viewer;
    PetscViewerAndFormat *vf;
    AppCtx user;                             /* mesh context */

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Initialize program
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = PetscInitialize(&argc, &argv, (char *) 0, help);
    if (ierr) return ierr;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    CHKERRQ(ierr);
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create distributed array (DMDA) to manage parallel grid and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = ProcessOptions(PETSC_COMM_WORLD, &user);
    CHKERRQ(ierr);
    ierr = CreateMesh(PETSC_COMM_WORLD, &user, &da);
    CHKERRQ(ierr);
    ierr = CheckMeshTopology(da);
    CHKERRQ(ierr);
    ierr = CheckMeshGeometry(da);
    CHKERRQ(ierr);

    ierr = DMSetBasicAdjacency(da, PETSC_TRUE, PETSC_FALSE);CHKERRQ(ierr);

    PetscInt size;
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
    DM dmRedist = NULL;
    if (size > 1) {
        ierr = DMPlexDistributeOverlap(da,1,NULL,&dmRedist);CHKERRQ(ierr);
        DMDestroy(&da);
        da = dmRedist;
    }

    DM dmDist;
    ierr = DMPlexDistribute(da, 0, NULL, &dmDist);CHKERRQ(ierr);
    if (dmDist) {
        ierr = DMDestroy(&da);CHKERRQ(ierr);
        da   = dmDist;
    }
    DM dmghost;
    ierr = DMPlexConstructGhostCells(da, NULL, NULL, &dmghost);CHKERRQ(ierr);
    DMDestroy(&da);
    da = dmghost;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Specifying the fields and dof for the formula through PETSc Section
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscInt numFields = 1, numBC, i;
    PetscInt numComp[1];
    PetscInt numDof[numFields * (user.dim + 1)];
    PetscInt bcField[1];
    PetscSection section;
    IS bcPointIS[1];

    /* Create a scalar field u, a vector field v, and a surface vector field w */
    numComp[0] = 1;

    for (i = 0; i < numFields * (user.dim + 1); ++i) numDof[i] = 0;
    // Vertices - 0
    // Faces - 1 or (dim - 1)
    // Cells - 2 or (dim)
    // numDof[field no * (dim + 1) + d]
    // d - mesh dimension, use the above values
    /* Let u be defined on cells and faces */
    numDof[0 * (user.dim + 1)] = 1;
    numDof[0 * (user.dim + 1) + user.dim - 1] = 1;
    numDof[0 * (user.dim + 1) + user.dim] = 1;

    /* Setup boundary conditions */
    numBC = 1;
    /* Prescribe a Dirichlet condition on u on the boundary
         Label "marker" is made by the mesh creation routine */
    bcField[0] = 0;
    ierr = DMGetStratumIS(da, "marker", 1, &bcPointIS[0]);
    CHKERRQ(ierr);
    /* Create a PetscSection with this data layout */
    ierr = DMSetNumFields(da, numFields);
    CHKERRQ(ierr);
    ierr = DMPlexCreateSection(da, NULL, numComp, numDof, numBC, bcField, NULL, bcPointIS, NULL, &section);
    CHKERRQ(ierr);
    ierr = ISDestroy(&bcPointIS[0]);
    CHKERRQ(ierr);
    /* Name the Field variables */
    ierr = PetscSectionSetFieldName(section, 0, "u");
    CHKERRQ(ierr);
    /* Tell the DM to use this data layout */
    ierr = DMSetLocalSection(da, section);
    CHKERRQ(ierr);
    user.da = da;
    // ierr = FindMeshWidths(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);

    PetscInt cStart, cEnd, fStart, fEnd, nStart, nEnd;
    ierr = DMPlexGetInteriorCellStratum(da, &cStart, &cEnd);
    CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(da, 1, &fStart, &fEnd);
    CHKERRQ(ierr);
    DMPlexGetGhostCellStratum(da, &nStart, &nEnd);
    printf(" cstart = %4d \n", cStart);
    printf(" cend   = %4d \n", cEnd);
    printf(" fstart = %4d \n", fStart);
    printf(" fend   = %4d \n", fEnd);
    printf(" boundary start = %4d \n", nStart);
    printf(" boundary start   = %4d \n", nEnd);
    /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Extract global vectors from DMDA; then duplicate for remaining
       vectors that are the same types
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /* Create a Vec with this layout and view it */
    ierr = DMGetGlobalVector(da, &x);
    CHKERRQ(ierr);
    ierr = VecDuplicate(x, &r);
    CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create timestepping solver context
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = TSCreate(PETSC_COMM_WORLD, &ts);
    CHKERRQ(ierr);
    ierr = TSSetProblemType(ts, TS_NONLINEAR);
    CHKERRQ(ierr);
    ierr = TSSetRHSFunction(ts, NULL, FormFunction, &user);
    CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create matrix data structure; set Jacobian evaluation routine

       Set Jacobian matrix data structure and default Jacobian evaluation
       routine. User can override with:
       -snes_mf : matrix-free Newton-Krylov method with no preconditioning
                  (unless user explicitly sets preconditioner)
       -snes_mf_operator : form preconditioning matrix as set by the user,
                           but use matrix-free approx for Jacobian-vector
                           products within Newton-Krylov method

       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ierr = TSSetMaxTime(ts, 1.0);
    CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);
    CHKERRQ(ierr);
    ierr = TSMonitorSet(ts, MyTSMonitor, PETSC_VIEWER_STDOUT_WORLD, NULL);
    CHKERRQ(ierr);
    ierr = TSSetDM(ts, da);
    CHKERRQ(ierr);
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Customize nonlinear solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = TSSetType(ts, TSEULER);
    CHKERRQ(ierr);
    ierr = TSGetSNES(ts, &snes);
    CHKERRQ(ierr);
    ierr = PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_DEFAULT, &vf);
    CHKERRQ(ierr);
    ierr = SNESMonitorSet(snes, (PetscErrorCode (*)(SNES, PetscInt, PetscReal, void *)) MySNESMonitor, vf,
                          (PetscErrorCode (*)(void **)) PetscViewerAndFormatDestroy);
    CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Set initial conditions
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = FormInitialSolution(da, x);
    CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts, .0001);
    CHKERRQ(ierr);
    ierr = TSSetSolution(ts, x);
    CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Set runtime options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = TSSetFromOptions(ts);
    CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Solve nonlinear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = TSSolve(ts, x);
    CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      View the Solution
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD, NULL, NULL, 0, 0, PETSC_DECIDE, PETSC_DECIDE, &viewer);
    CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) viewer, "Line graph Plot");
    CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_DRAW_LG);
    CHKERRQ(ierr);
    /*
       View the vector
    */
//    ierr = VecView(x, viewer);
//    CHKERRQ(ierr);

    /*
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
    */
    ierr = PetscViewerPopFormat(viewer);
    CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);
    CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = DMRestoreGlobalVector(da, &x);
    CHKERRQ(ierr);
    ierr = VecDestroy(&r);
    CHKERRQ(ierr);
    ierr = TSDestroy(&ts);
    CHKERRQ(ierr);
    ierr = DMDestroy(&da);
    CHKERRQ(ierr);

    ierr = PetscFinalize();
    return ierr;
}
/* ========================================================================== */
/*
   FormFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  ts - the TS context
.  X - input vector
.  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector
 */
PetscErrorCode FormFunction(TS ts, PetscReal ftime, Vec X, Vec F, void *ctx) {
    AppCtx *user = (AppCtx *) ctx;
    DM da = (DM) user->da;
    PetscErrorCode ierr;
    PetscScalar *x, *f;
    Vec localX;
    PetscInt fStart, fEnd, nF;
    PetscInt cell, cStart, cEnd, nC;


    PetscFunctionBeginUser;
    ierr = DMGetLocalVector(da, &localX);
    CHKERRQ(ierr);

    /*
       Scatter ghost points to local vector,using the 2-step process
          DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
       By placing code between these two statements, computations can be
       done while messages are in transition.
    */
    ierr = DMGlobalToLocalBegin(da, X, INSERT_VALUES, localX);
    CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, X, INSERT_VALUES, localX);
    CHKERRQ(ierr);

    /*
       Get pointers to vector data
    */
    ierr = VecGetArray(localX, &x);
    CHKERRQ(ierr);
    ierr = VecGetArray(F, &f);
    CHKERRQ(ierr);

    /* ---------------Obtaining local cell and face ownership------------------ */
    ierr = DMPlexGetInteriorCellStratum(da, &cStart, &cEnd);
    CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(da, 1, &fStart, &fEnd);
    CHKERRQ(ierr);
    /* ------------------------------------------------------------------------ */

    DM dmFace, gradDM;      /* DMPLEX for face geometry */
    PetscFV fvm;                /* specify type of FVM discretization */
    Vec cellGeom, faceGeom, grad, locGrad; /* vector of structs related to cell/face geometry*/
    const PetscScalar *fgeom, *pointGrads;             /* values stored in the vector facegeom */
    PetscFVFaceGeom *fgA;               /* struct with face geometry information */
    PetscLimiter    limiter = NULL, LimiterType = NULL;
    PetscDS prob;

    /*....Create FV object....*/
    ierr = PetscFVCreate(PETSC_COMM_WORLD, &fvm);
    CHKERRQ(ierr);

    /*....Set FV type: required for subsequent function call....*/
    ierr = PetscFVSetType(fvm, PETSCFVUPWIND);
    CHKERRQ(ierr);
    ierr = PetscFVSetNumComponents(fvm, 1);
    CHKERRQ(ierr);
    ierr = PetscFVSetSpatialDimension(fvm, user->dim);
    CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fvm,"");
    CHKERRQ(ierr);
    ierr = PetscFVSetUp(fvm);
    CHKERRQ(ierr);

    ierr = DMSetField(da, 0, NULL, (PetscObject) fvm);
    CHKERRQ(ierr);
    ierr = DMCreateDS(da);
    CHKERRQ(ierr);
//    ierr = DMGetDS(da, &prob);
//    CHKERRQ(ierr);
//    ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fvm);
//    CHKERRQ(ierr);

    /*....Setting the FV limiter....*/
    ierr = PetscFVGetLimiter(fvm, &limiter);
    CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject) limiter);
    CHKERRQ(ierr);
    ierr = PetscLimiterCreate(PetscObjectComm((PetscObject) fvm), &LimiterType);
    CHKERRQ(ierr);
    ierr = PetscLimiterSetType(LimiterType, PETSCLIMITERNONE);
    CHKERRQ(ierr);
    ierr = PetscFVSetLimiter(fvm, LimiterType);
    CHKERRQ(ierr);

    ierr = PetscFVSetComputeGradients(fvm, PETSC_TRUE);
    CHKERRQ(ierr);
    /*....Retrieve precomputed cell geometry....*/
    ierr = DMPlexGetDataFVM(da, fvm, &cellGeom, &faceGeom, &gradDM);
    CHKERRQ(ierr);
    ierr = DMPlexInsertBoundaryValues(da, PETSC_TRUE, localX, 0.0, faceGeom, cellGeom, NULL);
    CHKERRQ(ierr);

    /*....Getting the gradient vector....*/
    ierr = DMCreateGlobalVector(gradDM, &grad);
    CHKERRQ(ierr);
//    ierr = DMPlexComputeGradientFVM(da, fvm, faceGeom, cellGeom, &gradDM);
//    CHKERRQ(ierr);

    /*....Reconstructing gradient of the local vector....*/
    ierr = DMPlexReconstructGradientsFVM(da, localX, grad);
    CHKERRQ(ierr);

//    PetscViewer view;
//    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&view);CHKERRQ(ierr);
//    ierr = PetscViewerSetType(view,PETSCVIEWERASCII);CHKERRQ(ierr);
//    ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,300,300,&view);CHKERRQ(ierr);
//    DMView(gradDM, view);

    ierr = DMCreateLocalVector(gradDM, &locGrad);
    CHKERRQ(ierr);

    ierr = DMGlobalToLocalBegin(gradDM, grad, INSERT_VALUES, locGrad);
    CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(gradDM, grad, INSERT_VALUES, locGrad);
    CHKERRQ(ierr);

    ierr = VecGetArrayRead(locGrad,&pointGrads);
    CHKERRQ(ierr);

    /*....get DM defining the data layout of the faceGeom vector....*/
    // Setting the vector size/dimension using the DM
    ierr = VecGetDM(faceGeom, &dmFace);
    CHKERRQ(ierr);
    /*....Get read-only access to array from vector....*/
    /*....observe GetArray and RestoreArray to perform memory (de)allocation....*/
    ierr = VecGetArrayRead(faceGeom, &fgeom);
    CHKERRQ(ierr);

    /*
       Spanning through all the cells and an inner loop through the
       faces. Find the face neighbors and pick the upwinded cell value for flux.
    */
    const PetscInt *cellcone, *cellsupport;
    PetscScalar flux_east, flux_west, flux_north, flux_south, flux_centre;
    PetscScalar centroid_x[2], centroid_y[2], boundary = 0.0;
    PetscScalar boundary_left = 0.0;
    PetscScalar u_plus, u_minus, v_plus, v_minus;
    PetscScalar delta_x, delta_y;

    u_plus = PetscMax(user->u, 0);
    u_minus = PetscMin(user->u, 0);
    v_plus = PetscMax(user->v, 0);
    v_minus = PetscMin(user->v, 0);

    for (cell = cStart; cell < cEnd; cell++) {
        PetscScalar *pointGrad;
        /* Obtaining the faces of the cell */
        ierr = DMPlexGetConeSize(da, cell, &nF);
        CHKERRQ(ierr);
        ierr = DMPlexGetCone(da, cell, &cellcone);
        CHKERRQ(ierr);
        ierr = DMPlexPointLocalRead(gradDM,cell,pointGrads,&pointGrad);
        CHKERRQ(ierr);
        printf("Gradient x = %f and Gradient y = %f \n", pointGrad[0], pointGrad[1]);
        // south
        ierr = DMPlexPointLocalRead(dmFace, cellcone[0], fgeom, &fgA);
        CHKERRQ(ierr);
        centroid_y[0] = fgA->centroid[1];
//        printf("normal of face - %f, %f \n", fgA->normal[0], fgA->normal[1]);
        // North
        ierr = DMPlexPointLocalRead(dmFace, cellcone[2], fgeom, &fgA);
        CHKERRQ(ierr);
        centroid_y[1] = fgA->centroid[1];
        // west
        ierr = DMPlexPointLocalRead(dmFace, cellcone[3], fgeom, &fgA);
        CHKERRQ(ierr);
        centroid_x[0] = fgA->centroid[0];
        // east
        ierr = DMPlexPointLocalRead(dmFace, cellcone[1], fgeom, &fgA);
        CHKERRQ(ierr);
        centroid_x[1] = fgA->centroid[0];
//        printf("gradient of face - %f, %f \n", fgA->grad[0][0], fgA->grad[0][1]);

        delta_x = centroid_x[1] - centroid_x[0];
        delta_y = centroid_y[1] - centroid_y[0];

        /* Getting the neighbors of each face */

        // Going through the faces by the order (cellcone)
        // cellcone[0] - south
        DMPlexGetSupportSize(da, cellcone[0], &nC);
        DMPlexGetSupport(da, cellcone[0], &cellsupport);
        if (nC == 2) flux_south = (x[cellsupport[0]] * (-v_plus - user->diffusion * delta_x)) / delta_y;
        else flux_south = (boundary * (-v_plus - user->diffusion * delta_x)) / delta_y;

        // cellcone[1] - east
        DMPlexGetSupportSize(da, cellcone[1], &nC);
        DMPlexGetSupport(da, cellcone[1], &cellsupport);
        if (nC == 2) flux_east = (x[cellsupport[1]] * (u_minus - user->diffusion * delta_y)) / delta_x;
        else flux_east = (boundary * (u_minus - user->diffusion * delta_y)) / delta_x;

        // cellcone[2] - north
        DMPlexGetSupportSize(da, cellcone[2], &nC);
        DMPlexGetSupport(da, cellcone[2], &cellsupport);
        if (nC == 2) flux_north = (x[cellsupport[1]] * (v_minus - user->diffusion * delta_x)) / delta_y;
        else flux_north = (boundary * (v_minus - user->diffusion * delta_x)) / delta_y;

        // cellcone[3] - west
        DMPlexGetSupportSize(da, cellcone[3], &nC);
        DMPlexGetSupport(da, cellcone[3], &cellsupport);
        if (nC == 2) flux_west = (x[cellsupport[0]] * (-u_plus - user->diffusion * delta_y)) / delta_x;
        else flux_west = (boundary_left * (-u_plus - user->diffusion * delta_y)) / delta_x;
        // else flux_west = 2.0;

        flux_centre = x[cell] * ((u_plus - u_minus + 2 * user->diffusion * delta_y) / delta_x +
                                 (v_plus - v_minus + 2 * user->diffusion * delta_x) / delta_y) -
                      PetscExpScalar(x[cell]) * delta_x * delta_y;

        // Need to multiply with delta x and delta y
        f[cell] = -(flux_centre + flux_east + flux_west + flux_north + flux_south);


    }
//    printf("delta x - %f, delta y - %f \n", delta_x, delta_y);
    // printf("delta x = %f, delta_y = %f \n", delta_x, delta_y);
    /*
       Restore vectors
    */
    VecRestoreArrayRead(locGrad,&pointGrads);
    VecDestroy(&locGrad);
    VecDestroy(&grad);
    ierr = VecRestoreArray(localX, &x);
    CHKERRQ(ierr);
    ierr = VecRestoreArray(F, &f);
    CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(da, &localX);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

/* ========================================================================== */
PetscErrorCode FormInitialSolution(DM da, Vec U) {
    PetscErrorCode ierr;
    PetscScalar *u;

    /*
      No need for a local vector because there is exchange of information
      across the processors. Unlike for FormFunction which depends on the neighbours
    */

    PetscFunctionBeginUser;
    /*
       Get pointers to vector data
    */
    ierr = VecGetArray(U, &u);
    CHKERRQ(ierr);

    /*
       Get local grid boundaries
    */
    PetscInt cell, cStart, cEnd;
    PetscReal cellvol, centroid[3], normal[3];
    ierr = DMPlexGetInteriorCellStratum(da, &cStart, &cEnd);
    CHKERRQ(ierr);

    /*
       Compute function over the locally owned part of the grid
    */
    // Assigning the values at the cell centers based on x and y directions
    for (cell = cStart; cell < cEnd; cell++) {
        DMPlexComputeCellGeometryFVM(da, cell, &cellvol, centroid, normal);
        if (centroid[0] >= 0.2 && centroid[0] <= 0.4) {
            u[cell] = 2.0;
        }
        else if(centroid[0] > 0.4 && centroid[0] <= 0.6) {
            u[cell] = 4.0;
        }
        else
            u[cell] = 1.0;
    } /*..end for loop over cells..*/

    for (cell = cStart; cell < cEnd; cell++) {
        printf("cell no - %d = %f \n", cell, u[cell]);
    }

    /*
       Restore vectors
    */
    ierr = VecRestoreArray(U, &u);
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

/*
   MySNESMonitor - illustrate how to set user-defined monitoring routine for SNES.
   Input Parameters:
     snes - the SNES context
     its - iteration number
     fnorm - 2-norm function value (may be estimated)
     ctx - optional user-defined context for private data for the
         monitor routine, as set by SNESMonitorSet()
 */
PetscErrorCode MySNESMonitor(SNES snes, PetscInt its, PetscReal fnorm, PetscViewerAndFormat *vf) {
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = SNESMonitorDefaultShort(snes, its, fnorm, vf);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/*TEST

    test:
      args: -ts_max_steps 5

    test:
      suffix: 2
      args: -ts_max_steps 5  -snes_mf_operator

    test:
      suffix: 3
      args: -ts_max_steps 5  -snes_mf -pc_type none

TEST*/
