
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
    #include <petscts.h>
    #include <petscblaslapack.h>


    #if defined(PETSC_HAVE_CGNS)
    #undef I
    #include <cgnslib.h>
    #endif

    /*
    User-defined routines
    */
    extern PetscErrorCode FormFunction(TS,PetscReal,Vec,Vec,void*),FormInitialSolution(DM,Vec,PetscSection);
    extern PetscErrorCode MyTSMonitor(TS,PetscInt,PetscReal,Vec,void*);
    extern PetscErrorCode MySNESMonitor(SNES,PetscInt,PetscReal,PetscViewerAndFormat*);

    /* ========================================================================== */
    typedef struct {
    DM        da;
    PetscBool interpolate;                  /* Generate intermediate mesh elements */
    char      filename[PETSC_MAX_PATH_LEN]; /* Mesh filename */
    PetscInt  dim;
    PetscErrorCode (**bcFuncs)(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
    PetscScalar delta_x, delta_y;
    PetscInt    cells[1];
    PetscSection sect;
    } AppCtx;
    /* ========================================================================== */
    static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
    {
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    options->interpolate = PETSC_TRUE;
    options->filename[0] = '\0';
    options->dim         = 1;
    options->bcFuncs     = NULL;
    options->cells[0]    = 20;

    ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
    ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "advection_DMPLEX.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-filename", "The mesh file", "advection_DMPLEX.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dim", "The dimension of problem used for non-file mesh", "advection_DMPLEX.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();
    PetscFunctionReturn(0);
    }
    /* ========================================================================== */
    // Routine for Creating the Mesh
    static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
    {
    size_t         len;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = PetscStrlen(user->filename, &len);CHKERRQ(ierr);
    // If you dont specify a file_name/location, run this routine
    if (!len) {
    DMLabel  label;
    // PetscInt id = 1;

    ierr = DMPlexCreateBoxMesh(comm, user->dim, PETSC_FALSE, user->cells, NULL, NULL, NULL, user->interpolate, dm);CHKERRQ(ierr);
    /* Mark boundary and set BC */
    ierr = DMCreateLabel(*dm, "boundary");CHKERRQ(ierr);
    ierr = DMGetLabel(*dm, "boundary", &label);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(*dm, 1, label);CHKERRQ(ierr);
    ierr = DMPlexLabelComplete(*dm, label);CHKERRQ(ierr);
    ierr = PetscMalloc1(1, &user->bcFuncs);CHKERRQ(ierr);
    // user->bcFuncs[0] = zero;
    // ierr = DMAddBoundary(*dm, DM_BC_ESSENTIAL, "wall", "boundary", 0, 0, NULL, (void (*)(void)) user->bcFuncs[0], 1, &id, user);CHKERRQ(ierr);
    } else {
    ierr = DMPlexCreateFromFile(comm, user->filename, user->interpolate, dm);CHKERRQ(ierr);
    }
    ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
    ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
    ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
    PetscFunctionReturn(0);
    }
    /* ========================================================================== */
    /* Checking the Mesh Structure (Simplex or Tensor) */
    static PetscErrorCode CheckMeshTopology(DM dm)
    {
    PetscInt       dim, coneSize, cStart;
    PetscBool      isSimplex;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    // Get starting and ending point (here NULL) for needed height
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, cStart, &coneSize);CHKERRQ(ierr);
    // Possibly for a triangle for a 2D mesh?
    isSimplex = coneSize == dim+1 ? PETSC_TRUE : PETSC_FALSE;
    ierr = DMPlexCheckSymmetry(dm);CHKERRQ(ierr);
    ierr = DMPlexCheckSkeleton(dm, 0);CHKERRQ(ierr);
    ierr = DMPlexCheckFaces(dm, 0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
    }
    /* ========================================================================== */
    /* Subroutine to define faces information and corresponding neighbors */
    static PetscErrorCode CheckMeshGeometry(DM dm)
    {
    PetscInt       dim, coneSize, cStart, cEnd, c; //cStart, cEnd - cells
    PetscReal      *v0, *J, *invJ, detJ;
    PetscInt       conesize;
    const PetscInt *cone;
    PetscInt       nC;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, cStart, &coneSize);CHKERRQ(ierr);
    ierr = PetscMalloc3(dim,&v0,dim*dim,&J,dim*dim,&invJ);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
    // conesize - no. of nodes supporting the cell
    ierr = DMPlexGetConeSize(dm, c, &conesize); CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, c, &cone); CHKERRQ(ierr);
    /* printf("  element = %4d, cone size for this element = %4d \n", c, conesize); */
    /* for (i = 0; i<conesize;i++) printf("    element[%2d] = %4d\n",i,cone[i]); */

    // Possibly a check for an invalid Jacobian
    ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for cell %D", (double)detJ, c);
    }
    ierr = PetscFree3(v0,J,invJ);CHKERRQ(ierr);
    /* ierr = DMPlexGetTransitiveClosure(dm, 1, PETSC_TRUE, &numPoints, &points); CHKERRQ(ierr); */
    /* ierr = DMPlexRestoreTransitiveClosure(dm, 1, PETSC_TRUE, &numPoints, &points); CHKERRQ(ierr); */

    for (c = cStart; c < cEnd; ++c) {
        const PetscInt *faces;
        PetscInt       numFaces, f;

        if ((c < cStart) || (c >= cEnd)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Got invalid point %d which is not a cell", c);
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

    int main(int argc,char **argv)
    {
    TS                   ts;                         /* time integrator */
    SNES                 snes;
    Vec                  x,r;                        /* solution, residual vectors */
    PetscErrorCode       ierr;
    DM                   da;
    PetscMPIInt          rank;
    PetscViewer          viewer;
    PetscViewerAndFormat *vf;
    AppCtx               user;                             /* mesh context */

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
    ierr = CreateMesh(PETSC_COMM_WORLD, &user, &da);CHKERRQ(ierr);
    ierr = CheckMeshTopology(da);CHKERRQ(ierr);
    ierr = CheckMeshGeometry(da);CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Specifying the fields and dof for the formula through PETSc Section
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscInt       numFields = 2, numBC, i;
    PetscInt       numComp[2];
    PetscInt       numDof[numFields*(user.dim+1)];
    PetscInt       bcField[1];
    IS             bcPointIS[1];

    /* Create a scalar field u, a vector field v, and a surface vector field w */
    numComp[0] = 1;
    numComp[1] = 1;

    for (i = 0; i < numFields*(user.dim+1); ++i) numDof[i] = 0;
    /* Definition of field - h on the faces and cells */
//    numDof[0*(user.dim+1)+user.dim-1]   = 1;
    numDof[0*(user.dim+1)+user.dim]     = 1;

    /* Definition of field - uh on the faces and cells */
//    numDof[1*(user.dim+1)+user.dim-1]   = 1;
    numDof[1*(user.dim+1)+user.dim]     = 1;

    /* Setup boundary conditions */
    numBC = 1;
    /* Prescribe a Dirichlet condition on u on the boundary
       Label "marker" is made by the mesh creation routine */
    bcField[0] = 0;
    ierr = DMGetStratumIS(da, "marker", 1, &bcPointIS[0]);CHKERRQ(ierr);
    /* Create a PetscSection with this data layout */
    ierr = DMSetNumFields(da, numFields);CHKERRQ(ierr);
    ierr = DMPlexCreateSection(da, NULL, numComp, numDof, numBC, bcField, NULL, bcPointIS, NULL, &user.sect);CHKERRQ(ierr);
    ierr = ISDestroy(&bcPointIS[0]);CHKERRQ(ierr);

    /* Name the Field variables */
    ierr = PetscSectionSetFieldName(user.sect, 0, "h");CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(user.sect, 0, "uh");CHKERRQ(ierr);

    /* Tell the DM to use this data layout */
    ierr = DMSetLocalSection(da, user.sect);CHKERRQ(ierr);
//    ierr = PetscSectionView(user.sect, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    user.da = da;

    /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /* Create a Vec with this layout and view it */
    ierr = DMGetGlobalVector(da,&x);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&r);CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
    ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
    ierr = TSSetRHSFunction(ts,NULL,FormFunction,&user);CHKERRQ(ierr);

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

    ierr = TSSetMaxTime(ts,1.0);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
    ierr = TSMonitorSet(ts,MyTSMonitor,PETSC_VIEWER_STDOUT_WORLD,NULL);CHKERRQ(ierr);
    ierr = TSSetDM(ts,da);CHKERRQ(ierr);
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = TSSetType(ts,TSEULER);CHKERRQ(ierr);
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_DEFAULT,&vf);CHKERRQ(ierr);
    ierr = SNESMonitorSet(snes,(PetscErrorCode (*)(SNES,PetscInt,PetscReal,void*))MySNESMonitor,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = FormInitialSolution(da,x,user.sect);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,.0001);CHKERRQ(ierr);
    ierr = TSSetSolution(ts,x);CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = TSSolve(ts,x);CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    View the Solution
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
       View the vector
    */
    ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    /*
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
    */
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = DMRestoreGlobalVector(da, &x);CHKERRQ(ierr);
    ierr = VecDestroy(&r);CHKERRQ(ierr);
    ierr = TSDestroy(&ts);CHKERRQ(ierr);
    ierr = DMDestroy(&da);CHKERRQ(ierr);

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
    PetscErrorCode FormFunction(TS ts,PetscReal ftime,Vec X,Vec F,void *ctx)
    {
    AppCtx         *user=(AppCtx*)ctx;
    PetscSection   section=(PetscSection)user->sect;
    DM             da = (DM)user->da;
    PetscErrorCode ierr;
    PetscScalar    *x,*f;
    Vec            localX;
    PetscInt       fStart, fEnd, nF;
    PetscInt       cell, cStart, cEnd, nC;


    PetscFunctionBeginUser;
    ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);


    /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
    */
    ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);

    /*
     Get pointers to vector data
    */
    ierr = VecGetArray(localX, &x);CHKERRQ(ierr);
    ierr = VecGetArray(F, &f);CHKERRQ(ierr);

    /* ---------------Obtaining local cell and face ownership------------------ */
    ierr = DMPlexGetHeightStratum(da, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(da, 1, &fStart, &fEnd);CHKERRQ(ierr);
    /* ------------------------------------------------------------------------ */

    /*
     Spanning through all the cells and an inner loop through the
     faces. Find the face neighbors and pick the upwinded cell value for flux.
    */

    const PetscInt *cellcone, *cellsupport;
    PetscScalar    flux_east[2], flux_west[2];
    PetscInt offset_cell[2], offset[2], field;
    PetscScalar u, h, g = 9.81;
    PetscScalar delta_x, cellno=5;
    PetscScalar boundary_h = 0.0, boundary_uh = 0.0;

    for (cell = cStart; cell < cEnd; cell++) {
        /* Obtaining the faces of the cell */
        DMPlexGetConeSize(da, cell, &nF);
        DMPlexGetCone(da, cell, &cellcone);

        delta_x = 1 / (cellno - 1);

        /* Obtaining the offset of the cell component of the fields */
        for(field = 0; field < 2; field++) {
            PetscSectionGetFieldOffset(section, cell, field, &offset_cell[field]);
        }

        /* Getting the neighbors of each face */
        // Going through the faces by the order (cellcone)
        // cellcone[1] - east
        DMPlexGetSupportSize(da, cellcone[1], &nC);
        DMPlexGetSupport(da, cellcone[1], &cellsupport);
        for(field = 0; field < 2; field++)
            PetscSectionGetFieldOffset(section, cellsupport[1], field, &offset[field]);
        if ((x[offset_cell[1]]/x[offset_cell[0]]) < 0) {
            h = x[offset[0]];
            u = (h == 0) ? 0 : (x[offset[1]] / h);
        }
        else{
            h = x[offset_cell[0]];
            u = (h == 0) ? 0 : (x[offset_cell[1]] / h);
        }
        if (nC == 2) {
            flux_east[0] = (u * h) / delta_x;
            flux_east[1] = (PetscSqr(u) * h + 0.5 * g * PetscSqr(h)) / delta_x;
        } else {
            flux_east[0] = (boundary_uh) / delta_x;
            if (boundary_h != 0)
                flux_east[1] = (boundary_uh * (boundary_uh / boundary_h) + 0.5 * g * PetscSqr(boundary_h)) / delta_x;
            else flux_east[1] = 0.0;
        }

        // cellcone[3] - west
        DMPlexGetSupportSize(da, cellcone[0], &nC);
        DMPlexGetSupport(da, cellcone[0], &cellsupport);
        for(field = 0; field < 2; field++)
            PetscSectionGetFieldOffset(section, cellsupport[0], field, &offset[field]);
        if ((x[offset_cell[1]]/x[offset_cell[0]]) < 0) {
            h = x[offset_cell[0]];
            u = (h == 0) ? 0 : (x[offset_cell[1]] / h);
        }
        else{
            h = x[offset[0]];
            u = (h == 0) ? 0 : (x[offset[1]] / h);
        }
        if (nC == 2) {
            flux_west[0] = -(u * h) / delta_x;
            flux_west[1] = -(PetscSqr(u) * h + 0.5 * g * PetscSqr(h)) / delta_x;
        } else {
            flux_west[0] = -(boundary_uh) / delta_x;
            if (boundary_h != 0)
            flux_west[1] = -(boundary_uh * (boundary_uh / boundary_h) + 0.5 * g * PetscSqr(boundary_h)) / delta_x;
            else flux_west[1] = 0.0;
        }

//        printf("cell no - %d | flux east - %f | flux_west - %f \n", cell, flux_east[0], flux_west[0]);
//        printf("cell no - %d | cellcone[0] - %d | cellcone[1] - %d \n", cell, cellcone[0], cellcone[1]);

        f[offset_cell[0]] = -(flux_east[0] + flux_west[0]);
        f[offset_cell[1]] = -(flux_east[1] + flux_west[1]);

    }

    // printf("delta x = %f, delta_y = %f \n", delta_x, delta_y);
    /*
     Restore vectors
    */
    ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);
    ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);

    PetscFunctionReturn(0);
    }

    /* ========================================================================== */
    PetscErrorCode FormInitialSolution(DM da,Vec U, PetscSection section)
    {
    PetscErrorCode ierr;
    PetscScalar    *u;

    /*
    No need for a local vector because there is exchange of information
    across the processors. Unlike for FormFunction which depends on the neighbours
    */

    PetscFunctionBeginUser;
    /*
     Get pointers to vector data
    */
    ierr = VecGetArray(U, &u);CHKERRQ(ierr);

    /*
     Get local grid boundaries
    */
    PetscInt cell, cStart, cEnd;
    ierr = DMPlexGetHeightStratum(da, 0, &cStart, &cEnd);CHKERRQ(ierr);

    /*
     Compute function over the locally owned part of the grid
    */

    PetscInt field, offset_cell[2];
    double r, x, delta_x=1/20, c = -30.0;;

    // Assigning the values at the cell centers based on x and y directions
    for (cell = cStart; cell < cEnd; cell++) {
        for(field = 0; field < 2; field++)
            PetscSectionGetFieldOffset(section, cell, field, &offset_cell[field]);

        x = (double)cell * delta_x;
        r = x - 0.5;
        printf("r - %f \n", r);
//        u[offset_cell[0]] = (double)cell * 0.01;
//        u[offset_cell[1]] = 0.0;

        if (x > 0.125) {
            u[offset_cell[0]] = PetscExpScalar(c * x * x * x);
            u[offset_cell[1]] = 0.0;
        }
        else{
            u[offset_cell[0]] = 0.0;
            u[offset_cell[1]] = 0.0;
        }
    }

    /*
     Restore vectors
    */
    ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);

    PetscFunctionReturn(0);
    }

    PetscErrorCode MyTSMonitor(TS ts,PetscInt step,PetscReal ptime,Vec v,void *ctx)
    {
    PetscErrorCode ierr;
    PetscReal      norm;
    MPI_Comm       comm;

    PetscFunctionBeginUser;
    if (step < 0) PetscFunctionReturn(0); /* step of -1 indicates an interpolated solution */
    ierr = VecNorm(v,NORM_2,&norm);CHKERRQ(ierr);
    ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"timestep %D time %g norm %g\n",step,(double)ptime,(double)norm);CHKERRQ(ierr);
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
    PetscErrorCode MySNESMonitor(SNES snes,PetscInt its,PetscReal fnorm,PetscViewerAndFormat *vf)
    {
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = SNESMonitorDefaultShort(snes,its,fnorm,vf);CHKERRQ(ierr);
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
