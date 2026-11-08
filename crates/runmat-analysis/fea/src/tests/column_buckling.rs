//! Fixed-free solid column under axial compression against Euler's critical load.
//!
//! Reference: Timoshenko & Gere, *Theory of Elastic Stability*, 2nd ed., §2.1: a column fixed at
//! its base and free at its loaded top buckles at `P_cr = π² E I / (4 L²)`. The column here is the
//! /physics/structural preload example: 1 m steel, 100 × 50 mm section, 250 kN, weak axis
//! `I = 100 · 50³ / 12 mm⁴`, so `P_cr = 514.0 kN` and the load factor is `2.056`. Shear
//! flexibility lowers it by `P_cr / (κ G A) = 0.16 %` (Timoshenko & Gere §2.17).
use runmat_analysis_core::{
    AnalysisConstraint, AnalysisModel, CoordinateFrame, ExcitationAction, ExcitationTarget,
    MaterialElasticityModel, MaterialMechanicalModel, ProcedureStep, PropertyRegionTarget,
    RegionTarget, StructuralConstraint, StructuralConstraintKind,
};
use runmat_meshing_core::{
    fixtures::{elevate_planar_tet4_fixture, fixture_boundary_selection, layered_box_mesh},
    SolverMeshArtifact,
};

use crate::{
    fixtures::{
        fixture_model, replace_fixture_excitations, FixtureExcitation, FixtureExcitationKind,
        FixtureId,
    },
    ComputeBackend, LinearStaticSolveOptions,
};

const LENGTH_M: f64 = 1.0;
const WIDTH_M: f64 = 0.1;
const THICKNESS_M: f64 = 0.05;
const YOUNGS_MODULUS_PA: f64 = 200.0e9;
const POISSON_RATIO: f64 = 0.3;
const AXIAL_LOAD_N: f64 = 250.0e3;

fn bound_target(mesh: &SolverMeshArtifact, name: &str) -> RegionTarget {
    let mut target = RegionTarget::boundary(fixture_boundary_selection(name));
    let entities = mesh
        .topology
        .boundary_faces
        .iter()
        .flat_map(|face| face.provenance.iter())
        .filter(|entity| target.matches_persistent_entity(entity))
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    target.bind_resolved_entities(entities).unwrap();
    target
}

fn column_model(mesh: &SolverMeshArtifact) -> AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = None;
    model.thermo_mechanical = None;
    model.electro_thermal = None;
    model.electromagnetic = None;
    model.constraints.clear();
    model.boundaries.clear();
    model.initial_conditions.clear();
    model.connectors.clear();
    model.response_spectra.clear();
    model.random_excitations.clear();
    model.frame = CoordinateFrame::Global;
    model.sequence.steps = vec![ProcedureStep::structural_static("preload")];
    let mut steel = model.materials[0].clone();
    steel.material_id = "steel".into();
    steel.name = "steel".into();
    steel.mass_density_kg_per_m3 = Some(runmat_types::physical::Quantity::new(7850.0));
    steel.elasticity = Some(MaterialElasticityModel::IsotropicLinear(
        MaterialMechanicalModel {
            youngs_modulus_pa: runmat_types::physical::Quantity::new(YOUNGS_MODULUS_PA),
            poisson_ratio: runmat_types::physical::Quantity::new(POISSON_RATIO),
        },
    ));
    model.materials = vec![steel];
    let mut assignment = model.material_assignments[0].clone();
    assignment.target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("column"));
    assignment.expected_material_id = "steel".into();
    assignment.assigned_material_id = "steel".into();
    model.material_assignments = vec![assignment];
    model
        .constraints
        .push(AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "fixed_base".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Fixed {
                target: bound_target(mesh, "column_x_min"),
            },
        }));
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "axial".into(),
            region_id: "column_x_max".into(),
            kind: FixtureExcitationKind::Traction([
                -AXIAL_LOAD_N / (WIDTH_M * THICKNESS_M),
                0.0,
                0.0,
            ]),
        }],
    );
    let top = bound_target(mesh, "column_x_max");
    for excitation in &mut model.excitations {
        if let ExcitationAction::Traction(action) = excitation {
            action.target = ExcitationTarget::Region(top.clone());
        }
    }
    model
}

// witness: FEA-DYNAMICS-PRESTRESS-001/references @ structural_stability/tet10/numerical_kernel;
//   coverage: partial;
//   check: euler-column; boundary: A fixed-free Tet10 solid column (1 m, 100 x 50 mm, 250 kN) converges its buckling
//   eigenpairs to the authored relative tolerance and matches Euler's fixed-free load within 1 %;
//   frames, plates and mesh-convergence studies are not covered.
#[test]
fn fixed_free_column_buckles_at_the_euler_load_with_converged_eigenpairs() {
    let mesh = elevate_planar_tet4_fixture(layered_box_mesh(
        [40, 4, 2],
        [LENGTH_M, WIDTH_M, THICKNESS_M],
        &[(WIDTH_M + 1.0, "column")],
    ));
    let model = column_model(&mesh);
    let plan = crate::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
        &model,
        Some(&mesh),
    )
    .unwrap();
    let topology_digest = plan.topology.discretization_digest();
    let preload = crate::run_linear_static_with_prepared_constraint_plan_and_linearization(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh.clone()),
            require_solver_mesh_for_solid: true,
            ..LinearStaticSolveOptions::default()
        },
        &plan,
        topology_digest,
        None,
    )
    .unwrap();
    let excitations = crate::excitation::realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        crate::excitation::ExcitationSamplePoint::Static,
    )
    .unwrap();
    let summary = crate::assembly::assemble_structural_system_with_constraint_plan(
        &model,
        &excitations,
        Some(mesh.clone()),
        crate::assembly::StructuralFormulationRequirement::SyntheticOperator,
        &plan,
    )
    .unwrap();
    assert!(
        summary.dof_count > 400,
        "the column must exercise the Lanczos path"
    );
    // The study planner's defaults: the baseline algebraic target and 256 restarts.
    let relative_tolerance = 1.0e-8;
    let solved = crate::solve::buckling::solve_eigenvalue_buckling(
        &summary,
        &preload.linearization,
        4,
        relative_tolerance,
        256,
    )
    .unwrap();
    let residuals = solved
        .modes
        .iter()
        .map(|mode| mode.residual_norm)
        .collect::<Vec<_>>();
    assert!(
        solved.converged,
        "iterations={} residuals={residuals:?}",
        solved.iteration_count
    );
    assert_eq!(solved.modes.len(), 4);
    assert!(residuals.iter().all(|&r| r <= relative_tolerance));

    let second_moment = WIDTH_M * THICKNESS_M.powi(3) / 12.0;
    let euler_load = std::f64::consts::PI.powi(2) * YOUNGS_MODULUS_PA * second_moment
        / (4.0 * LENGTH_M * LENGTH_M);
    let euler_factor = euler_load / AXIAL_LOAD_N;
    let first = solved.modes[0].load_factor;
    eprintln!(
        "column buckling: load factors {:?}, Euler fixed-free {euler_factor:.5} (P_cr {:.1} kN), first/Euler {:.5}, residuals {residuals:?}, iterations {}",
        solved.modes.iter().map(|mode| mode.load_factor).collect::<Vec<_>>(),
        euler_load / 1.0e3,
        first / euler_factor,
        solved.iteration_count
    );
    assert!(
        (first / euler_factor - 1.0).abs() <= 0.01,
        "first load factor {first} vs Euler {euler_factor}"
    );
}
