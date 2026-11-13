use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, Displacement, MaterialElasticityModel,
    PrescribedStructuralDofs, PrescribedValue, RegionTarget, StructuralConstraint,
    StructuralConstraintKind, StructuralTranslationDof, TranslationDofValue,
};
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};

use super::{estimate_structural_recovery_error, StructuralRecoveryEstimatorOptions};
use crate::{
    contracts::{
        FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS,
        FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY, FEA_FIELD_STRUCTURAL_STRESS,
        FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
    },
    fixtures::{
        fixture_model, replace_fixture_excitations, FixtureExcitation, FixtureExcitationKind,
        FixtureId,
    },
    topology::AnalysisProjection,
    ComputeBackend, LinearStaticSolveOptions,
};

#[path = "qualification_tests/cantilever_refinement.rs"]
mod cantilever_refinement;

#[path = "qualification_tests/cantilever_equilibrium.rs"]
mod cantilever_equilibrium;

#[path = "qualification_tests/cantilever_stiffness.rs"]
mod cantilever_stiffness;

#[path = "qualification_tests/interface.rs"]
mod interface;

#[path = "qualification_tests/elastic_static_path.rs"]
mod elastic_static_path;

// witness: FEA-SHARED-OPERATORS-001/compatible_solver_and_preconditioner_selection @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: The elastic rigidity path (no global factor, AMG-PCG, true-residual acceptance) matches the proving factorization's displacement to 1e-6 on a structured cantilever.
#[test]
fn elastic_rigidity_static_path_agrees_with_the_proving_factorization() {
    elastic_static_path::qualify();
}

// witness: FEA-SHARED-OPERATORS-001/compatible_solver_and_preconditioner_selection @ shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial; boundary: The planner selects the elastic rigidity path exactly when the effective ceiling, min(authored, machine), lies below the proving factorization's plan; recorded and direct requests keep their path.
#[test]
fn effective_ceiling_selects_the_static_solve_path() {
    elastic_static_path::ceiling_selects_the_static_path();
}

// witness: FEA-SHARED-OPERATORS-001/compatible_solver_and_preconditioner_selection @ shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial; boundary: The elastic rigidity fallback is admitted only when its resident memory and scratch fit the effective ceiling; below both plans the planner fails with a resource error naming both paths' requirements and the ceiling.
#[test]
fn static_solve_fallback_is_admitted_against_the_effective_ceiling() {
    elastic_static_path::fallback_is_admitted_against_the_ceiling();
}

#[test]
fn solved_tet10_recovery_estimator_decreases_under_uniform_refinement() {
    cantilever_refinement::qualify();
}

#[path = "qualification_tests/curved_patch.rs"]
mod curved_patch;
#[path = "qualification_tests/heterogeneous.rs"]
mod heterogeneous;
#[path = "qualification_tests/manufactured.rs"]
mod manufactured;
#[path = "qualification_tests/mesh.rs"]
pub(crate) mod mesh;
use mesh::{entity, structured_cantilever};

#[test]
fn manufactured_quadratic_solid_solution_converges_for_tet4_and_is_exact_for_tet10() {
    manufactured::manufactured_quadratic_solid_solution_converges_for_tet4_and_is_exact_for_tet10();
}

#[test]
fn curved_interior_tet10_patch_recovers_affine_stress_and_energy() {
    curved_patch::curved_interior_tet10_patch_recovers_affine_stress_and_energy();
}

#[test]
fn solved_heterogeneous_tet4_stress_matches_an_independent_constitutive_reference() {
    heterogeneous::solved_heterogeneous_tet4_stress_matches_an_independent_constitutive_reference();
}

#[test]
fn solved_heterogeneous_tet10_stress_matches_independent_quadratic_constitutive_reference() {
    heterogeneous::solved_heterogeneous_tet10_stress_matches_independent_quadratic_constitutive_reference();
}
