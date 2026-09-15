use runmat_analysis_core::{validate_model, AnalysisProcedure, StateKind};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    assembly::{assemble_structural_system, StructuralFormulationRequirement},
    contracts::{FeaRunError, SolverMethod, SolverPreconditioner},
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    execution::{
        CanonicalSemanticState, StructuralBucklingMode, StructuralBucklingSolverResult,
        StructuralBucklingSolverResultInput, StructuralBucklingWorkload,
    },
    solve::{backend::kind::LinearAlgebraBackendKind, buckling::solve_eigenvalue_buckling},
};

/// Executes a path-free eigenvalue-buckling workload from one immutable preload state.
///
/// The consumer always reassembles its own elastic operator and validates it against the
/// producer-bound linearization before solving. The state therefore transfers physical preload
/// authority without transferring an unchecked current-model operator.
pub fn execute_structural_buckling_workload(
    workload: &StructuralBucklingWorkload,
    solver_mesh: &SolverMeshArtifact,
    prestress_state: &CanonicalSemanticState,
) -> Result<StructuralBucklingSolverResult, FeaRunError> {
    workload
        .validate()
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    prestress_state
        .validate()
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    solver_mesh
        .validate_canonical()
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let topology_digest = Digest::from_bytes(*solver_mesh.canonical_digest.bytes());
    if solver_mesh.geometry.geometry_revision != u64::from(workload.model().geometry_revision)
        || prestress_state.kind != StateKind::GeometricStiffness
        || prestress_state.identity != workload.prestress_transition().source_state
        || prestress_state.topology_digest != topology_digest
        || workload.prestress_transition().target_topology_digest != topology_digest
    {
        return Err(FeaRunError::InvalidModel(
            "buckling mesh, workload transition, and geometric-stiffness state do not identify one topology and source state".into(),
        ));
    }
    let linearization = prestress_state
        .payload
        .structural_linearization()
        .ok_or_else(|| {
            FeaRunError::InvalidModel(
                "buckling prestress state does not carry structural linearization data".into(),
            )
        })?;
    validate_model(workload.model())
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let step = super::admission::require_structural(
        workload.model(),
        AnalysisProcedure::EigenvalueBuckling,
        workload.formulation_requirement(),
    )?;
    let excitations =
        realize_step_excitations(workload.model(), step, ExcitationSamplePoint::Static)
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let summary = assemble_structural_system(
        workload.model(),
        &excitations,
        Some(solver_mesh.clone()),
        StructuralFormulationRequirement::ExperimentalSyntheticOperator,
    )
    .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let run =
        super::admission::admit_structural(&summary, step, workload.formulation_requirement())?;
    super::validate_rotational_dof_targets(workload.model(), &summary)?;
    let solved = solve_eigenvalue_buckling(
        &summary,
        linearization,
        workload.solver().requested_mode_count(),
        workload.solver().relative_tolerance(),
        workload.solver().maximum_iterations(),
    )
    .map_err(|error| FeaRunError::Solve(error.to_string()))?;
    if !solved.converged
        || solved.modes.len() != workload.solver().requested_mode_count()
        || solved
            .modes
            .iter()
            .any(|mode| mode.residual_norm > workload.solver().relative_tolerance())
    {
        return Err(FeaRunError::Solve(
            "buckling eigensolver did not qualify every requested positive load factor".into(),
        ));
    }
    StructuralBucklingSolverResult::new(
        workload,
        StructuralBucklingSolverResultInput {
            topology_digest,
            source_state: prestress_state.identity,
            linearization: linearization.identity(),
            run,
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            solver_method: SolverMethod::MatrixFreeBucklingSubspaceIteration,
            preconditioner: SolverPreconditioner::from_spd(solved.preconditioner),
            iteration_count: solved.iteration_count,
            modes: solved
                .modes
                .into_iter()
                .map(|mode| StructuralBucklingMode {
                    load_factor: mode.load_factor,
                    residual_norm: mode.residual_norm,
                    dof_shape: mode.dof_shape,
                })
                .collect(),
        },
    )
    .map_err(|error| FeaRunError::Solve(error.to_string()))
}
