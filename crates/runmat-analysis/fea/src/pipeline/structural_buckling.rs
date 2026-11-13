use runmat_analysis_core::{validate_model, AnalysisField, AnalysisProcedure, StateKind};
use runmat_analysis_core::{AnalysisFieldLayout, FieldLocation, ObservationQuantity};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    assembly::{assemble_structural_perturbation_system, StructuralFormulationRequirement},
    contracts::{
        fea_buckling_mode_shape_field_id, fea_buckling_rotation_shape_field_id, ExecutionBackend,
        FeaBucklingRunResult, FeaRunError, FeaRunResult, ModalRecoveredMode, SolverMethod,
        SolverPreconditioner, FEA_FIELD_BUCKLING_LOAD_FACTOR, FEA_FIELD_BUCKLING_RESIDUAL_NORM,
    },
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    execution::{
        CanonicalSemanticState, StructuralBucklingMode, StructuralBucklingNumericalReceipt,
        StructuralBucklingSolverResult, StructuralBucklingSolverResultInput,
        StructuralBucklingWorkload,
    },
    pipeline::structural_nodal::{nodal_vector_shape, project_structural_nodal_snapshots},
    post::fields::recover_deformation_result_fields,
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
    )?;
    let excitations =
        realize_step_excitations(workload.model(), step, ExcitationSamplePoint::Static)
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let summary = assemble_structural_perturbation_system(
        workload.model(),
        &excitations,
        Some(solver_mesh.clone()),
        StructuralFormulationRequirement::SyntheticOperator,
    )
    .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let run = super::admission::admit_structural(&summary, step)?;
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
        let maximum_residual = solved
            .modes
            .iter()
            .map(|mode| mode.residual_norm)
            .fold(0.0_f64, f64::max);
        return Err(FeaRunError::Solve(
            format!(
                "buckling eigensolver retained {}/{} positive modes after {} iterations (converged={}, maximum_residual={maximum_residual})",
                solved.modes.len(),
                workload.solver().requested_mode_count(),
                solved.iteration_count,
                solved.converged,
            ),
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
            solver_method: SolverMethod::BucklingBlockLanczos,
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

/// Reconstructs result-facing buckling fields from a portable solve without repeating it.
pub fn recover_completed_structural_buckling_run(
    workload: &StructuralBucklingWorkload,
    solver_mesh: &SolverMeshArtifact,
    prestress_state: &CanonicalSemanticState,
    result: &StructuralBucklingSolverResult,
    numerical_receipt: &StructuralBucklingNumericalReceipt,
) -> Result<FeaBucklingRunResult, FeaRunError> {
    workload
        .validate()
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    result
        .validate_for(workload)
        .map_err(|error| FeaRunError::Solve(error.to_string()))?;
    numerical_receipt
        .validate_against(numerical_receipt.source_result(), result)
        .map_err(|error| FeaRunError::Solve(error.to_string()))?;
    prestress_state
        .validate()
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    solver_mesh
        .validate_canonical()
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let topology_digest = Digest::from_bytes(*solver_mesh.canonical_digest.bytes());
    if result.topology_digest() != topology_digest
        || result.source_state() != prestress_state.identity
        || workload.prestress_transition().source_state != prestress_state.identity
        || solver_mesh.geometry.geometry_revision != u64::from(workload.model().geometry_revision)
    {
        return Err(FeaRunError::InvalidModel(
            "portable buckling result does not belong to the supplied mesh and prestress state"
                .into(),
        ));
    }
    let step = super::admission::require_structural(
        workload.model(),
        AnalysisProcedure::EigenvalueBuckling,
    )?;
    let excitations =
        realize_step_excitations(workload.model(), step, ExcitationSamplePoint::Static)
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let summary = assemble_structural_perturbation_system(
        workload.model(),
        &excitations,
        Some(solver_mesh.clone()),
        StructuralFormulationRequirement::SyntheticOperator,
    )
    .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let expected_run = super::admission::admit_structural(&summary, step)?;
    if result.run() != &expected_run {
        return Err(FeaRunError::Assembly(
            "portable buckling run identity differs from the recovered operator".into(),
        ));
    }
    let dof_shapes = result
        .modes()
        .iter()
        .map(|mode| mode.dof_shape.clone())
        .collect::<Vec<_>>();
    let projected = project_structural_nodal_snapshots(&summary.structural_dof_layout, &dof_shapes)
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let shape = nodal_vector_shape(&summary.structural_dof_layout);
    let mode_shapes = projected
        .iter()
        .enumerate()
        .map(|(index, mode)| {
            AnalysisField::host_f64(
                fea_buckling_mode_shape_field_id(index + 1),
                AnalysisFieldLayout::vector3(
                    ObservationQuantity::Displacement,
                    FieldLocation::Node,
                )
                .mode(index),
                shape.clone(),
                mode.translations.clone(),
            )
        })
        .collect::<Vec<_>>();
    let rotation_mode_shapes = projected
        .iter()
        .enumerate()
        .map(|(index, mode)| {
            AnalysisField::host_f64(
                fea_buckling_rotation_shape_field_id(index + 1),
                AnalysisFieldLayout::vector3(ObservationQuantity::Rotation, FieldLocation::Node)
                    .mode(index),
                shape.clone(),
                mode.rotations.clone(),
            )
        })
        .collect::<Vec<_>>();
    let recovered_modes = dof_shapes
        .iter()
        .enumerate()
        .map(|(index, mode)| {
            recover_deformation_result_fields(&summary, mode)
                .map(|fields| ModalRecoveredMode {
                    mode_index: index + 1,
                    fields,
                })
                .map_err(|error| FeaRunError::Assembly(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let load_factors = result
        .modes()
        .iter()
        .map(|mode| mode.load_factor)
        .collect::<Vec<_>>();
    let residual_norms = result
        .modes()
        .iter()
        .map(|mode| mode.residual_norm)
        .collect::<Vec<_>>();
    let fields = vec![
        AnalysisField::host_f64(
            FEA_FIELD_BUCKLING_LOAD_FACTOR,
            AnalysisFieldLayout::scalar(ObservationQuantity::LoadFactor, FieldLocation::Mode),
            vec![load_factors.len()],
            load_factors.clone(),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_BUCKLING_RESIDUAL_NORM,
            AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Mode),
            vec![residual_norms.len()],
            residual_norms.clone(),
        ),
    ];
    let run = FeaRunResult {
        identity: expected_run,
        requested_backend: result.requested_backend(),
        execution_backend: ExecutionBackend::from_solver_backend(result.solver_backend()),
        solver_backend: result.solver_backend(),
        solver_device_apply_k_ratio: 0.0,
        solver_method: result.solver_method(),
        preconditioner: result.preconditioner(),
        solver_host_sync_count: 0,
        accelerator_evidence: Vec::new(),
        diagnostics: vec![FeaDiagnostic {
            code: "FEA_BUCKLING_CONVERGENCE".into(),
            severity: FeaDiagnosticSeverity::Info,
            message: format!(
                "qualified {} positive buckling modes in {} iterations; maximum residual={}",
                result.modes().len(),
                result.iteration_count(),
                numerical_receipt.maximum_residual_norm(),
            ),
        }],
        spatial_projection_evidence: summary.spatial_projection_evidence,
        structural_static_convergence: None,
        structural_static_balance: None,
        structural_constraint_result: None,
        structural_connector_result: None,
        observations: Vec::new(),
        observation_histories: Vec::new(),
        series_reductions: Vec::new(),
        fields,
    };
    run.validate_solver_formulation()?;
    Ok(FeaBucklingRunResult {
        run,
        load_factors,
        residual_norms,
        mode_shapes,
        rotation_mode_shapes,
        recovered_modes,
    })
}
