use runmat_analysis_core::{validate_model, AnalysisField, AnalysisModel, AnalysisProcedure};
use runmat_analysis_core::{AnalysisFieldLayout, FieldLocation, ObservationQuantity};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    assembly::solver_solid::SolidEnergyPreparation,
    contracts::{
        ComputeBackend, FeaModalRunResult, FeaRunError, FeaRunResult, ModalSolveOptions,
        FEA_FIELD_MODAL_CUMULATIVE_EFFECTIVE_MASS, FEA_FIELD_MODAL_CUMULATIVE_MASS_RATIO,
        FEA_FIELD_MODAL_EFFECTIVE_MASS, FEA_FIELD_MODAL_EIGENVALUE, FEA_FIELD_MODAL_FREQUENCY_HZ,
        FEA_FIELD_MODAL_MODAL_MASS, FEA_FIELD_MODAL_MODAL_STIFFNESS,
        FEA_FIELD_MODAL_M_ORTHOGONALITY, FEA_FIELD_MODAL_PARTICIPATION_FACTOR,
        FEA_FIELD_MODAL_RELATIVE_FREQUENCY_SEPARATION, FEA_FIELD_MODAL_RESIDUAL_NORM,
        FEA_FIELD_MODAL_TOTAL_FREE_MASS,
    },
    diagnostics::builders::{extend_common_run_diagnostics, CommonRunDiagnosticInputs},
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    execution::{CanonicalSemanticState, StructuralModalSolverResult, StructuralModalWorkload},
    pipeline::modal_basis::{ModalBasisArtifact, ModalBasisInputs},
    pipeline::observation_history::required_mode_history_indices,
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
    solve::modal::solve_modal_system_reusing,
};

mod assembly;
mod recovery;

use recovery::recover_modal_core;
pub use recovery::{
    recover_completed_structural_modal_run, recover_completed_structural_prestressed_modal_run,
};

pub fn run_modal(
    model: &AnalysisModel,
    backend: ComputeBackend,
) -> Result<FeaModalRunResult, FeaRunError> {
    run_modal_with_options(model, backend, ModalSolveOptions::default())
}

pub fn run_modal_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: ModalSolveOptions,
) -> Result<FeaModalRunResult, FeaRunError> {
    execute_modal_numerics(model, backend, options)
}

/// Executes one path-free modal-extraction workload against its separately transported mesh.
///
/// This is the numerical boundary used by portable execution. The workload owns the authored
/// model and solver policy; the canonical mesh remains an independent object whose digest is
/// recorded in the returned result.
pub fn execute_structural_modal_workload(
    workload: &StructuralModalWorkload,
    solver_mesh: &SolverMeshArtifact,
) -> Result<StructuralModalSolverResult, FeaRunError> {
    if workload.prestress_transition().is_some() {
        return Err(FeaRunError::InvalidModel(
            "prestressed modal extraction requires its exact geometric-stiffness state".into(),
        ));
    }
    execute_structural_modal_workload_with_state(workload, solver_mesh, None)
}

pub fn execute_structural_prestressed_modal_workload(
    workload: &StructuralModalWorkload,
    solver_mesh: &SolverMeshArtifact,
    prestress_state: &CanonicalSemanticState,
) -> Result<StructuralModalSolverResult, FeaRunError> {
    execute_structural_modal_workload_with_state(workload, solver_mesh, Some(prestress_state))
}

fn execute_structural_modal_workload_with_state(
    workload: &StructuralModalWorkload,
    solver_mesh: &SolverMeshArtifact,
    prestress_state: Option<&CanonicalSemanticState>,
) -> Result<StructuralModalSolverResult, FeaRunError> {
    workload
        .validate()
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    solver_mesh
        .validate_canonical()
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    if solver_mesh.geometry.geometry_revision != u64::from(workload.model().geometry_revision) {
        return Err(FeaRunError::InvalidModel(format!(
            "structural modal workload geometry revision {} does not match solver mesh geometry revision {}",
            workload.model().geometry_revision,
            solver_mesh.geometry.geometry_revision
        )));
    }
    let topology_digest = Digest::from_bytes(*solver_mesh.canonical_digest.bytes());
    let linearization = match (workload.prestress_transition(), prestress_state) {
        (None, None) => None,
        (Some(transition), Some(state)) => Some(
            super::prestress::structural_linearization_from_state(
                transition,
                state,
                topology_digest,
            )
            .map_err(FeaRunError::InvalidModel)?,
        ),
        _ => {
            return Err(FeaRunError::InvalidModel(
                "modal workload and prestress-state presence differ".into(),
            ))
        }
    };
    let procedure_step = workload
        .procedure_step()
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let active_model =
        super::activation::model_with_active_constraints(workload.model(), procedure_step);
    let modal = execute_modal_core(
        &active_model,
        workload.backend(),
        ModalSolveOptions {
            truncation: workload.truncation(),
            convergence: workload.convergence(),
            residual_vector_directions: workload.residual_vector_directions().to_vec(),
            solver_mesh: Some(solver_mesh.clone()),
        },
        procedure_step.procedure,
        linearization,
    )?;
    StructuralModalSolverResult::new(workload, topology_digest, modal.run, modal.basis)
        .map_err(|error| FeaRunError::Solve(error.to_string()))
}

pub(super) struct CompletedModalCore {
    pub(super) run: FeaRunResult,
    pub(super) basis: ModalBasisArtifact,
    pub(super) summary: crate::assembly::AssemblySummary,
}

fn execute_modal_numerics(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: ModalSolveOptions,
) -> Result<FeaModalRunResult, FeaRunError> {
    let execution_step =
        super::admission::require_structural(model, AnalysisProcedure::ModalExtraction)?;
    let solver_mesh = options.solver_mesh.clone();
    let core = execute_modal_core(
        model,
        backend,
        options,
        AnalysisProcedure::ModalExtraction,
        None,
    )?;
    recover_modal_core(model, execution_step, solver_mesh.as_ref(), core)
}

fn execute_modal_core(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: ModalSolveOptions,
    procedure: AnalysisProcedure,
    linearization: Option<
        &crate::pipeline::structural_linearization::StructuralLinearizationArtifact,
    >,
) -> Result<CompletedModalCore, FeaRunError> {
    let execution_step = super::admission::require_structural(model, procedure)?;
    crate::contracts::accelerator_admission::admit_accelerator(
        crate::contracts::accelerator_admission::AcceleratorFamily::StructuralModal,
        backend,
    )?;
    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Started,
        "validating modal FEA model",
        Some(0),
        Some(5),
    );
    check_cancelled("fea.run_modal")?;
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;
    let required_history_modes = required_mode_history_indices(execution_step)
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    options.truncation.validate()?;
    options.convergence.validate()?;
    if required_history_modes
        .last()
        .is_some_and(|mode| *mode > options.truncation.requested_mode_count())
    {
        return Err(FeaRunError::Observation(
            "modal history requests a mode beyond the admitted solver mode count".into(),
        ));
    }
    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Completed,
        "modal model validation complete",
        Some(1),
        Some(5),
    );
    check_cancelled("fea.run_modal")?;

    let excitations =
        realize_step_excitations(model, execution_step, ExcitationSamplePoint::Static)
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;

    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Started,
        "assembling modal system",
        Some(1),
        Some(5),
    );
    let mut solid_energy = SolidEnergyPreparation::default();
    let summary = assembly::assemble(
        model,
        &excitations,
        options.solver_mesh.as_ref(),
        linearization.is_some(),
        &mut solid_energy,
    )?;
    let summary = if let Some(linearization) = linearization {
        super::prestress::prestressed_structural_summary(&summary, linearization)
            .map_err(FeaRunError::Assembly)?
    } else {
        summary
    };
    let identity = super::admission::admit_structural(&summary, execution_step)?;
    super::validate_rotational_dof_targets(model, &summary)?;
    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Completed,
        "modal system assembly complete",
        Some(2),
        Some(5),
    );
    check_cancelled("fea.run_modal")?;

    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Started,
        "solving modal basis",
        Some(2),
        Some(5),
    );
    let modal = solve_modal_system_reusing(
        &summary,
        options.truncation.requested_mode_count(),
        options.truncation.eigenspace_relative_tolerance(),
        options.convergence,
        backend,
        &mut solid_energy.into_reuse(),
    );
    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Completed,
        "modal solve complete",
        Some(3),
        Some(5),
    );
    check_cancelled("fea.run_modal")?;

    let mut diagnostics = modal.diagnostics.clone();
    if !modal.converged {
        return Err(FeaRunError::Assembly(format!(
            "modal eigensolver did not qualify its requested subspace: {:?}",
            modal.diagnostics
        )));
    }
    extend_common_run_diagnostics(
        &mut diagnostics,
        CommonRunDiagnosticInputs {
            model,
            summary: &summary,
        },
    );

    let basis = ModalBasisArtifact::new(
        identity.clone(),
        &summary,
        ModalBasisInputs {
            frequencies_hz: &modal.frequencies_hz,
            dof_shapes: &modal.mode_shapes,
            residual_norms: &modal.residual_norms,
            truncation_policy: options.truncation,
            available_mode_count: modal.available_mode_count,
            boundary_probe_count: modal.boundary_probe_count,
            boundary_probe_frequency_hz: modal.boundary_probe_frequency_hz,
        },
    )?;
    let residual_directions = super::modal_residual::canonical_residual_vector_directions(
        &options.residual_vector_directions,
    )
    .map_err(FeaRunError::InvalidModel)?;
    let basis = if residual_directions.is_empty() {
        basis
    } else {
        basis.with_directional_residual_vectors(&summary, &residual_directions)?
    };
    diagnostics.push(crate::diagnostics::FeaDiagnostic {
        code: "FEA_MODAL_TRUNCATION".into(),
        severity: crate::diagnostics::FeaDiagnosticSeverity::Info,
        message: format!(
            "requested_mode_count={} retained_mode_count={} available_mode_count={} boundary_probe_count={} boundary_probe_frequency_hz={:?} expanded_for_eigenspace={} complete_boundary_eigenspace={}",
            basis.truncation().policy().requested_mode_count(),
            basis.truncation().retained_mode_count(),
            basis.truncation().available_mode_count(),
            basis.truncation().boundary_probe_count(),
            basis.truncation().boundary_probe_frequency_hz(),
            basis.truncation().expanded_for_eigenspace(),
            basis.truncation().complete_boundary_eigenspace(),
        ),
    });
    diagnostics.push(crate::diagnostics::FeaDiagnostic {
        code: "FEA_MODAL_EIGENSPACES".into(),
        severity: crate::diagnostics::FeaDiagnosticSeverity::Info,
        message: format!(
            "eigenspace_count={} repeated_eigenspace_count={} relative_frequency_tolerance={}",
            basis.eigenspaces().len(),
            basis
                .eigenspaces()
                .iter()
                .filter(|eigenspace| eigenspace.is_repeated())
                .count(),
            basis.truncation().policy().eigenspace_relative_tolerance(),
        ),
    });
    let run = FeaRunResult {
        identity,
        requested_backend: backend,
        execution_backend: crate::contracts::ExecutionBackend::from_solver_backend(
            modal.solver_backend,
        ),
        solver_backend: modal.solver_backend,
        solver_device_apply_k_ratio: if modal.device_apply_k_attempt_count == 0 {
            0.0
        } else {
            modal.device_apply_k_count as f64 / modal.device_apply_k_attempt_count as f64
        },
        solver_method: modal.solver_method,
        preconditioner: modal.preconditioner,
        solver_host_sync_count: modal.solver_host_sync_count,
        accelerator_evidence: modal.accelerator_evidence,
        diagnostics,
        spatial_projection_evidence: summary.spatial_projection_evidence.clone(),
        structural_static_convergence: None,
        structural_static_balance: None,
        structural_constraint_result: None,
        structural_connector_result: None,
        observations: Vec::new(),
        observation_histories: Vec::new(),
        series_reductions: Vec::new(),
        fields: modal_result_fields(&basis),
    };
    run.validate_solver_formulation()?;
    Ok(CompletedModalCore {
        run,
        basis,
        summary,
    })
}

fn modal_result_fields(basis: &ModalBasisArtifact) -> Vec<AnalysisField> {
    let mode_count = basis.modes().len();
    let directional = |select: fn(crate::pipeline::modal_basis::ModalDirectionalMetric) -> f64| {
        basis
            .modes()
            .iter()
            .flat_map(|mode| mode.directional().iter().copied().map(select))
            .collect::<Vec<_>>()
    };
    vec![
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_FREQUENCY_HZ,
            AnalysisFieldLayout::scalar(ObservationQuantity::ModalFrequency, FieldLocation::Mode),
            vec![mode_count],
            basis
                .modes()
                .iter()
                .map(|mode| mode.frequency_hz())
                .collect(),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_EIGENVALUE,
            AnalysisFieldLayout::scalar(ObservationQuantity::Eigenvalue, FieldLocation::Mode),
            vec![mode_count],
            basis
                .modes()
                .iter()
                .map(|mode| mode.eigenvalue_rad2_per_s2())
                .collect(),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_MODAL_MASS,
            AnalysisFieldLayout::scalar(ObservationQuantity::ModalMass, FieldLocation::Mode),
            vec![mode_count],
            basis.modes().iter().map(|mode| mode.modal_mass()).collect(),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_MODAL_STIFFNESS,
            AnalysisFieldLayout::scalar(ObservationQuantity::ModalStiffness, FieldLocation::Mode),
            vec![mode_count],
            basis
                .modes()
                .iter()
                .map(|mode| mode.modal_stiffness())
                .collect(),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_PARTICIPATION_FACTOR,
            AnalysisFieldLayout::vector3(
                ObservationQuantity::ModalParticipation,
                FieldLocation::Mode,
            ),
            vec![mode_count, 3],
            directional(|metric| metric.participation_factor()),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_EFFECTIVE_MASS,
            AnalysisFieldLayout::vector3(ObservationQuantity::ModalMass, FieldLocation::Mode),
            vec![mode_count, 3],
            directional(|metric| metric.effective_mass()),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_CUMULATIVE_EFFECTIVE_MASS,
            AnalysisFieldLayout::vector3(ObservationQuantity::ModalMass, FieldLocation::Mode),
            vec![mode_count, 3],
            directional(|metric| metric.cumulative_effective_mass()),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_CUMULATIVE_MASS_RATIO,
            AnalysisFieldLayout::vector3(ObservationQuantity::MassRatio, FieldLocation::Mode),
            vec![mode_count, 3],
            directional(|metric| metric.cumulative_mass_ratio()),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_TOTAL_FREE_MASS,
            AnalysisFieldLayout::vector3(ObservationQuantity::ModalMass, FieldLocation::Global),
            vec![3],
            basis.total_free_mass().to_vec(),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_RESIDUAL_NORM,
            AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Mode),
            vec![mode_count],
            basis
                .modes()
                .iter()
                .map(|mode| mode.residual_norm())
                .collect(),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_RELATIVE_FREQUENCY_SEPARATION,
            AnalysisFieldLayout::scalar(
                ObservationQuantity::FrequencySeparation,
                FieldLocation::Mode,
            ),
            vec![mode_count],
            basis
                .modes()
                .iter()
                .map(|mode| mode.relative_frequency_separation())
                .collect(),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_M_ORTHOGONALITY,
            // One mode-by-mode matrix for the whole basis: a tensor whatever the mode count.
            AnalysisFieldLayout::new(
                ObservationQuantity::MassOrthogonality,
                FieldLocation::Global,
                runmat_analysis_core::ObservationValueShape::Tensor {
                    extents: vec![mode_count, mode_count],
                },
            ),
            vec![mode_count, mode_count],
            basis.mass_orthogonality().to_vec(),
        ),
    ]
}

#[cfg(test)]
mod curved_tet10_tests;

#[cfg(test)]
pub(crate) mod rod_qualification;

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{IndependentDomain, ProcedureStep, ProcedureStepId};
    use runmat_meshing_core::CanonicalMeshingContract;

    use super::*;
    use crate::{
        execution::{StructuralModalNumericalReceipt, StructuralModalWorkloadInput},
        fixtures::{fixture_model, FixtureId},
        pipeline::modal_governance::{ModalConvergencePolicy, ModalTruncationPolicy},
        results::observation::ObservationExecutionPolicy,
    };

    fn portable_fixture() -> (StructuralModalWorkload, SolverMeshArtifact) {
        let truncation = ModalTruncationPolicy::requested(2);
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        let mut step = ProcedureStep::structural_modal("modes");
        step.independent_domain = IndependentDomain::Eigenvalue {
            requested_count: truncation.requested_mode_count(),
        };
        model.sequence.steps = vec![step];
        let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
        let workload = StructuralModalWorkload::new(StructuralModalWorkloadInput {
            model,
            step_id: ProcedureStepId("modes".into()),
            backend: ComputeBackend::Cpu,
            observation_execution: ObservationExecutionPolicy::scalable(),
            truncation,
            convergence: ModalConvergencePolicy::exploratory(),
            residual_vector_directions: Vec::new(),
            prestress_transition: None,
        })
        .unwrap();
        (workload, mesh)
    }

    #[test]
    fn portable_modal_boundary_is_numerically_equivalent_to_the_in_process_api() {
        let (workload, mesh) = portable_fixture();
        let in_process = run_modal_with_options(
            workload.model(),
            workload.backend(),
            ModalSolveOptions {
                truncation: workload.truncation(),
                convergence: workload.convergence(),
                residual_vector_directions: workload.residual_vector_directions().to_vec(),
                solver_mesh: Some(mesh.clone()),
            },
        )
        .unwrap();
        let portable = execute_structural_modal_workload(&workload, &mesh).unwrap();

        assert_eq!(
            portable.topology_digest(),
            Digest::from_bytes(*mesh.canonical_digest.bytes())
        );
        portable.validate_for(&workload).unwrap();
        let source_workload =
            crate::execution::AnalysisWorkloadIdentity::for_test("portable-modal-equivalence")
                .unwrap();
        let receipt = StructuralModalNumericalReceipt::new(source_workload, &portable).unwrap();
        let recovered =
            recover_completed_structural_modal_run(&workload, &mesh, &portable, &receipt).unwrap();

        assert_eq!(portable.basis(), &in_process.basis);
        assert_eq!(recovered, in_process);
    }

    #[test]
    fn portable_modal_boundary_rejects_a_canonical_mesh_from_another_geometry_revision() {
        let (workload, mut mesh) = portable_fixture();
        mesh.geometry.geometry_revision += 1;
        mesh.canonical_digest = runmat_meshing_core::StableDigest::ZERO;
        mesh.seal_canonical_digest().unwrap();

        mesh.validate_canonical().unwrap();
        let error = execute_structural_modal_workload(&workload, &mesh).unwrap_err();
        assert!(matches!(error, FeaRunError::InvalidModel(_)));
    }
}
