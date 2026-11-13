//! Authored ordered blocks and terminal authority through terminated real producers.
mod execution;
use super::*;
pub(super) use execution::execute_preparation;
use runmat_analysis_execution::{
    import_nonlinear_solid_study_completion, NonlinearSolidStudyCoordinator,
    NonlinearSolidStudyPhase, NonlinearSolidStudyPreparation, PreparedAnalysisProgram,
};
use runmat_analysis_fea::{
    planning::resource::FeaResourcePlan, results::accuracy::AccuracyStatus,
    study::plan_nonlinear_solid_study_product,
};
use runmat_execution_artifact::{cache::FilesystemObjectStore, ProgramExecutionResponse};

fn preparation(quadratic: bool) -> NonlinearSolidStudyPreparation {
    preparation_with_fields(quadratic, false)
}
fn preparation_with_fields(quadratic: bool, final_fields: bool) -> NonlinearSolidStudyPreparation {
    let (workload, mesh) = fixture(quadratic);
    preparation_from(workload, mesh, final_fields)
}

pub(super) fn preparation_from(
    mut workload: NonlinearSolidWorkload,
    mesh: SolverMeshArtifact,
    final_fields: bool,
) -> NonlinearSolidStudyPreparation {
    if final_fields {
        workload.model.sequence.steps[0].observations = vec![
            super::observations::field(
                "displacement",
                ObservationQuantity::Displacement,
                FieldLocation::Node,
                ObservationValueShape::Vector3,
            ),
            super::observations::field(
                "stress",
                ObservationQuantity::Stress,
                FieldLocation::Element,
                ObservationValueShape::SymmetricTensor3,
            ),
        ];
    }
    // The receipt answers every mandatory criterion of the fixture's plan (Newton closure and the
    // achieved tangent-solve residual), so its report is satisfied; an unavailable criterion
    // would still publish here, labeled indeterminate, under this explicit policy.
    workload.model.sequence.steps[0]
        .accuracy
        .unavailable_evidence = runmat_analysis_core::UnavailableEvidencePolicy::MarkIndeterminate;
    let product = plan_nonlinear_solid_study_product(
        super::planning::compiled(&workload, &mesh),
        &mesh,
        workload.controls.clone(),
    )
    .unwrap();
    let task = prepare(
        &workload,
        &mesh,
        NonlinearSolidBlockRequest::initial(
            &workload,
            Digest::from_bytes(*mesh.canonical_digest.bytes()),
            9,
        )
        .unwrap(),
        None,
    );
    NonlinearSolidStudyPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        product,
        mesh,
        accepted_block_limit: 9,
        frozen_project_digest: None,
        program_revision: task.program.request.recipe.program_revision,
        artifact_access: task.host.artifact_access,
        output_policy: task.output_policy,
    }
}

fn verify_lifecycle(quadratic: bool, final_fields: bool) {
    let p = preparation_with_fields(quadratic, final_fields);
    let policy = p.output_policy;
    let (completed, directory) = execute_preparation(p);
    let open = || {
        FilesystemObjectStore::open(
            directory.path().join("store"),
            policy.inventory_limits().max_object_bytes,
        )
        .unwrap()
    };
    let expected_phases = if final_fields { 13 } else { 7 };
    assert_eq!(completed.closures.len(), expected_phases);
    assert_eq!(completed.states.len(), 3);
    assert_eq!(
        completed.baseline_usage.evidence.attempts.len(),
        expected_phases - 1
    );
    assert_eq!(
        completed.terminal_usage.evidence.attempts.len(),
        expected_phases
    );
    assert_eq!(
        completed.baseline_usage.evidence.result_digest,
        completed
            .terminal_block
            .block
            .source_result_identity()
            .unwrap()
            .digest()
    );
    assert!(completed
        .terminal_block
        .block
        .result
        .accepted_state
        .point_history()
        .all(|(_, _, point)| (point.equivalent_plastic_strain - 0.02).abs() < 2e-9));
    super::durable::verify_original_root(&completed, &open(), policy);
    if final_fields {
        let catalog = completed.observation_catalog.as_ref().unwrap();
        catalog
            .validate_source_result(
                completed.workload.producer_identity().unwrap(),
                runmat_analysis_fea::execution::AccuracySourceKind::NonlinearSolid,
                completed
                    .terminal_block
                    .block
                    .source_result_identity()
                    .unwrap(),
            )
            .unwrap();
        let stress = runmat_analysis_execution::import_observation_result(
            &open(),
            completed.observations.as_ref().unwrap(),
            &ObservationId("stress".into()),
            policy.inventory_limits(),
        )
        .unwrap();
        for tensor in stress.field.as_host_f64().unwrap().chunks_exact(6) {
            assert!((tensor[0] - 5.6).abs() < 2e-9);
            assert!(tensor[1..].iter().all(|v| v.abs() < 2e-9));
        }
    }
    let original = completed.terminal_block.block.result.accepted_state.clone();
    let closures = completed.closures;
    let mut reordered = closures.clone();
    reordered.swap(0, 1);
    assert!(import_nonlinear_solid_study_completion(
        &open(),
        preparation_with_fields(quadratic, final_fields),
        reordered
    )
    .is_err());
    let mut missing = closures.clone();
    missing.remove(1);
    assert!(import_nonlinear_solid_study_completion(
        &open(),
        preparation_with_fields(quadratic, final_fields),
        missing
    )
    .is_err());
    let mut trailing = closures.clone();
    trailing.push(closures[0].clone());
    assert!(import_nonlinear_solid_study_completion(
        &open(),
        preparation_with_fields(quadratic, final_fields),
        trailing
    )
    .is_err());
    let cold = import_nonlinear_solid_study_completion(
        &open(),
        preparation_with_fields(quadratic, final_fields),
        closures,
    )
    .unwrap();
    assert_eq!(cold.terminal_block.block.result.accepted_state, original);
    assert_eq!(cold.states, completed.states);
    assert_eq!(cold.observation_catalog, completed.observation_catalog);
    assert_eq!(cold.fields, completed.fields);
    assert_eq!(cold.observations, completed.observations);
    assert_eq!(cold.publication, completed.publication);
}

#[test]
fn authored_j2_coordinator_cold_replay_preserves_every_original_phase_and_blocks_production() {
    for quadratic in [false, true] {
        verify_lifecycle(quadratic, false);
    }
}
#[test]
fn original_final_observations_survive_terminated_tasks_reverse_completion_and_cold_replay() {
    for quadratic in [false, true] {
        verify_lifecycle(quadratic, true);
    }
}

#[test]
fn coordinator_rejects_substituted_authored_workload_and_insufficient_retention_ceiling() {
    let mut p = preparation(false);
    // Controls are caller-owned planning inputs; changing only the model violates authored authority.
    p.product.workload.model.sequence.steps[0].step_id = ProcedureStepId("foreign".into());
    assert!(NonlinearSolidStudyCoordinator::new(p).is_err());
    let mut p = preparation(false);
    p.product.compiled.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1024);
    p.product = plan_nonlinear_solid_study_product(
        p.product.compiled.clone(),
        &p.mesh,
        p.product.workload.controls.clone(),
    )
    .unwrap();
    assert!(NonlinearSolidStudyCoordinator::new(p).is_err());
}
