//! Complete production-coordinator lifecycle, including transactional failed observation imports.
use super::{execute, planning_policy, preparation};
use runmat_analysis_core::*;
use runmat_analysis_execution::*;
use runmat_analysis_fea::{
    execution::ObservationHistorySourceContext,
    results::observation::{ObservationPublicationCatalog, ObservationPublicationEntry},
    study::plan_thermal_study,
};
use runmat_execution_artifact::{cache::FilesystemObjectStore, ProgramExecutionResponse};
use runmat_meshing_core::ElementOrder;

#[test]
fn thermal_coordinator_publishes_mixed_histories_and_final_fields() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for storage in [
            PartialAccessPolicy::WholeArtifact,
            PartialAccessPolicy::ManifestAndChunks,
        ] {
            for transient in [false, true] {
                run(order, storage, transient);
            }
        }
    }
}

fn run(order: ElementOrder, storage: PartialAccessPolicy, transient: bool) {
    let mut setup = preparation(order, transient);
    setup.product.compiled.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_samples = 4;
    let histories = super::history_finalization::histories(storage);
    let ObservationRequest::History(mut final_temperature) = histories[0].clone() else {
        unreachable!()
    };
    final_temperature.observation_id = ObservationId("final-temperature".into());
    final_temperature.sampling = SamplingPolicy::Final;
    final_temperature.reduction = None;
    let step = &mut setup.product.compiled.model.sequence.steps[0];
    if transient {
        step.observations = histories;
    }
    step.observations
        .push(ObservationRequest::Field(final_temperature));
    let step = step.clone();
    setup.product.workload =
        plan_thermal_study(&setup.product.compiled, planning_policy()).unwrap();
    let workload = setup.product.workload.clone();
    let mesh = setup.mesh.clone();
    let limits = setup.output_policy.inventory_limits();
    let mut coordinator = ThermalStudyCoordinator::new(setup).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let mut store =
        FilesystemObjectStore::open(directory.path().join("objects"), limits.max_object_bytes)
            .unwrap();
    let empty =
        FilesystemObjectStore::open(directory.path().join("empty"), limits.max_object_bytes)
            .unwrap();
    let solve = coordinator.prepare_solve().unwrap();
    let (response, accounting) = execute(
        &mut store,
        &solve.host,
        &solve.program,
        &solve.resource_plan,
        0,
    );
    coordinator
        .complete_solve(&store, response, accounting)
        .unwrap();
    assert_eq!(coordinator.phase(), ThermalStudyPhase::ReadyForObservations);
    assert!(coordinator.prepare_accuracy().is_err());
    let mut ordinal = 1;
    let mut observation_plans = Vec::new();
    loop {
        let Some(task) = coordinator.prepare_next_observation(&store).unwrap() else {
            break;
        };
        let plan = task.resource_plan.identity;
        let (response, accounting) = execute(
            &mut store,
            task.host,
            task.program,
            task.resource_plan,
            ordinal,
        );
        let wrong_response = ProgramExecutionResponse::ExternalizedSuccess {
            outputs: Vec::new(),
            result_objects: Vec::new(),
        };
        assert!(coordinator
            .complete_observation(&store, wrong_response, accounting.clone())
            .is_err());
        assert!(coordinator
            .complete_observation(&empty, response.clone(), accounting.clone())
            .is_err());
        assert_eq!(coordinator.phase(), ThermalStudyPhase::AwaitingObservation);
        assert!(coordinator.prepare_accuracy().is_err());
        assert!(coordinator.prepare_next_observation(&store).is_err());
        coordinator
            .complete_observation(&store, response, accounting)
            .unwrap();
        assert_eq!(coordinator.phase(), ThermalStudyPhase::ReadyForObservations);
        observation_plans.push(plan);
        ordinal += 1;
        assert!(
            ordinal < 100,
            "coordinator did not close a bounded workflow"
        );
    }
    assert_eq!(observation_plans.len(), if transient { 55 } else { 3 });
    let catalog = coordinator.observations().catalog.as_ref().unwrap().clone();
    let context = ObservationHistorySourceContext::Thermal {
        workload: &workload,
        mesh: &mesh,
    };
    catalog.validate_for(&step, context).unwrap();
    assert_eq!(
        coordinator.observations().histories.len(),
        if transient { 4 } else { 0 }
    );
    assert_eq!(coordinator.observations().samples.len(), 1);
    if transient {
        // The final artifact contains a temperature history sample too; it must never replace
        // the complete temperature History declaration in the publication coverage proof.
        let mut entries = catalog.entries().to_vec();
        entries.retain(|entry| {
            !matches!(entry, ObservationPublicationEntry::History(history)
            if history.history.observation_id.0 == "temperature")
        });
        assert!(ObservationPublicationCatalog::new(
            catalog.producer(),
            catalog.plan().clone(),
            catalog.source().clone(),
            &step,
            entries,
            context
        )
        .is_err());
        for history in &coordinator.observations().histories {
            assert_eq!(history.history.manifest.observed_sample_count, 4);
            let sample = import_observation_history_sample(
                &store,
                &history.history,
                history.history.manifest.samples[0].coordinate,
                limits,
            )
            .unwrap();
            sample.validate().unwrap();
        }
    }
    let accuracy = coordinator.prepare_accuracy().unwrap();
    let (response, accounting) = execute(
        &mut store,
        &accuracy.host,
        &accuracy.program,
        &accuracy.resource_plan,
        ordinal,
    );
    coordinator
        .complete_accuracy(&store, response, accounting)
        .unwrap();
    ordinal += 1;
    assert_eq!(
        coordinator.accuracy().unwrap().artifact.source_observations,
        Some(catalog.identity())
    );
    for _ in &step.state_outputs {
        let state = coordinator.prepare_next_state().unwrap();
        let (response, accounting) = execute(
            &mut store,
            &state.host,
            &state.program,
            &state.resource_plan,
            ordinal,
        );
        coordinator
            .complete_next_state(&store, response, accounting)
            .unwrap();
        ordinal += 1;
    }
    let baseline = coordinator.freeze_baseline().unwrap().clone();
    assert_eq!(baseline.evidence.attempts.len(), ordinal);
    for plan in observation_plans {
        assert_eq!(
            baseline
                .evidence
                .attempts
                .iter()
                .filter(|attempt| attempt.plan == plan)
                .count(),
            1
        );
    }
    let publication = coordinator.prepare_publication().unwrap();
    let (response, accounting) = execute(
        &mut store,
        &publication.host,
        &publication.program,
        &publication.resource_plan,
        ordinal,
    );
    coordinator
        .complete_publication(&store, response, accounting)
        .unwrap();
    let completed = coordinator.into_completed().unwrap();
    assert_eq!(completed.observations.catalog.unwrap(), catalog);
    assert_eq!(completed.baseline_usage, baseline);
    assert_eq!(
        completed.terminal_usage.evidence.attempts.len(),
        ordinal + 1
    );
    assert_eq!(completed.states.len(), step.state_outputs.len());
}
