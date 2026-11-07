//! Genuine scheduler observations, including cold publication-source replay and reads.
use super::super::publication_chain::MemoryStore;
use super::*;
use runmat_analysis_execution::{import_thermal_study_root, prepare_thermal_study_root};
use runmat_execution_artifact::cache::CacheImport;

fn cases() -> impl Iterator<Item = (ElementOrder, PartialAccessPolicy)> {
    [ElementOrder::Tet4, ElementOrder::Tet10]
        .into_iter()
        .flat_map(|order| {
            [
                PartialAccessPolicy::WholeArtifact,
                PartialAccessPolicy::ManifestAndChunks,
            ]
            .into_iter()
            .map(move |storage| (order, storage))
        })
}

fn preparation(
    order: ElementOrder,
    storage: PartialAccessPolicy,
    scope: &str,
    revision: ProgramRevision,
) -> ThermalPrestressStudyPreparation {
    let heat = fixture::setup(base(order, scope, revision), false);
    let mut compiled = heat.product.compiled;
    let step = compiled
        .model
        .sequence
        .steps
        .iter_mut()
        .find(|step| step.step_id.0 == "heat")
        .unwrap();
    let mut transient = ProcedureStep::thermal_transient("heat");
    transient.state_outputs = step.state_outputs.clone();
    transient.independent_domain = IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: 2.0,
        interval_count: 2,
    };
    transient.observations = super::super::thermal::authored_observations(storage);
    transient.accuracy.resources.maximum_samples = 3;
    *step = transient;
    // The prescribed303.15K face heats an initially293.15K body. Free nodes evolve,
    // so temporal sample substitution cannot hide behind constant equilibrium values.
    compiled.properties = compile_property_map(&compiled.model, Some(&heat.mesh)).unwrap();
    let workloads = plan_structural_study(
        &compiled,
        StructuralStudyPlanningPolicy {
            backend: ComputeBackend::Cpu,
            numerics: StructuralNumericalPlanningPolicy::production(),
        },
    )
    .unwrap();
    ThermalPrestressStudyPreparation {
        structural: StructuralPrestressedFrequencyStudyPreparation {
            authored_study_digest: compiled.definition_digest().unwrap(),
            frozen_project_digest: None,
            workloads,
            mesh: heat.mesh,
            artifact_access: heat.artifact_access,
            program_revision: heat.program_revision,
            chunk_bytes: heat.output_policy.chunk_bytes(),
            inventory_limits: heat.output_policy.inventory_limits(),
            required_capabilities: Default::default(),
            accelerator_requests: Vec::new(),
            local_execution: LocalExecutionPlan::serial(),
        },
        compiled,
    }
}

pub(super) async fn run_local() {
    let directory = tempfile::tempdir().unwrap();
    for (ordinal, (order, storage)) in cases().enumerate() {
        let mut config = NativeExecutionConfig::for_current_executable().unwrap();
        config.executable = std::env::current_exe().unwrap();
        config.max_workers = 2;
        config.worker_capabilities = super::super::capabilities();
        config.store_root = directory.path().join(format!("observed-coupled-{ordinal}"));
        let (completed, source) = execute_thermal_prestress_study_locally_with_source(
            config,
            preparation(
                order,
                storage,
                "analysis-process-conformance",
                super::super::revision(),
            ),
            policy(),
        )
        .await
        .unwrap();
        assert_completion(&completed, 1);
        assert_cold_reads(completed.thermal, &source, storage);
    }
}

pub(super) async fn run_remote(pool: Arc<RemotePoolDriver>, revision: ProgramRevision) {
    for (order, storage) in cases() {
        let completed = execute_thermal_prestress_study_remotely(
            Arc::clone(&pool),
            preparation(order, storage, "remote-analysis", revision.clone()),
            policy(),
        )
        .await
        .unwrap();
        assert_completion(&completed, 1);
        assert_cold_reads(completed.thermal, pool.as_ref(), storage);
    }
}

fn assert_cold_reads(
    completed: runmat_analysis_execution::CompletedThermalStudy,
    source: &impl CacheImport,
    storage: PartialAccessPolicy,
) {
    let prepared = prepare_thermal_study_root(&completed).unwrap();
    let access = completed.preparation().artifact_access;
    let limits = completed.preparation().output_policy.inventory_limits();
    let catalog = completed.observations.catalog.clone().unwrap();
    assert_eq!(
        completed.publication.result.source_observations,
        Some(catalog.identity())
    );
    assert_eq!(
        completed.accuracy.artifact.source_observations,
        Some(catalog.identity())
    );
    assert_eq!(completed.observations.histories.len(), 1);
    assert_eq!(completed.observations.samples.len(), 1);
    let history = &completed.observations.histories[0];
    let coordinates = history
        .history
        .manifest
        .samples
        .iter()
        .map(|sample| sample.coordinate)
        .collect::<Vec<_>>();
    assert_eq!(coordinates.len(), 3);
    let series = history.series_reduction.clone().unwrap();
    let mut cold = MemoryStore::default();
    for object in &prepared.objects {
        cold.0
            .insert(object.descriptor.digest, object.bytes.clone());
    }
    for reference in &prepared.catalog.objects {
        cold.0.entry(reference.logical_digest).or_insert_with(|| {
            source
                .read_verified(reference.logical_digest)
                .unwrap()
                .unwrap_or_else(|| panic!("completed thermal closure omitted {reference:?}"))
        });
    }
    drop(completed);
    let imported = import_thermal_study_root(
        &cold,
        &prepared.root,
        prepared.root.logical_digest,
        &access,
        limits,
    )
    .unwrap();
    let (dto, reader) =
        runmat_execution_runner_native::thermal_study_completion_from_source(imported, cold)
            .unwrap();
    let observations = dto.observations.as_ref().unwrap();
    assert_eq!(observations.catalog, catalog);
    let history = &observations.histories[0];
    let first = reader
        .read_history_sample(catalog.identity(), history.history, coordinates[0])
        .unwrap();
    let last = reader
        .read_history_sample(catalog.identity(), history.history, coordinates[2])
        .unwrap();
    assert_eq!(first.coordinate, coordinates[0]);
    assert_eq!(last.coordinate, coordinates[2]);
    assert_eq!(last.result.partial_access, storage);
    let before = first.result.field.as_host_f64().unwrap();
    let after = last.result.field.as_host_f64().unwrap();
    assert!(
        before
            .iter()
            .zip(after)
            .any(|(before, after)| (after - before).abs() > 1e-9),
        "heated thermal samples must remain distinguishable"
    );
    assert_eq!(
        reader
            .read_series_reduction(
                catalog.identity(),
                history.series.as_ref().unwrap().identity
            )
            .unwrap(),
        series
    );
    let final_field = reader
        .read_observation(
            catalog.identity(),
            observations.samples[0].artifact,
            &ObservationId("final-temperature".into()),
        )
        .unwrap();
    assert_eq!(final_field.field.as_host_f64().unwrap(), after);
    assert!(reader
        .read_observation(
            catalog.identity(),
            observations.samples[0].artifact,
            &ObservationId("foreign-temperature".into())
        )
        .is_err());
}
