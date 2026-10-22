//! Genuine singleton thermal execution, authenticated publication and reopened query evidence.
use super::*;
use runmat_analysis_core::{AnalysisSequence, IndependentDomain, PhysicsFamily, ProcedureStep};
use runmat_analysis_execution::{ThermalStudyPreparation, ThermalStudyScope};
use runmat_analysis_fea::study::{plan_thermal_study, ThermalStudyPlanningPolicy};
use runmat_runtime::analysis::{
    publish_thermal_study_result, AnalysisFieldPageValues, AnalysisFieldRequestOptions,
    AnalysisFieldSelection, AnalysisStudyCompletionIdentity, AnalysisThermalRunOptions,
    PrecisionMode, QualityPolicy,
};
use runmat_runtime::operations::OperationContext;

struct CountedThermalObjects {
    objects: DownloadedThermalObjects,
    reads: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}
impl CacheImport for CountedThermalObjects {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        CacheImport::read_verified(&self.objects, digest)
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.objects.read_verified_bounded(digest, maximum)
    }
}

fn standalone_preparation(
    order: ElementOrder,
    transient: bool,
    identity: &str,
) -> ThermalStudyPreparation {
    // Reuse the real canonical thermal fixture, removing its authored structural consumers
    // before planning. This produces a fresh standalone workload, not a relabeled prefix result.
    let mut prepared = crate::analysis_thermal::tests::prestress::setup(order, false);
    let model = &mut prepared.product.compiled.model;
    model.thermo_mechanical = None;
    model.constraints.retain(|constraint| {
        matches!(
            constraint,
            runmat_analysis_core::AnalysisConstraint::Thermal(_)
        )
    });
    let mut step = model.sequence.steps[0].clone();
    if transient {
        let mut transient = ProcedureStep::thermal_transient(step.step_id.0.clone());
        transient.state_outputs = step.state_outputs;
        transient.independent_domain = IndependentDomain::TimeSeconds {
            origin: 0.0,
            start: 0.0,
            end: 2.0,
            interval_count: 2,
        };
        transient.accuracy.resources.maximum_samples = 3;
        transient.observations = crate::analysis_thermal::tests::history_finalization::histories(
            runmat_analysis_core::PartialAccessPolicy::ManifestAndChunks,
        );
        let runmat_analysis_core::ObservationRequest::History(mut final_sample) =
            transient.observations[0].clone()
        else {
            unreachable!()
        };
        final_sample.observation_id =
            runmat_analysis_core::ObservationId("temperature-final".into());
        final_sample.sampling = runmat_analysis_core::SamplingPolicy::Final;
        final_sample.reduction = None;
        transient
            .observations
            .push(runmat_analysis_core::ObservationRequest::Field(
                final_sample,
            ));
        step = transient;
    }
    model.sequence = AnalysisSequence::from_steps("standalone-thermal-publication", vec![step]);
    prepared.product.compiled.study_id = identity.into();
    prepared.product.compiled.properties =
        runmat_analysis_fea::properties::compile_property_map(model, Some(&prepared.mesh)).unwrap();
    prepared.product.workload = plan_thermal_study(
        &prepared.product.compiled,
        ThermalStudyPlanningPolicy {
            backend: runmat_analysis_fea::ComputeBackend::Cpu,
            element_heat_source: None,
        },
    )
    .unwrap();
    prepared.scope = ThermalStudyScope::Standalone;
    prepared
}

#[test]
fn standalone_thermal_publication_preserves_both_orders_and_cold_queries() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for transient in [false, true] {
            check(order, transient);
        }
    }
}

fn check(order: ElementOrder, transient: bool) {
    let directory = tempfile::tempdir().unwrap();
    let allocation = format!("standalone-thermal-{}", directory.path().display());
    let prepared = standalone_preparation(order, transient, &allocation);
    let mesh = prepared.mesh.clone();
    let evidence = runmat_meshing_core::fixtures::canonical_tetrahedron_meshing_evidence(&mesh);
    let mut pool = RecordingPool::new(directory.path());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let executed = runtime
        .block_on(Box::pin(crate::analysis_thermal::execute_with_pool(
            &mut pool,
            prepared,
            crate::ThermalStudyExecutionPolicy {
                resources: ResourceRequest {
                    cpu_millicores: 1000,
                    memory_bytes: 64 * 1024 * 1024,
                    scratch_bytes: 0,
                    max_wall_millis: 60_000,
                    max_artifact_bytes: 64 * 1024 * 1024,
                    max_egress_bytes: 0,
                    max_relay_bytes: 0,
                    accelerators: Vec::new(),
                    required_capabilities: Default::default(),
                },
                retry: RetryPolicy::IdempotentInfrastructure,
                deadline_unix_millis: None,
                priority: 0,
            },
        )))
        .unwrap();
    let task_count = pool.tasks.len();
    let root = prepare_thermal_study_root(&executed.completed).unwrap();
    let source = RetainedThermalSource {
        inventory_limits: executed.completed.output_policy().inventory_limits(),
        catalog: root.catalog,
        completed: executed.completed,
        store: pool.store.clone(),
        objects: root
            .objects
            .into_iter()
            .map(|object| (object.descriptor.digest, object.bytes))
            .collect(),
    };
    let dto = completion(&source.completed).unwrap();
    let downloaded = DownloadedThermalObjects(
        dto.catalog
            .objects
            .iter()
            .map(|reference| {
                (
                    reference.logical_digest,
                    PublishedObjectSource::read_verified(&source, reference).unwrap(),
                )
            })
            .collect(),
    );
    let (completed, reader) =
        thermal_study_completion_from_source(source.completed.clone(), downloaded.clone()).unwrap();
    let options = AnalysisThermalRunOptions {
        deterministic_mode: true,
        precision_mode: PrecisionMode::Fp64,
        quality_policy: QualityPolicy::Strict,
        snapshot_count: if transient { 3 } else { 1 },
        residual_warn_threshold: completed.workload.residual_target,
        solver_mesh_artifact_path: None,
    };
    let definition = completed.compiled.definition_digest().unwrap();
    let publish = |candidate: &ThermalStudyCompletion,
                   objects: &ThermalStudyObjectSource,
                   candidate_mesh: &runmat_meshing_core::SolverMeshArtifact,
                   controls: &AnalysisThermalRunOptions,
                   digest: Digest| {
        publish_thermal_study_result(
            candidate,
            objects,
            candidate_mesh,
            &evidence,
            controls,
            digest,
            OperationContext::new(None, None),
            AnalysisStudyCompletionIdentity::new(&allocation).unwrap(),
        )
    };
    // An impossible declared source workspace rejects before producer authentication reads.
    // Keep every original numerical control and execute the authentic retry below unchanged.
    let reads = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counted = ThermalStudyObjectSource::new(RetainedThermalSource {
        inventory_limits: source.inventory_limits,
        catalog: source.catalog.clone(),
        completed: source.completed.clone(),
        store: CountedThermalObjects {
            objects: downloaded.clone(),
            reads: reads.clone(),
        },
        objects: BTreeMap::new(),
    });
    let mut oversized = completed.clone();
    oversized.catalog.objects[0].encoded_length = u64::MAX;
    let rejected = publish(&oversized, &counted, &mesh, &options, definition).unwrap_err();
    assert_eq!(
        rejected.error_type,
        runmat_runtime::operations::OperationErrorType::Capacity
    );
    assert_eq!(
        rejected.error_code,
        "RM.FEA.RUN_STUDY.PUBLICATION_RESOURCE_LIMIT"
    );
    assert_eq!(reads.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(pool.tasks.len(), task_count);

    // All negative calls precede publication. The authentic retained closure is immutable;
    // none is allowed to substitute caller metadata or run additional numerical tasks.
    assert!(publish(
        &completed,
        &reader,
        &mesh,
        &options,
        Digest::sha256(b"foreign definition")
    )
    .is_err());
    let foreign_mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(
        if order == ElementOrder::Tet4 {
            ElementOrder::Tet10
        } else {
            ElementOrder::Tet4
        },
    );
    assert!(publish(&completed, &reader, &foreign_mesh, &options, definition).is_err());
    let mut controls = options.clone();
    controls.snapshot_count += 1;
    assert!(publish(&completed, &reader, &mesh, &controls, definition).is_err());
    let mut controls = options.clone();
    controls.residual_warn_threshold *= 2.0;
    assert!(publish(&completed, &reader, &mesh, &controls, definition).is_err());
    let mut forged = completed.clone();
    forged.compiled.model.sequence.steps[0].family = PhysicsFamily::Structural;
    assert!(publish(&forged, &reader, &mesh, &options, definition).is_err());
    let mut forged = completed.clone();
    forged.field_partitions.clear();
    assert!(publish(&forged, &reader, &mesh, &options, definition).is_err());
    let mut forged = completed.clone();
    forged.states.clear();
    assert!(publish(&forged, &reader, &mesh, &options, definition).is_err());
    let mut missing = downloaded;
    let descendant = completed
        .catalog
        .objects
        .iter()
        .find(|reference| reference.id != completed.root.id)
        .unwrap();
    missing.0.remove(&descendant.logical_digest);
    let incomplete = ThermalStudyObjectSource::new(RetainedThermalSource {
        inventory_limits: source.inventory_limits,
        catalog: source.catalog.clone(),
        completed: source.completed.clone(),
        store: missing,
        objects: BTreeMap::new(),
    });
    assert!(publish(&completed, &incomplete, &mesh, &options, definition).is_err());
    assert_eq!(pool.tasks.len(), task_count);

    let published = publish(&completed, &reader, &mesh, &options, definition).unwrap();
    assert_eq!(
        published.data.steps.len(),
        1,
        "no fabricated structural suffix"
    );
    assert_eq!(published.data.states.len(), completed.states.len());
    assert_eq!(published.data.study_fingerprint, definition.to_string());
    assert!(published.data.solver_mesh_artifact_path.is_some());
    assert!(published.data.meshing_evidence_artifact_path.is_some());
    let result_manifest_path = published
        .data
        .result_manifest_artifact_path
        .as_ref()
        .expect("standalone study persists its actual whole-study result manifest");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(result_manifest_path).unwrap()).unwrap();
    assert_eq!(
        manifest["study_definition_digest"],
        serde_json::to_value(definition).unwrap()
    );
    assert_eq!(
        manifest["solver_mesh_digest"],
        serde_json::to_value(mesh.canonical_digest).unwrap()
    );
    assert_eq!(
        manifest["compiled"],
        serde_json::to_value(&completed.compiled).unwrap()
    );
    assert_eq!(
        manifest["workloads"],
        serde_json::to_value(&completed.workload).unwrap()
    );
    assert_eq!(manifest["numerical_evidence"]["kind"], "thermal");
    assert_eq!(
        manifest["numerical_evidence"]["root"],
        serde_json::to_value(&completed.root).unwrap()
    );
    assert_eq!(
        manifest["numerical_evidence"]["catalog"],
        serde_json::to_value(&completed.catalog).unwrap()
    );
    for path in manifest["numerical_evidence"]["object_artifact_paths"]
        .as_array()
        .unwrap()
    {
        let path = std::path::Path::new(path.as_str().unwrap());
        let reference = completed
            .catalog
            .objects
            .iter()
            .find(|reference| {
                path.file_name().unwrap().to_str().unwrap()
                    == format!(
                        "sha256_{}.object",
                        reference
                            .logical_digest
                            .to_string()
                            .trim_start_matches("sha256:")
                    )
            })
            .expect("published binary object belongs to original catalog");
        let bytes = runmat_filesystem::NativeFsProvider
            .read_bounded_sync(path, reference.encoded_length)
            .unwrap();
        assert_eq!(bytes.len() as u64, reference.encoded_length);
        assert_eq!(Digest::sha256(&bytes), reference.logical_digest);
    }
    let step = &published.data.steps[0];
    assert_eq!(step.identity.family(), PhysicsFamily::Thermal);
    assert_eq!(
        step.procedure_step,
        completed.compiled.model.sequence.steps[0]
    );
    for (actual, retained) in published.data.states.iter().zip(&completed.states) {
        assert_eq!(actual.identity, retained.identity);
    }
    let run = runmat_runtime::analysis::storage::load_run_result(&step.run_id)
        .unwrap()
        .unwrap();
    assert!(run.run.fields.is_empty());
    let histories = run.thermal_results.as_ref().unwrap();
    assert_eq!(histories.time_points_s, completed.manifest.time_points_s);
    assert_eq!(
        histories.temperature_snapshots.len(),
        options.snapshot_count
    );
    let final_temperature = histories.temperature_snapshots.last().unwrap();
    let selected = AnalysisFieldSelection::new(final_temperature.field_id.clone(), None);
    let page = runmat_runtime::analysis::analysis_field_page_by_run_id_op(
        &step.run_id,
        selected,
        AnalysisFieldRequestOptions::default(),
        OperationContext::new(None, None),
    )
    .unwrap();
    let AnalysisFieldPageValues::Real(values) = page.data.values else {
        panic!("thermal field is real");
    };
    assert_eq!(values, final_temperature.as_host_f64().unwrap());
    if transient {
        super::observation_queries::check_persistence(
            &step.run_id,
            &run,
            &completed,
            &reader,
            &source,
        );
    }
    let replayed = publish(&completed, &reader, &mesh, &options, definition).unwrap();
    assert_eq!(
        replayed.data, published.data,
        "retry preserves immutable publication"
    );
    assert_eq!(
        pool.tasks.len(),
        task_count,
        "publication, negative admission and cold queries never solve again"
    );
}
