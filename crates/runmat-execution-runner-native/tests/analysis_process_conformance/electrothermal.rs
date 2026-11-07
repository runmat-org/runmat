//! Retained one-way Joule loss through actual admitted workers and cold completion replay.
use runmat_analysis_core::{PartialAccessPolicy, RequiredAccuracyEvidence};
use runmat_analysis_execution::{
    import_thermal_study_root, prepare_thermal_study_root, AnalysisOutputPolicy,
    ThermalStudyPreparation, ThermalStudyScope,
};
use runmat_analysis_fea::{
    execution::RetainedJouleHeatSource,
    properties::compile_property_map,
    study::{
        plan_thermal_step, CompiledAnalysisStudy, ThermalStudyPlanningPolicy, ThermalStudyProduct,
    },
    ComputeBackend,
};
use runmat_execution_artifact::{
    cache::{CacheExport, CacheImport, FilesystemObjectStore},
    ArtifactAccess, LogicalObject, ObjectNamespace,
};
use runmat_execution_runner_native::{
    execute_thermal_study_locally_with_source, NativeAnalysisHostLimits, NativeExecutionConfig,
    ThermalStudyExecutionPolicy,
};
use runmat_meshing_core::ElementOrder;

#[path = "../../src/analysis_kernel/electrical_static/fixture.rs"]
mod electrical;
use electrical::electrical_fixture;

#[path = "../../src/analysis_kernel/thermal/tests/electrothermal/fixture.rs"]
mod fixture;

pub(super) async fn run() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (retained, _, mesh) = fixture::fixture(order);
        let mut model = retained.thermal_intent().model.clone();
        let step = &mut model.sequence.steps[0];
        step.observations =
            super::thermal::authored_observations(PartialAccessPolicy::ManifestAndChunks);
        // A steady consumer has one accepted sample, not a transient history integral.
        step.observations.retain(|request| {
            matches!(request, runmat_analysis_core::ObservationRequest::Field(_))
        });
        let mut energy = step.accuracy.criteria[0].clone();
        energy.criterion_id = "energy_balance".into();
        energy.evidence = RequiredAccuracyEvidence::ConservationBalance;
        step.accuracy.criteria.push(energy);
        let compiled = CompiledAnalysisStudy {
            module_path: "conformance/electrothermal.fea".into(),
            study_id: "one_way_heat".into(),
            study_inputs: vec![],
            assessments: Vec::new(),
            outputs: vec![],
            properties: compile_property_map(&model, Some(&mesh)).unwrap(),
            model,
            solver_mesh_digest: mesh.canonical_digest,
        };
        let intent = plan_thermal_step(
            &compiled,
            "heat",
            ThermalStudyPlanningPolicy {
                backend: ComputeBackend::Cpu,
                element_heat_source: None,
            },
        )
        .unwrap();
        let source = RetainedJouleHeatSource::new(
            retained.electrical_workload().clone(),
            retained.electrical_result().clone(),
            retained
                .electrical_result()
                .source_result_identity()
                .unwrap(),
            intent,
            &mesh,
        )
        .unwrap();
        drop(retained);
        let bytes = source.canonical_bytes().unwrap();
        drop(source);
        let source = RetainedJouleHeatSource::from_canonical_bytes(&bytes).unwrap();
        let workload = source.bind(&mesh).unwrap();
        let source_identity = source.identity().unwrap();
        let directory = tempfile::tempdir().unwrap();
        let producing_root = directory.path().join("producing");
        let mut config = NativeExecutionConfig::for_current_executable().unwrap();
        config.max_workers = 1;
        config.store_root = producing_root.clone();
        let limits = NativeAnalysisHostLimits::default();
        let mut resources = super::analysis_resources();
        resources.memory_bytes = 1024 * 1024 * 1024;
        let executed = execute_thermal_study_locally_with_source(
            config,
            ThermalStudyPreparation {
                local_execution:
                    runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
                scope: ThermalStudyScope::SelectedStep,
                field_source: Some(source),
                product: ThermalStudyProduct { compiled, workload },
                mesh,
                frozen_project_digest: None,
                artifact_access: ArtifactAccess {
                    authorization_scope: "retained-electrothermal".into(),
                    encryption_context: super::digest("retained-electrothermal"),
                },
                program_revision: super::revision(),
                output_policy: AnalysisOutputPolicy::new(limits.chunk_bytes, limits.inventory)
                    .unwrap(),
                required_capabilities: Default::default(),
            },
            ThermalStudyExecutionPolicy {
                resources,
                retry: runmat_execution::task::RetryPolicy::IdempotentInfrastructure,
                deadline_unix_millis: None,
                priority: 0,
            },
        )
        .await
        .unwrap();
        let completed = executed.completed;
        let producer = executed.source;
        assert!(
            (completed.result.manifest.energy_balances[0].applied_power_w
                - runmat_types::physical::Quantity::new(12.0))
            .abs()
                < runmat_types::physical::Quantity::new(1e-8)
        );
        assert!(
            (completed.result.manifest.energy_balances[0].prescribed_boundary_power_w
                - runmat_types::physical::Quantity::new(12.0))
            .abs()
                < runmat_types::physical::Quantity::new(1e-8)
        );
        assert!(!completed.states.is_empty());
        assert!(!completed.observations.samples.is_empty());
        assert_eq!(
            completed.accuracy.artifact.gate().status,
            runmat_analysis_fea::results::accuracy::AccuracyStatus::Satisfied
        );
        assert_eq!(completed.accuracy.artifact.evidence().len(), 2);
        let expected_manifest = completed.result.manifest.clone();
        let expected_accuracy = completed.accuracy.artifact.clone();
        let expected_states: Vec<_> = completed
            .states
            .iter()
            .map(|state| state.state.clone())
            .collect();
        let expected_publication = completed.publication.result.clone();
        let access = completed.preparation().artifact_access;
        let prepared = prepare_thermal_study_root(&completed).unwrap();
        let cold_root = directory.path().join("cold");
        let mut cold =
            FilesystemObjectStore::open(&cold_root, limits.inventory.max_object_bytes).unwrap();
        for reference in &prepared.catalog.objects {
            let bytes = prepared
                .objects
                .iter()
                .find(|object| object.descriptor.digest == reference.logical_digest)
                .map(|object| object.bytes.clone())
                .unwrap_or_else(|| {
                    producer
                        .read_verified(reference.logical_digest)
                        .unwrap()
                        .expect("declared child")
                });
            let object = LogicalObject::new(
                ObjectNamespace::ResultValue,
                "electrothermal-cold",
                "application/octet-stream",
                bytes,
            )
            .unwrap();
            assert_eq!(object.descriptor.digest, reference.logical_digest);
            cold.write_verified(&object).unwrap();
        }
        drop(cold);
        drop(producer);
        drop(completed);
        let cold =
            FilesystemObjectStore::open(&cold_root, limits.inventory.max_object_bytes).unwrap();
        // The producing pool and source solver are gone: this is authentication, not a re-solve.
        let imported = import_thermal_study_root(
            &cold,
            &prepared.root,
            prepared.root.logical_digest,
            &access,
            limits.inventory,
        )
        .unwrap();
        assert_eq!(imported.result.manifest, expected_manifest);
        if order == ElementOrder::Tet10 {
            let projection = runmat_analysis_fea::topology::AnalysisProjection::derive(
                &imported.preparation().mesh,
            )
            .unwrap();
            let mut values = Vec::new();
            for descriptor in &imported.result.manifest.fields[0][0].partitions {
                let bytes = imported
                    .result
                    .objects
                    .partition_payload(descriptor.partition.digest())
                    .unwrap();
                let partition = runmat_analysis_fea::results::field::RecoveredFieldPartition::from_canonical_bytes(&bytes).unwrap();
                values.extend_from_slice(partition.storage.as_host_f64().unwrap());
            }
            assert_eq!(values.len(), projection.nodes.len());
            assert!(projection.nodes.iter().zip(values).all(|(node, value)| {
                let x = node.coordinates_m.canonical_value()[0];
                (value - (300.0 + 6.0 * x * (1.0 - x))).abs() < 1e-8
            }));
        }
        assert_eq!(imported.accuracy.artifact, expected_accuracy);
        assert_eq!(
            imported
                .states
                .iter()
                .map(|state| &state.state)
                .collect::<Vec<_>>(),
            expected_states.iter().collect::<Vec<_>>()
        );
        assert_eq!(imported.publication.result, expected_publication);
        assert_eq!(
            imported
                .preparation()
                .field_source
                .unwrap()
                .identity()
                .unwrap(),
            source_identity
        );
        assert_eq!(
            prepare_thermal_study_root(&imported).unwrap().root,
            prepared.root
        );
        eprintln!("retained electrothermal {order:?}: admitted thermal/observations/accuracy/states/publication and cold root passed");
    }
}
