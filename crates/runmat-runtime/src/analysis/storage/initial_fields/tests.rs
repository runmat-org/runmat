use super::*;
use crate::analysis::{analysis_run_linear_static_with_options, tests, AnalysisRunOptions};
use crate::operations::OperationContext;
use runmat_analysis_core::{FieldLocation, ObservationQuantity, ObservationValueShape};
use runmat_analysis_fea::{
    execution::{
        FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoverySource, FieldTopologyIdentity,
        SequenceIdentity, SolverResultIdentity, StepIdentity,
    },
    results::field::{
        RecoveredFieldDefinition, RecoveredFieldEntityCounts, RecoveredFieldEntityDomain,
        RecoveredFieldNumericRepresentation, RecoveredFieldSample,
    },
    ComputeBackend,
};

const LIMIT: u64 = 128 * 1024 * 1024;
mod process;

fn source() -> (AnalysisRunResult, RecoveredFieldArtifact) {
    let mut run = analysis_run_linear_static_with_options(
        &tests::sample_model(),
        ComputeBackend::Cpu,
        AnalysisRunOptions::default(),
        OperationContext::new(None, None),
    )
    .unwrap()
    .data;
    run.run_id = "original_field_source".into();
    let field = run
        .run
        .field(runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .unwrap()
        .clone();
    let topology = Digest::sha256(b"original-field-storage-fixture-topology");
    let step = StepIdentity::new(
        SequenceIdentity::new(Digest::sha256(b"source-sequence")).unwrap(),
        Digest::sha256(b"source-step"),
    )
    .unwrap();
    let result = SolverResultIdentity::from_step_result(
        step,
        Digest::sha256(serde_json::to_vec(&run).unwrap()),
    )
    .unwrap();
    let source = FieldRecoverySource::SolverResult(result);
    let request = FieldRecoveryRequest::new(
        step,
        source,
        FieldTopologyIdentity::Structural {
            topology:
                runmat_analysis_fea::constraints::CanonicalStructuralTopologyIdentity::Explicit {
                    structural_topology_digest: topology,
                },
        },
        FieldRecoveryPlan::default(),
    )
    .unwrap();
    let definition = RecoveredFieldDefinition::new(
        &field.field_id,
        ObservationQuantity::Displacement,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Vector3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let counts = RecoveredFieldEntityCounts {
        nodes: field.shape[0],
        beam_elements: 1,
        volume_elements: 0,
        volume_integration_points: 0,
        oriented_edges: 0,
        oriented_faces: 0,
        shell_elements: 0,
        connectors: 0,
        constraint_groups: 0,
    };
    let artifact = RecoveredFieldArtifact::new(
        request.identity(),
        source,
        topology,
        counts,
        vec![RecoveredField::new(definition, RecoveredFieldSample::Final, field).unwrap()],
    )
    .unwrap();
    (run, artifact)
}

fn prepare(
    run: &AnalysisRunResult,
    catalog: &RecoveredFieldArtifactManifest,
) -> PreparedAnalysisPublication {
    let definition = Digest::sha256(b"original-field-publication");
    let builder = AnalysisPublicationBuilder::new(definition, LIMIT).unwrap();
    builder.stage_run(run).unwrap();
    builder.stage_original_fields(&run.run_id, catalog).unwrap();
    assert!(builder.stage_original_fields(&run.run_id, catalog).is_err());
    builder
        .finish(serde_json::json!({"study_definition_digest": definition,
        "public_run": {"steps": [{"run_id": run.run_id}]}}))
        .unwrap()
}

#[test]
fn original_selected_field_survives_closed_publication_and_cold_bounded_read() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let (run, artifact) = source();
    let catalog = artifact.manifest().unwrap();
    let rebuilt = RecoveredFieldArtifactManifest::from_entries(
        catalog.recovery_request,
        catalog.source,
        catalog.topology_digest,
        catalog.entity_counts,
        catalog.fields.clone(),
    )
    .unwrap();
    assert_eq!(rebuilt, catalog);
    let field_id = artifact.fields()[0].definition().field_id();
    let root = std::env::temp_dir().join(format!(
        "runmat-original-field-{}",
        Utc::now().timestamp_nanos_opt().unwrap()
    ));
    let disk = FilesystemAnalysisArtifactStore::new(root.clone());
    let memory = InMemoryAnalysisArtifactStore::new();
    for store in [&memory as &dyn AnalysisArtifactStore, &disk] {
        assert!(store
            .export_initial_field(&run.run_id, field_id, LIMIT)
            .unwrap()
            .is_none());
        store.commit_publication(prepare(&run, &catalog)).unwrap();
        let binding = store
            .export_initial_field(&run.run_id, field_id, LIMIT)
            .unwrap()
            .unwrap();
        assert_eq!(
            binding.reference().content_digest,
            artifact.identity().digest().to_string()
        );
        assert_eq!(binding.artifact(), &artifact);
        assert!(store
            .export_initial_field(&run.run_id, field_id, 1)
            .is_err());
        assert!(store
            .export_initial_field(&run.run_id, "missing", LIMIT)
            .unwrap()
            .is_none());
    }
    let cold = FilesystemAnalysisArtifactStore::new(root.clone());
    assert_eq!(
        cold.export_initial_field(&run.run_id, field_id, LIMIT)
            .unwrap()
            .unwrap()
            .artifact(),
        &artifact
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn altered_original_values_cannot_attach_a_valid_but_foreign_catalog() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let (mut run, artifact) = source();
    let id = artifact.fields()[0].definition().field_id();
    let original = run.run.field(id).unwrap();
    let altered = AnalysisField::host_f64(
        id,
        original.layout.clone(),
        original.shape.clone(),
        vec![123.0; original.shape.iter().product()],
    );
    *run.run
        .fields
        .iter_mut()
        .find(|field| field.field_id == id)
        .unwrap() = altered;
    let builder =
        AnalysisPublicationBuilder::new(Digest::sha256(b"rejected-field"), LIMIT).unwrap();
    builder.stage_run(&run).unwrap();
    assert!(builder
        .stage_original_fields(&run.run_id, &artifact.manifest().unwrap())
        .is_err());
    // Failed attachment leaves the staged record publishable without a source claim.
    let publication = builder
        .finish(serde_json::json!({
            "study_definition_digest": Digest::sha256(b"rejected-field"),
            "public_run": {"steps": [{"run_id": run.run_id}]}
        }))
        .unwrap();
    let store = InMemoryAnalysisArtifactStore::new();
    store.commit_publication(publication).unwrap();
    assert!(store
        .export_initial_field(&run.run_id, id, LIMIT)
        .unwrap()
        .is_none());
}

#[test]
fn selected_read_ignores_unrelated_chunks_and_rejects_selected_corruption() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let (run, artifact) = source();
    let field_id = artifact.fields()[0].definition().field_id();
    let prepared = preparation::prepare_run_artifact(&run, Utc::now().to_rfc3339()).unwrap();
    let selected = prepared
        .persisted
        .fields
        .iter()
        .find(|field| field.field_id == field_id)
        .unwrap();
    let mut required = HashSet::new();
    field_reader::read_field_payload(selected, &mut |digest| {
        required.insert(digest);
        Ok(prepared.chunks.get(&digest).cloned())
    })
    .unwrap();
    let root = std::env::temp_dir().join(format!(
        "runmat-selected-source-{}",
        Utc::now().timestamp_nanos_opt().unwrap()
    ));
    let disk = FilesystemAnalysisArtifactStore::new(root.clone());
    disk.commit_publication(prepare(&run, &artifact.manifest().unwrap()))
        .unwrap();
    let mut removed = 0;
    for digest in prepared
        .chunks
        .keys()
        .filter(|digest| !required.contains(digest))
    {
        std::fs::remove_file(disk.object_path(*digest)).unwrap();
        removed += 1;
    }
    assert!(
        removed > 0,
        "fixture must contain unrelated original fields"
    );
    let cold = FilesystemAnalysisArtifactStore::new(root.clone());
    assert_eq!(
        cold.export_initial_field(&run.run_id, field_id, LIMIT)
            .unwrap()
            .unwrap()
            .artifact(),
        &artifact
    );
    assert!(cold.load_run(&run.run_id).is_err());
    let digest = *required.iter().next().unwrap();
    std::fs::write(disk.object_path(digest), b"corrupt original values").unwrap();
    assert!(cold
        .export_initial_field(&run.run_id, field_id, LIMIT)
        .is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn authored_selection_resolves_once_and_rejects_reference_substitution() {
    use runmat_analysis_core::{
        CoordinateFrame, Displacement, InitialCondition, InitialConditionTarget,
        InitialConditionValue, InitialValue, ProcedureStep,
    };
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let (run, artifact) = source();
    let store = Arc::new(InMemoryAnalysisArtifactStore::new());
    store
        .commit_publication(prepare(&run, &artifact.manifest().unwrap()))
        .unwrap();
    let binding = store
        .export_initial_field(&run.run_id, artifact.fields()[0].field_id.as_str(), LIMIT)
        .unwrap()
        .unwrap();
    set_artifact_store_for_tests(store);
    let step = ProcedureStep::structural_transient("consume");
    let mut model = tests::sample_model();
    let condition = InitialCondition::Displacement(InitialConditionValue::<Displacement> {
        initial_condition_id: "original-displacement".into(),
        target: InitialConditionTarget::WholeModel,
        frame: CoordinateFrame::Global,
        value: InitialValue::FieldArtifact(binding.reference().clone()),
        applies_before_step_id: "consume".into(),
    });
    model.initial_conditions = vec![condition.clone(), condition];
    assert_eq!(
        resolve_initial_fields(&model, &step, LIMIT).unwrap(),
        vec![binding]
    );
    let InitialCondition::Displacement(condition) = &mut model.initial_conditions[0] else {
        unreachable!()
    };
    let InitialValue::FieldArtifact(reference) = &mut condition.value else {
        unreachable!()
    };
    reference.content_digest = Digest::sha256(b"substituted-field").to_string();
    assert!(resolve_initial_fields(&model, &step, LIMIT).is_err());
    reset_artifact_store_for_tests();
}
