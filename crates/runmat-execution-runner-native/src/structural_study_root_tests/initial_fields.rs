//! Actual solved U/V through portable source closure. CAD consumer physics is separate.
use super::*;
use runmat_analysis_core::{
    AuthoredInitialCondition, AuthoredInitialConditionKind, AuthoredInitialConditionTarget,
    CoordinateFrame, InitialValue, ObservationQuantity,
};
use runmat_analysis_fea::execution::InitialFieldBinding;

pub(crate) fn verify_original_field_transport(fields: &[InitialFieldBinding]) {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("main.fea");
    std::fs::write(&path, SOURCE).unwrap();
    let checked =
        runmat_runtime::analysis::check_fea_path_in_root(temporary.path(), &path).unwrap();
    let resolved = checked.resolve_study("main.fea", "static").unwrap();
    let mut study = resolved
        .study
        .realize_with_models(&resolved.models, &resolved.geometry, &[])
        .unwrap();
    study.sequence = runmat_analysis_core::AnalysisSequence::from_steps(
        "original-source-consumer",
        vec![runmat_analysis_core::ProcedureStep::structural_transient(
            "consume",
        )],
    );
    study.model.initial_conditions = fields
        .iter()
        .enumerate()
        .map(|(index, field)| {
            let kind = match field.artifact().fields()[0].definition().quantity() {
                ObservationQuantity::Displacement => AuthoredInitialConditionKind::Displacement(
                    InitialValue::FieldArtifact(field.reference().clone()),
                ),
                ObservationQuantity::Velocity => AuthoredInitialConditionKind::Velocity(
                    InitialValue::FieldArtifact(field.reference().clone()),
                ),
                _ => panic!("original U/V fixture"),
            };
            AuthoredInitialCondition {
                initial_condition_id: format!("original-{index}"),
                target: AuthoredInitialConditionTarget::WholeModel,
                frame: CoordinateFrame::Global,
                kind,
            }
        })
        .collect();
    let access = ArtifactAccess {
        authorization_scope: "portable-original-fields".into(),
        encryption_context: Digest::sha256("portable-original-fields-key"),
    };
    let source = PreparedGeometrySource::constructed();
    let geometry_request = prepare_geometry_materialization_request(
        resolved.geometry,
        study.geometry.bindings.clone(),
        &source,
        contract_geometry_provider_profile(),
        GeometryMaterializationPreparation::default(),
    )
    .unwrap();
    let geometry = prepare_geometry_materialization_program(
        geometry_request.clone(),
        revision(),
        &access,
        &source.roots,
    )
    .unwrap();
    let geometry_meshing =
        crate::prepare_geometry_meshing_root(crate::GeometryMeshingRootPreparation {
            program_revision: revision(),
            artifact_access: access.clone(),
            geometry,
            geometry_source_objects: source.objects,
            meshing_request: crate::resolve_native_meshing_request(
                &geometry_request.import_options,
                resolved.mesh.settings,
            )
            .unwrap(),
            preferred_edges_per_partition: 8,
            preferred_faces_per_partition: 8,
        })
        .unwrap();
    let preparation = |initial_fields| StructuralStudyRootPreparation {
        program_revision: revision(),
        artifact_access: access.clone(),
        authored_study_digest: runmat_analysis_fea::study::structural_study_source_digest(
            &study,
            &resolved.material_catalog,
        )
        .unwrap(),
        frozen_project_digest: None,
        study: study.clone(),
        materials: resolved.material_catalog.clone(),
        planning: crate::StudyRootPlanning::Structural(StructuralStudyPlanningPolicy {
            backend: runmat_analysis_fea::ComputeBackend::Cpu,
            numerics: StructuralNumericalPlanningPolicy::production(),
        }),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        memory_per_worker_bytes: None,
        accelerator_admission: Default::default(),
        analysis_accelerator_requests: Vec::new(),
        input_materializations: Vec::new(),
        initial_fields,
        geometry_meshing: geometry_meshing.clone(),
    };
    assert!(prepare_structural_study_root(preparation(Vec::new())).is_err());
    let prepared = prepare_structural_study_root(preparation(fields.to_vec())).unwrap();
    let mut store =
        crate::FilesystemObjectStore::open(temporary.path().join("transport"), 64 << 20).unwrap();
    prepared.write_to(&mut store).unwrap();
    let decoded = decode_structural_study_root(prepared.request(), &store).unwrap();
    assert_eq!(decoded.initial_fields, fields);
    assert_eq!(prepared.result_expectation().initial_fields, fields);
    assert_eq!(
        prepared.transport_objects().unwrap().len(),
        prepared.objects().len()
    );
    let mut missing_expectation = prepared.result_expectation();
    missing_expectation.initial_fields.clear();
    assert!(missing_expectation.validate().is_err());

    let source_inputs = super::super::initial_fields::prepare(&study, fields, &access).unwrap();
    let mut understated = source_inputs.inputs.clone();
    understated[0]
        .resources
        .decoded_resident_bytes_admission_budget
        .0 -= 1;
    assert!(super::super::initial_fields::decode(
        &study,
        &understated,
        &source_inputs.arguments,
        &access,
        &store
    )
    .is_err());
    let mut missing_argument = source_inputs.arguments.clone();
    missing_argument.pop();
    assert!(super::super::initial_fields::decode(
        &study,
        &source_inputs.inputs,
        &missing_argument,
        &access,
        &store
    )
    .is_err());
    let mut duplicate = source_inputs.arguments.clone();
    duplicate.push(duplicate[0].clone());
    assert!(super::super::initial_fields::decode(
        &study,
        &source_inputs.inputs,
        &duplicate,
        &access,
        &store
    )
    .is_err());

    let rebound = prepared
        .rebind_artifact_access(ArtifactAccess {
            authorization_scope: "rebound-original-fields".into(),
            encryption_context: Digest::sha256("rebound-original-fields-key"),
        })
        .unwrap();
    rebound.write_to(&mut store).unwrap();
    assert_eq!(
        decode_structural_study_root(rebound.request(), &store)
            .unwrap()
            .initial_fields,
        fields
    );
    let mut capped = study;
    capped.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1);
    assert!(super::super::initial_fields::prepare(&capped, fields, &access).is_err());
}
