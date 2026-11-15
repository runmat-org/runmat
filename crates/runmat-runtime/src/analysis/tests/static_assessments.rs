//! A static study's assessment names the run field it reads, and the result view lists that
//! field first.
use super::*;
use runmat_analysis_core::*;

fn element_peak(step_id: &str, geometry_id: &str, output_id: &str) -> AuthoredOutput {
    AuthoredOutput {
        output_id: output_id.into(),
        expression: AuthoredOutputExpression::SpatialReduction {
            reduction: AuthoredSpatialReduction::Maximum,
            source: Box::new(AuthoredOutputExpression::AnalysisField {
                step_id: step_id.into(),
                quantity: ObservationQuantity::VonMisesStress,
                location: FieldLocation::Element,
            }),
            target: region_target(geometry_id),
            ties: AuthoredExtremaTiePolicy::FirstCanonical,
        },
        policy: Default::default(),
    }
}

fn displacement_peak(step_id: &str, geometry_id: &str) -> AuthoredOutput {
    AuthoredOutput {
        output_id: "peak_displacement".into(),
        expression: AuthoredOutputExpression::SpatialReduction {
            reduction: AuthoredSpatialReduction::Maximum,
            source: Box::new(AuthoredOutputExpression::Projection {
                projection: AuthoredFieldProjection::Magnitude,
                source: Box::new(AuthoredOutputExpression::AnalysisField {
                    step_id: step_id.into(),
                    quantity: ObservationQuantity::Displacement,
                    location: FieldLocation::Node,
                }),
            }),
            target: region_target(geometry_id),
            ties: AuthoredExtremaTiePolicy::FirstCanonical,
        },
        policy: Default::default(),
    }
}

fn region_target(geometry_id: &str) -> AuthoredRegionSource {
    AuthoredRegionSource::Geometry(runmat_geometry_core::GeometryRegionRef {
        module_path: "study.fea".into(),
        geometry_id: geometry_id.into(),
        region_id: runmat_meshing_core::fixtures::fixture_region("region"),
        role: runmat_geometry_core::GeometryRegionRole::Solid,
    })
}

#[test]
fn static_peak_stress_evidence_names_the_element_von_mises_field_and_leads_the_field_list() {
    let _guard = analysis_test_guard();
    let (_mesh_root, mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("static-assessment-source-field");
    let mesh = SolverMeshArtifact::canonical_decode(&fs::read(&mesh_path).unwrap()).unwrap();
    let (mut compiled, mut publication, _) = assessment::tests::fixture();
    let mut model = sample_solid_model();
    let step_id = model.sequence.steps[0].step_id.0.clone();
    // The displacement output is declared first, so only the assessment can move von Mises
    // ahead of it.
    compiled.outputs = vec![
        displacement_peak(&step_id, &model.geometry_id),
        element_peak(&step_id, &model.geometry_id, "peak_stress"),
    ];
    compiled.assessments = vec![AuthoredAssessment {
        assessment_id: "stress".into(),
        output_id: "peak_stress".into(),
        rule: AssessmentRule::Maximum {
            bound: AssessmentQuantity::try_from_descriptor(
                (runmat_types::PhysicalDimension::PRESSURE).into(),
                1.0e12,
            )
            .expect("checked finite assessment quantity"),
        },
        evidence: AssessmentEvidenceRequirement::AuthoredSamples,
    }];
    attach_authored_outputs(&compiled.outputs, &mut model.sequence).unwrap();
    compiled.model = model.clone();
    compiled.solver_mesh_digest = mesh.canonical_digest;
    compiled.properties =
        runmat_analysis_fea::properties::compile_property_map(&model, Some(&mesh)).unwrap();
    publication.model_id = model.model_id.0.clone();
    publication.sequence_identity =
        runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(&model.sequence)
            .unwrap();
    publication.study_fingerprint = compiled.definition_digest().unwrap().to_string();
    let options = AnalysisRunOptions {
        deterministic_mode: true,
        precision_mode: PrecisionMode::Fp64,
        preconditioner_mode: PreconditionerMode::Auto,
        quality_policy: QualityPolicy::Balanced,
        solver_mesh_artifact_path: Some(mesh_path.display().to_string()),
    };
    let completed = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        options.clone(),
        OperationContext::new(None, None),
    )
    .unwrap();
    let run = completed.data;
    assert!(run.numerically_acceptable);
    // A solid tet model has no rotational DOFs: no rotation or moment field, and no field
    // without entries.
    assert!(run.run.field(FEA_FIELD_STRUCTURAL_ROTATION).is_none());
    assert!(run
        .run
        .field(FEA_FIELD_STRUCTURAL_REACTION_MOMENT)
        .is_none());
    assert!(run.run.fields.iter().all(|field| !field.is_empty()));
    publication.steps = vec![AnalysisStudyStepRunData {
        ordinal: 0,
        procedure_step: model.sequence.steps[0].clone(),
        execution: AnalysisExecutionStep {
            step_id: model.sequence.steps[0].step_id.clone(),
            backend: ComputeBackend::Cpu,
            accelerator_admission: Default::default(),
            solver_options: AnalysisSolverOptions::StructuralLinear(options),
        },
        identity: run.run.identity.clone(),
        run_operation: completed.operation,
        run_op_version: completed.op_version,
        run_id: run.run_id.clone(),
        run_status: run.run_status,
        numerically_acceptable: run.numerically_acceptable,
        solver_convergence: run.solver_convergence,
        result_quality: run.result_quality,
        quality_reasons: run.quality_reasons.clone(),
        provenance: run.provenance.clone(),
        numerical_result: AnalysisStudyStepNumericalResultData::Standard,
        evidence_artifact_path: String::new(),
        render_topology: None,
    }];
    let store: Arc<dyn storage::AnalysisArtifactStore> =
        Arc::new(storage::InMemoryAnalysisArtifactStore::new());
    store.persist_run(&run).unwrap();
    storage::set_artifact_store_for_tests(store);

    let assessed = assessment::assess_study_results(&compiled, &publication).unwrap();
    let evidence = assessed[0].evidence.as_ref().unwrap();
    assert_eq!(
        evidence.source_field_id.as_deref(),
        Some(FEA_FIELD_STRUCTURAL_VON_MISES)
    );
    assert!(evidence.entity.is_some());
    // Publication states the element peak against its limit; the study has no display, so the
    // line reads in Pa.
    let line = assessed[0].summary_line.as_deref().unwrap();
    assert!(
        line.starts_with("Peak ") && line.contains(" Pa (element) · limit 1e12 Pa · factor "),
        "{line}"
    );

    // The run lists declared outputs' fields first, in output order.
    let results = analysis_results_by_run_id_op(
        &run.run_id,
        AnalysisResultsQuery::metadata_only(),
        OperationContext::new(None, None),
    )
    .unwrap()
    .data;
    let run_order = results
        .field_descriptors
        .iter()
        .map(|descriptor| descriptor.field_id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        &run_order[..2],
        &[
            FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            FEA_FIELD_STRUCTURAL_VON_MISES
        ]
    );

    // The study result puts the assessed field first.
    publication.assessments = assessed;
    let document =
        analysis_document_study_result("study.fea".into(), publication, Vec::new()).unwrap();
    let study_order = document
        .field_bindings
        .iter()
        .map(|binding| binding.descriptor.field_id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        &study_order[..2],
        &[
            FEA_FIELD_STRUCTURAL_VON_MISES,
            FEA_FIELD_STRUCTURAL_DISPLACEMENT
        ]
    );
    assert_eq!(study_order.len(), run_order.len());
}

/// A field output names its source field only on its observation, which a stored run keeps
/// outside its metadata. The stored inventory carries the result order, so a metadata read and
/// a full read of the published run list the same descriptors in the same order.
#[test]
fn stored_run_lists_field_outputs_first_in_metadata_and_full_reads() {
    let _guard = analysis_test_guard();
    let (_mesh_root, mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("static-field-output-order");
    let mut model = sample_solid_model();
    let step_id = model.sequence.steps[0].step_id.0.clone();
    // The field output is declared ahead of the scalar one, so only the field output's
    // observation can put von Mises first.
    let outputs = vec![
        AuthoredOutput {
            output_id: "stress".into(),
            expression: AuthoredOutputExpression::AnalysisField {
                step_id: step_id.clone(),
                quantity: ObservationQuantity::VonMisesStress,
                location: FieldLocation::Element,
            },
            policy: Default::default(),
        },
        displacement_peak(&step_id, &model.geometry_id),
    ];
    attach_authored_outputs(&outputs, &mut model.sequence).unwrap();
    let run = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Balanced,
            solver_mesh_artifact_path: Some(mesh_path.display().to_string()),
        },
        OperationContext::new(None, None),
    )
    .unwrap()
    .data;
    assert!(run.run.observations.iter().any(|observation| {
        observation.source_field_id.as_deref() == Some(FEA_FIELD_STRUCTURAL_VON_MISES)
    }));
    let live = analysis_results_op(
        &run,
        AnalysisResultsQuery::metadata_only(),
        OperationContext::new(None, None),
    )
    .unwrap()
    .data
    .field_descriptors;
    let store: Arc<dyn storage::AnalysisArtifactStore> =
        Arc::new(storage::InMemoryAnalysisArtifactStore::new());
    store.persist_run(&run).unwrap();
    storage::set_artifact_store_for_tests(store);

    let read = |query| {
        analysis_results_by_run_id_op(&run.run_id, query, OperationContext::new(None, None))
            .unwrap()
            .data
            .field_descriptors
            .into_iter()
            .map(|descriptor| descriptor.field_id)
            .collect::<Vec<_>>()
    };
    let metadata = read(AnalysisResultsQuery::metadata_only());
    let full = read(AnalysisResultsQuery::default());
    assert_eq!(
        &metadata[..2],
        &[
            FEA_FIELD_STRUCTURAL_VON_MISES,
            FEA_FIELD_STRUCTURAL_DISPLACEMENT
        ]
    );
    assert_eq!(metadata, full);
    assert_eq!(
        live.iter()
            .map(|descriptor| descriptor.field_id.as_str())
            .collect::<Vec<_>>(),
        metadata
    );
}
