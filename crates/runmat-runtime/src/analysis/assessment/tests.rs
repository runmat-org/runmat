use super::*;
use runmat_analysis_core::*;
use runmat_geometry_core::{GeometryRegionRef, GeometryRegionRole};

pub(in crate::analysis) fn fixture() -> (
    CompiledAnalysisStudy,
    AnalysisStudyRunData,
    AuthoredAssessment,
) {
    let model = crate::analysis::tests::sample_solid_model();
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(
        runmat_meshing_core::ElementOrder::Tet4,
    );
    let definition = AuthoredAssessment {
        assessment_id: "deflection".into(),
        output_id: "peak".into(),
        rule: AssessmentRule::Maximum {
            bound: AssessmentQuantity::try_from_descriptor(
                (runmat_types::PhysicalDimension::LENGTH).into(),
                0.001,
            )
            .expect("checked finite assessment quantity"),
        },
        evidence: AssessmentEvidenceRequirement::AuthoredSamples,
    };
    let output = AuthoredOutput {
        output_id: "peak".into(),
        policy: AuthoredOutputPolicy::default(),
        expression: AuthoredOutputExpression::SpatialReduction {
            reduction: AuthoredSpatialReduction::Maximum,
            source: Box::new(AuthoredOutputExpression::Projection {
                projection: AuthoredFieldProjection::Magnitude,
                source: Box::new(AuthoredOutputExpression::AnalysisField {
                    step_id: model.sequence.steps[0].step_id.0.clone(),
                    quantity: ObservationQuantity::Displacement,
                    location: FieldLocation::Node,
                }),
            }),
            target: AuthoredRegionSource::Geometry(GeometryRegionRef {
                module_path: "study.fea".into(),
                geometry_id: model.geometry_id.clone(),
                region_id: "region".into(),
                role: GeometryRegionRole::Solid,
            }),
            ties: AuthoredExtremaTiePolicy::FirstCanonical,
        },
    };
    let compiled = CompiledAnalysisStudy {
        module_path: "study.fea".into(),
        study_id: "study".into(),
        study_inputs: Vec::new(),
        outputs: vec![output],
        assessments: vec![definition.clone()],
        properties: runmat_analysis_fea::properties::compile_property_map(&model, Some(&mesh))
            .unwrap(),
        solver_mesh_digest: mesh.canonical_digest,
        model,
    };
    let published = AnalysisStudyRunData {
        assessments: Vec::new(),
        study_id: compiled.study_id.clone(),
        model_id: compiled.model.model_id.0.clone(),
        sequence_identity: runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
            &compiled.model.sequence,
        )
        .unwrap(),
        steps: Vec::new(),
        states: Vec::new(),
        solver_mesh_artifact_path: None,
        meshing_evidence_artifact_path: None,
        study_fingerprint: compiled.definition_digest().unwrap().to_string(),
        operation_sequence: Vec::new(),
        run_status: crate::analysis::RunStatus::Completed,
        numerically_acceptable: true,
        solver_convergence: crate::analysis::QualityGate::Pass,
        result_quality: crate::analysis::QualityGate::Pass,
        evidence_artifact_path: String::new(),
        result_manifest_artifact_path: None,
        diagnostics: Vec::new(),
    };
    (compiled, published, definition)
}

#[test]
fn checked_reassessment_rejects_changed_output_meaning_under_the_same_name() {
    let (compiled, _, mut criterion) = fixture();
    criterion.rule = AssessmentRule::Maximum {
        bound: AssessmentQuantity::try_from_descriptor(
            (runmat_types::PhysicalDimension::LENGTH).into(),
            0.002,
        )
        .expect("checked finite assessment quantity"),
    };
    let validate = |outputs: &[AuthoredOutput], id: &str| {
        validate_reassessment_outputs(
            &compiled,
            id,
            outputs,
            &compiled.model.sequence.steps,
            &[criterion.clone()],
        )
    };
    validate(&compiled.outputs, &compiled.study_id).unwrap();
    assert!(validate(&compiled.outputs, "different_study").is_err());
    let mut changed = compiled.outputs.clone();
    let AuthoredOutputExpression::SpatialReduction { target, .. } = &mut changed[0].expression
    else {
        unreachable!()
    };
    let AuthoredRegionSource::Geometry(region) = target else {
        unreachable!()
    };
    region.region_id = "different_region".into();
    assert!(validate(&changed, &compiled.study_id).is_err());
    let mut changed = compiled.outputs.clone();
    let AuthoredOutputExpression::SpatialReduction { reduction, .. } = &mut changed[0].expression
    else {
        unreachable!()
    };
    *reduction = AuthoredSpatialReduction::Minimum;
    assert!(validate(&changed, &compiled.study_id).is_err());
}

#[test]
fn reassessment_retains_new_criteria_and_original_identity_without_inventing_evidence() {
    let (compiled, published, mut criterion) = fixture();
    criterion.rule = AssessmentRule::Maximum {
        bound: AssessmentQuantity::try_from_descriptor(
            (runmat_types::PhysicalDimension::LENGTH).into(),
            0.002,
        )
        .expect("checked finite assessment quantity"),
    };
    let reports = reassess_study_results(&compiled, &published, &[criterion.clone()]).unwrap();
    assert_eq!(reports[0].definition, criterion);
    assert_eq!(
        reports[0].study_definition,
        compiled.definition_digest().unwrap()
    );
    assert_eq!(
        reports[0].result.issue,
        Some(AssessmentIssue::MissingOutput)
    );
    assert_eq!(reports[0].result.status, AssessmentStatus::Indeterminate);
    assert!(reports[0].evidence.is_none());
    assert!(reports[0].run_id.is_none());
    let mut replaced = compiled.clone();
    replaced.assessments = vec![criterion];
    assert!(reassess_study_results(&replaced, &published, &replaced.assessments).is_err());
}

#[test]
fn reassessment_publication_is_immutable_and_content_addressed() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let root = tempfile::tempdir().unwrap();
    let _config = crate::analysis::tests::scoped_study_artifact_root(root.path());
    let (compiled, published, mut criterion) = fixture();
    let first = publish_study_reassessment(
        &compiled,
        &published,
        &[criterion.clone()],
        AssessmentEvaluationLimits::default(),
        || false,
    )
    .unwrap();
    let repeated = publish_study_reassessment(
        &compiled,
        &published,
        &[criterion.clone()],
        AssessmentEvaluationLimits::default(),
        || false,
    )
    .unwrap();
    let original_bytes = std::fs::read(&first.evidence_artifact_path).unwrap();
    criterion.rule = AssessmentRule::Maximum {
        bound: AssessmentQuantity::try_from_descriptor(
            (runmat_types::PhysicalDimension::LENGTH).into(),
            0.002,
        )
        .expect("checked finite assessment quantity"),
    };
    let revised = publish_study_reassessment(
        &compiled,
        &published,
        &[criterion],
        AssessmentEvaluationLimits::default(),
        || false,
    )
    .unwrap();
    assert_eq!(first, repeated);
    assert_ne!(first.evidence_artifact_path, revised.evidence_artifact_path);
    assert_eq!(
        original_bytes,
        std::fs::read(&first.evidence_artifact_path).unwrap()
    );
    assert_eq!(first.study_definition, revised.study_definition);
    assert_ne!(
        first.assessments[0].definition,
        revised.assessments[0].definition
    );
}

#[test]
fn reassessment_rejects_wrong_dimensions_and_honors_limits_and_cancellation() {
    let (compiled, published, mut criterion) = fixture();
    assert!(matches!(
        reassess_study_results_with_limits(
            &compiled,
            &published,
            &[criterion.clone()],
            AssessmentEvaluationLimits {
                maximum_assessments: 0,
                maximum_runs: 1,
                ..AssessmentEvaluationLimits::default()
            },
            || false
        ),
        Err(AssessmentEvaluationError::ResourceLimit)
    ));
    assert!(matches!(
        reassess_study_results_with_limits(
            &compiled,
            &published,
            &[criterion.clone()],
            AssessmentEvaluationLimits::default(),
            || true
        ),
        Err(AssessmentEvaluationError::Cancelled)
    ));
    criterion.rule = AssessmentRule::Maximum {
        bound: AssessmentQuantity::try_from_descriptor(
            (runmat_types::PhysicalDimension::MASS).into(),
            1.0,
        )
        .expect("checked finite assessment quantity"),
    };
    assert!(matches!(
        reassess_study_results_with_limits(
            &compiled,
            &published,
            &[criterion],
            AssessmentEvaluationLimits::default(),
            || false
        ),
        Err(AssessmentEvaluationError::InvalidCriteria(_))
    ));
}
