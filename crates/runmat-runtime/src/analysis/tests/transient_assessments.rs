//! Numerical sampled-peak evidence must survive cold store reads and revised criteria.
use super::*;
use runmat_analysis_core::*;
use runmat_analysis_fea::results::field::RecoveredFieldSample;

#[test]
fn transient_assessment_preserves_sampled_peak_and_store_parity() {
    let _guard = analysis_test_guard();
    let (mesh_root, mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("transient-assessment-evidence");
    let mesh = SolverMeshArtifact::canonical_decode(&fs::read(&mesh_path).unwrap()).unwrap();
    let (mut compiled, mut publication, mut criterion) = assessment::tests::fixture();
    let mut model = sample_solid_model();
    let mut step = ProcedureStep::structural_transient("response");
    step.independent_domain = IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: 0.1,
        interval_count: 10,
    };
    model.sequence = AnalysisSequence::from_steps("assessment-transient", vec![step]);
    let spatial = AuthoredOutput {
        output_id: "spatial_peak".into(),
        expression: AuthoredOutputExpression::SpatialReduction {
            reduction: AuthoredSpatialReduction::Maximum,
            source: Box::new(AuthoredOutputExpression::Projection {
                projection: AuthoredFieldProjection::Magnitude,
                source: Box::new(AuthoredOutputExpression::AnalysisField {
                    step_id: "response".into(),
                    quantity: ObservationQuantity::Displacement,
                    location: FieldLocation::Node,
                }),
            }),
            target: AuthoredRegionSource::Geometry(runmat_geometry_core::GeometryRegionRef {
                module_path: "study.fea".into(),
                geometry_id: model.geometry_id.clone(),
                region_id: runmat_meshing_core::fixtures::fixture_region("region"),
                role: runmat_geometry_core::GeometryRegionRole::Solid,
            }),
            ties: AuthoredExtremaTiePolicy::FirstCanonical,
        },
        policy: Default::default(),
    };
    let history = AuthoredOutput {
        output_id: "history".into(),
        expression: AuthoredOutputExpression::Sampling {
            source: Box::new(AuthoredOutputExpression::OutputReference {
                output_id: "spatial_peak".into(),
            }),
            sampling: SamplingPolicy::TimeSeconds((0..=10).map(|i| i as f64 / 100.0).collect()),
            domain: AuthoredSeriesDomain::Time,
        },
        policy: Default::default(),
    };
    let peak = AuthoredOutput {
        output_id: "peak".into(),
        expression: AuthoredOutputExpression::SeriesReduction {
            source: Box::new(AuthoredOutputExpression::OutputReference {
                output_id: "history".into(),
            }),
            reduction: AuthoredSeriesReduction::Maximum,
            domain: AuthoredSeriesDomain::Time,
            ties: AuthoredExtremaTiePolicy::FirstCanonical,
        },
        policy: Default::default(),
    };
    compiled.outputs = vec![spatial, history, peak];
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
    let options = AnalysisTransientRunOptions {
        deterministic_mode: true,
        solver_mesh_artifact_path: Some(mesh_path.display().to_string()),
        residual_target: 1e-10,
        ..Default::default()
    };
    let context = OperationContext::new(None, None);
    let completed = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        options.clone(),
        context,
    )
    .unwrap();
    let run = completed.data;
    assert!(run.numerically_acceptable);
    let snapshots = &run
        .transient_results
        .as_ref()
        .unwrap()
        .displacement_snapshots;
    let independent_peak = snapshots
        .iter()
        .flat_map(|field| field.as_host_f64().unwrap().chunks_exact(3))
        .map(|v| v.iter().map(|x| x * x).sum::<f64>().sqrt())
        .fold(0.0_f64, f64::max);
    assert!(independent_peak > 0.0 && independent_peak < 0.001);
    publication.steps = vec![AnalysisStudyStepRunData {
        ordinal: 0,
        procedure_step: model.sequence.steps[0].clone(),
        execution: AnalysisExecutionStep {
            step_id: ProcedureStepId("response".into()),
            backend: ComputeBackend::Cpu,
            accelerator_admission: Default::default(),
            solver_options: AnalysisSolverOptions::StructuralTransient(options),
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
    }];
    let directory = tempfile::tempdir().unwrap();
    let stores: Vec<Arc<dyn storage::AnalysisArtifactStore>> = vec![
        Arc::new(storage::InMemoryAnalysisArtifactStore::new()),
        Arc::new(storage::FilesystemAnalysisArtifactStore::new(
            directory.path().into(),
        )),
    ];
    let mut reports = Vec::new();
    for store in stores {
        store.persist_run(&run).unwrap();
        let restored = store.load_run(&run.run_id).unwrap().unwrap();
        assert_eq!(
            restored.run.observation_histories,
            run.run.observation_histories
        );
        assert_eq!(restored.run.series_reductions, run.run.series_reductions);
        storage::set_artifact_store_for_tests(store);
        let assessed = assessment::assess_study_results(&compiled, &publication).unwrap();
        assert_eq!(assessed[0].result.status, AssessmentStatus::Passed);
        let evidence = assessed[0].evidence.as_ref().unwrap();
        assert!((evidence.value - independent_peak).abs() < 1e-14);
        assert!(
            matches!(evidence.sample, Some(RecoveredFieldSample::TimeSeconds(t)) if (0.0..=0.1).contains(&t))
        );
        assert!(evidence.entity.is_some());
        reports.push(assessed);
        criterion.evidence = AssessmentEvidenceRequirement::AuthoredSamples;
        criterion.rule = AssessmentRule::Maximum {
            bound: AssessmentQuantity::try_from_descriptor(
                (runmat_types::PhysicalDimension::LENGTH).into(),
                independent_peak * 0.5,
            )
            .expect("checked finite assessment quantity"),
        };
        assert_eq!(
            assessment::reassess_study_results(&compiled, &publication, &[criterion.clone()])
                .unwrap()[0]
                .result
                .status,
            AssessmentStatus::Failed
        );
        criterion.evidence = AssessmentEvidenceRequirement::CertifiedBound;
        assert_eq!(
            assessment::reassess_study_results(&compiled, &publication, &[criterion.clone()])
                .unwrap()[0]
                .result
                .status,
            AssessmentStatus::Indeterminate
        );
    }
    assert_eq!(reports[0], reports[1]);
    storage::reset_artifact_store_for_tests();
    fs::remove_dir_all(mesh_root).unwrap();
}
