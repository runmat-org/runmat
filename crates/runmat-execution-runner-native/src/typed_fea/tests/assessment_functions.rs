//! Real frozen-project callable and portable VM qualification; full solve/task placement is separate.
use runmat_analysis_core::{AssessmentFunctionOutcome, AssessmentRule};
use runmat_analysis_execution::assessment_program::{
    decode_assessment_outcome, FrozenAssessmentCallable,
};
use runmat_execution::Digest;
use runmat_execution_artifact::ProgramExecutionResponse;

const SOURCE: &str = r#"
import function verify from "./verify.m"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region body : solid { select result; expect exactly_one }
}
mesh part on geometry.part {
    tetrahedra { order = linear; target_size = 0.1 m; maximum_deviation = 0.01 m }
}
model structure {}
study response {
    geometry = geometry.part
    mesh = mesh.part
    model = model.structure
    analysis solve { structural static {} }
    output peak = max(magnitude(analysis.solve.field.node.displacement), over = geometry.part.region.body)
    assessment deflection {
        value = output.peak
        function = function.verify
        parameters = [1 mm]
        evidence = authored_samples
    }
}
"#;

#[tokio::test]
async fn frozen_imported_assessment_executes_portable_boolean_contract() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(directory.path().join("src")).unwrap();
    std::fs::write(
        directory.path().join("runmat.toml"),
        "[package]\nname = \"assessment-contract\"\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    std::fs::write(directory.path().join("src/verify.m"),
        "function result = verify(value, limit)\narguments\n value (1,1) double\n limit (1,1) double\nend\nresult = struct('evaluated', true, 'accepted', value <= limit);\nend\n").unwrap();
    let path = directory.path().join("src/study.fea");
    std::fs::write(&path, SOURCE).unwrap();
    let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    let project = runmat_package::build_frozen_project(
        &directory.path().join("runmat.toml"),
        std::collections::BTreeSet::new(),
    )
    .unwrap();
    session
        .install_project_handoff(runmat_package::FrozenProjectHandoff::new(project))
        .await
        .unwrap();
    let checked = session
        .load_and_check_fea_source_path(&path.canonicalize().unwrap())
        .await
        .unwrap();
    let report = checked.check_result();
    if report.status != runmat_runtime::analysis::FeaCheckStatus::Valid {
        let imported =
            super::callable::compile_imported_function(&checked, "src/verify.m", "verify", None)
                .await
                .unwrap();
        panic!(
            "assessment diagnostics: {:?}; actual callable facts: {:#?}",
            report.diagnostics, imported.function
        );
    }
    assert_eq!(
        report.status,
        runmat_runtime::analysis::FeaCheckStatus::Valid,
        "assessment source diagnostics: {:?}",
        report.diagnostics
    );
    let resolved = checked
        .resolve_root(Some(&runmat_runtime::analysis::FeaRootSelector::Study {
            id: "response".into(),
        }))
        .unwrap();
    let runmat_runtime::analysis::ResolvedFeaRoot::Study(study) = resolved else {
        panic!("expected study")
    };
    let assessment = &study.study.assessments[0];
    let AssessmentRule::Function { function, .. } = &assessment.rule else {
        panic!("expected function")
    };
    let imported = super::callable::compile_imported_function(
        &checked,
        &function.module_path,
        &function.symbol,
        None,
    )
    .await
    .unwrap();
    let frozen = FrozenAssessmentCallable::new(
        assessment.clone(),
        imported.revision.clone(),
        imported.callable.clone(),
        imported.executable.clone(),
    )
    .unwrap();
    let restored: FrozenAssessmentCallable =
        serde_json::from_slice(&serde_json::to_vec(&frozen).unwrap()).unwrap();
    restored.validate().unwrap();
    // A detached job must use the submitted function even after the local project changes.
    std::fs::write(directory.path().join("src/verify.m"), "function result = verify(value, limit)\nresult = struct('evaluated', true, 'accepted', false);\nend\n").unwrap();
    let mut substituted = assessment.clone();
    substituted.assessment_id = "another-criterion".into();
    assert!(restored
        .prepare(
            &substituted,
            0.0005,
            Digest::sha256("study"),
            Digest::sha256("immutable-output")
        )
        .is_err());
    for (value_si, accepted) in [(0.0005, true), (0.002, false)] {
        let prepared = restored
            .prepare(
                assessment,
                value_si,
                Digest::sha256("study"),
                Digest::sha256("immutable-output"),
            )
            .unwrap_or_else(|error| {
                panic!(
                    "assessment preparation failed: {error}; actual callable facts: {:#?}",
                    imported.function
                )
            });
        let response = crate::execute_host_program_request(prepared.request().clone()).await;
        response.validate_against(prepared.request()).unwrap();
        let ProgramExecutionResponse::Success { mut outputs } = response else {
            panic!("assessment execution failed: {response:?}")
        };
        assert_eq!(outputs.len(), 1, "one original semantic output");
        let value = outputs.remove(0);
        assert_eq!(
            decode_assessment_outcome(&value).unwrap_or_else(|error| panic!(
                "assessment return rejected: {error}; payload: {value:#?}"
            )),
            AssessmentFunctionOutcome {
                evaluated: true,
                accepted
            }
        );
    }
}
