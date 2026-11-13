//! Authored observations, isolated imported criteria and immutable cold reassessment.
use super::*;
use runmat_analysis_core::AssessmentStatus;
use runmat_runtime::analysis::assessment::{PublishedStudyAssessment, PublishedStudyReassessment};

fn invoke(project: &TempDir, args: &[&str]) -> Value {
    let output = runmat(project, args);
    assert!(
        output.status.success(),
        "command={args:?}\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("typed command JSON")
}

#[test]
fn authored_transient_and_frequency_assessments_reopen_without_re_solving_the_numerical_run() {
    for (source, study) in [
        (
            include_str!("../fixtures/typed_box_transient_assessments.fea"),
            "structural_transient",
        ),
        (
            include_str!("../fixtures/typed_box_frequency_assessments.fea"),
            "structural_harmonic",
        ),
    ] {
        let project = TempDir::new().unwrap();
        fs::create_dir_all(project.path().join("src/geometry")).unwrap();
        fs::write(
            project.path().join("runmat.toml"),
            "[package]\nname = \"assessment-qualification\"\n[sources]\nroots = [\"src\"]\n",
        )
        .unwrap();
        fs::write(
            project.path().join("src/geometry/box.brep"),
            include_bytes!("../../../runmat-geometry/io/tests/fixtures/box.brep"),
        )
        .unwrap();
        fs::write(
            project.path().join("src/verify_peak.m"),
            include_str!("../fixtures/verify_peak.m"),
        )
        .unwrap();
        let source_path = project.path().join("src/main.fea");
        fs::write(&source_path, source).unwrap();
        let checked = invoke(
            &project,
            &[
                "--offline",
                "check",
                "src/main.fea",
                "--study",
                study,
                "--json",
            ],
        );
        assert_eq!(checked["status"], "valid");
        let run = invoke(
            &project,
            &[
                "--offline",
                "run",
                "src/main.fea",
                "--study",
                study,
                "--json",
            ],
        );
        let reports: Vec<PublishedStudyAssessment> =
            serde_json::from_value(run["assessments"].clone()).unwrap();
        assert_eq!(reports.len(), 2);
        assert!(reports
            .iter()
            .all(|report| report.result.status == AssessmentStatus::Passed));
        assert!(reports
            .iter()
            .any(|report| report.function_execution_identity.is_some()));
        let values: Vec<_> = reports
            .iter()
            .map(|report| report.evidence.as_ref().unwrap().value)
            .collect();
        assert!(
            values.iter().all(|value| *value > 0.0 && *value < 0.001),
            "{values:?}"
        );
        let manifest = run["result_manifest_artifact_path"].as_str().unwrap();
        let manifest_path = project.path().join(manifest);
        let original_bytes = fs::read(&manifest_path).unwrap();
        let definition = run["study_definition_digest"].as_str().unwrap();
        let reassess_args = [
            "--offline",
            "reassess",
            "src/main.fea",
            "--study",
            study,
            "--original-definition",
            definition,
            "--result-manifest",
            manifest,
            "--json",
        ];
        for (revised, expected) in [
            (
                source
                    .replace("maximum = 1 mm", "maximum = 0.000000000001 m")
                    .replace("parameters = [1 mm]", "parameters = [0.000000000001 m]"),
                AssessmentStatus::Failed,
            ),
            (
                source.replace("evidence = authored_samples", "evidence = certified_bound"),
                AssessmentStatus::Indeterminate,
            ),
        ] {
            fs::write(&source_path, revised).unwrap();
            // Each command starts a fresh process: it must reopen authenticated saved evidence.
            let report: PublishedStudyReassessment =
                serde_json::from_value(invoke(&project, &reassess_args)).unwrap();
            assert_eq!(report.assessments.len(), reports.len());
            assert_eq!(report.study_definition.to_string(), definition);
            assert!(report
                .assessments
                .iter()
                .all(|assessment| assessment.result.status == expected));
            assert_eq!(
                report
                    .assessments
                    .iter()
                    .map(|assessment| assessment.evidence.as_ref().unwrap().value)
                    .collect::<Vec<_>>(),
                values
            );
            assert_eq!(
                report
                    .assessments
                    .iter()
                    .map(|assessment| &assessment.run_id)
                    .collect::<Vec<_>>(),
                reports
                    .iter()
                    .map(|assessment| &assessment.run_id)
                    .collect::<Vec<_>>()
            );
            assert_eq!(fs::read(&manifest_path).unwrap(), original_bytes);
        }
    }
}
