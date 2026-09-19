use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use tempfile::TempDir;

fn runmat(temp: &TempDir, args: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_runmat"));
    command.current_dir(temp.path()).args(args);
    command
        .env_remove("RUNMAT_CONFIG")
        .env(
            "RUNMAT_PACKAGE_CACHE_DIR",
            temp.path().join("package-cache"),
        )
        .env(
            "RUNMAT_EXECUTION_STATE_DIR",
            temp.path().join("execution-state"),
        );
    command.output().expect("run runmat check")
}

fn runmat_with_env(temp: &TempDir, args: &[&str], env: &[(&str, &str)]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_runmat"));
    command.current_dir(temp.path()).args(args);
    command
        .env_remove("RUNMAT_CONFIG")
        .env_remove("NO_COLOR")
        .env_remove("CLICOLOR")
        .env_remove("CLICOLOR_FORCE")
        .env_remove("FORCE_COLOR")
        .env_remove("TERM");
    for (key, value) in env {
        command.env(key, value);
    }
    command.output().expect("run runmat check")
}

fn runmat_from_repository(temp: &TempDir, args: &[&str]) -> Output {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let config = temp.path().join("runmat.toml");
    fs::write(
        &config,
        "[runtime.accelerate]\nenabled = false\nprovider = \"inprocess\"\n",
    )
    .unwrap();
    Command::new(env!("CARGO_BIN_EXE_runmat"))
        .current_dir(repository)
        .args(args)
        .env("RUNMAT_CONFIG", config)
        .env("RUNMAT_FEA_ARTIFACT_ROOT", temp.path().join("artifacts"))
        .env(
            "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
            temp.path().join("study-artifacts"),
        )
        .output()
        .expect("run repository FEA example")
}

fn assert_fea_checks(temp: &TempDir, path: &str) {
    let output = runmat_from_repository(temp, &["check", path, "--json"]);
    assert!(
        output.status.success(),
        "{path} failed check:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["kind"], "fea", "{path}");
    assert_eq!(payload["status"], "valid", "{path}");
    assert_eq!(payload["diagnostics"], serde_json::json!([]), "{path}");
}

#[test]
fn canonical_fea_documentation_examples_check_cleanly() {
    let temp = TempDir::new().unwrap();
    for name in [
        "structural-static-experimental.fea",
        "structural-transient-experimental.fea",
        "modal-response-spectrum-experimental.fea",
        "modal-random-response-experimental.fea",
        "acoustic-production-rejected.fea",
    ] {
        let path = format!("docs/fea/examples/{name}");
        assert_fea_checks(&temp, &path);
    }
}

#[test]
fn response_spectrum_documentation_example_runs_its_two_step_study() {
    let temp = TempDir::new().unwrap();
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = "docs/fea/examples/modal-response-spectrum-experimental.fea";
    let output = runmat_from_repository(&temp, &["run", path, "--json"]);
    assert!(
        output.status.success(),
        "{path} failed run:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["study_id"], "modal_response_spectrum_experimental");
    let steps = payload["steps"].as_array().expect("study steps");
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0]["procedure_step"]["procedure"], "modal_extraction");
    assert_eq!(
        steps[1]["procedure_step"]["procedure"],
        "modal_response_spectrum"
    );
    assert_eq!(
        steps[1]["numerical_result"]["kind"],
        "structural_response_spectrum"
    );
    assert_eq!(
        steps[1]["numerical_result"]["data"]["accuracy"]["evaluation"]["report"]["status"],
        "satisfied"
    );
    let states = payload["states"].as_array().expect("published states");
    assert!(states.iter().any(|state| state["kind"] == "modal_basis"));
    assert!(states
        .iter()
        .any(|state| state["kind"] == "structural_response_envelope"));

    let study_path = repository.join(path);
    let script_path = temp.path().join("response-spectrum-field.m");
    fs::write(
        &script_path,
        format!(
            "study = fea.load(\"{}\");\nrun = fea.run(study);\nresponse = run.step(\"response_spectrum\");\nfield = response.field(fea.ResponseSpectrumField.absoluteAcceleration());\npage = field.page(1, 3);\ndisp(field.field_id);\ndisp(field.statistic);\ndisp(page.count);\n",
            study_path.display()
        ),
    )
    .unwrap();
    let output = runmat_from_repository(
        &temp,
        &["run", script_path.to_str().expect("UTF-8 script path")],
    );
    assert!(
        output.status.success(),
        "response-spectrum language field smoke failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("structural.response_spectrum.nodal.absolute_acceleration"));
    assert!(stdout.contains("component_magnitude_envelope"));
    assert!(stdout.contains("3"));
}

#[test]
fn named_response_spectrum_generator_materializes_before_the_solve() {
    let temp = TempDir::new().unwrap();
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    fs::create_dir_all(temp.path().join("src/+spectra")).unwrap();
    fs::write(
        temp.path().join("runmat.toml"),
        "[package]\nname = \"fea-spectrum-generator\"\n\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    fs::write(
        temp.path().join("src/+spectra/design_input.m"),
        "function y = design_input(frequency, scale)\narguments\n    frequency (:,1) double\n    scale (1,1) double\nend\ny = scale .* (1 + 0 .* frequency);\nend\n",
    )
    .unwrap();
    let source = repository.join("docs/fea/examples/modal-response-spectrum-experimental.fea");
    let geometry = repository.join("crates/runmat-geometry/io/tests/fixtures/box.brep");
    let document = fs::read_to_string(source)
        .unwrap()
        .replace(
            "../../../crates/runmat-geometry/io/tests/fixtures/box.brep",
            &geometry.display().to_string(),
        )
        .replace(
            "            authored:\n              values: [1.2, 2.5, 1.8, 0.7]\n              provenance_digest: sha256:documented-design-spectrum",
            "            generator:\n              callable:\n                project_function: { qualified_name: spectra.design_input }\n              arguments:\n                - real: 1.5\n              effects: pure",
        );
    fs::write(temp.path().join("study.fea"), document).unwrap();

    let checked = runmat(&temp, &["check", "study.fea", "--json"]);
    assert!(
        checked.status.success(),
        "generated response spectrum failed check:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&checked.stdout),
        String::from_utf8_lossy(&checked.stderr)
    );

    let run = runmat(&temp, &["run", "study.fea", "--json"]);
    assert!(
        run.status.success(),
        "generated response spectrum failed run:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&run.stdout).unwrap()["study_id"],
        "modal_response_spectrum_experimental"
    );
    assert!(
        fs::read_dir(temp.path().join("artifacts/generated-inputs/objects"))
            .unwrap()
            .count()
            >= 2,
        "expected a numerical-input manifest and data chunk"
    );
}

#[test]
fn random_response_documentation_example_exposes_paged_statistical_fields() {
    let temp = TempDir::new().unwrap();
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let study_path = repository.join("docs/fea/examples/modal-random-response-experimental.fea");
    let script_path = temp.path().join("random-response-fields.m");
    fs::write(
        &script_path,
        format!(
            "study = fea.load(\"{}\");\nrun = fea.run(study);\nresponse = run.step(\"random_response\");\nrms = response.field(fea.RandomResponseField.rms(fea.StructuralField.displacement()));\nrmsPage = rms.page(1, 3);\nvariance = response.field(fea.RandomResponseField.variance(fea.StructuralField.stress()));\nvariancePage = variance.page(1, 3);\ndisp(rms.field_id);\ndisp(rms.statistic);\ndisp(rmsPage.count);\ndisp(variance.field_id);\ndisp(variance.statistic);\ndisp(variancePage.count);\n",
            study_path.display()
        ),
    )
    .unwrap();
    let output = runmat_from_repository(
        &temp,
        &["run", script_path.to_str().expect("UTF-8 script path")],
    );
    assert!(
        output.status.success(),
        "random-response language field smoke failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("structural.random_response.rms.displacement"));
    assert!(stdout.contains("root_mean_square"));
    assert!(stdout.contains("structural.random_response.variance.stress"));
    assert!(stdout.contains("variance"));
}

#[test]
fn fea_check_and_run_materialize_a_named_project_signal_generator() {
    let temp = TempDir::new().unwrap();
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    fs::create_dir_all(temp.path().join("src/+loads")).unwrap();
    fs::write(
        temp.path().join("runmat.toml"),
        "[package]\nname = \"fea-callable-check\"\n\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    fs::write(
        temp.path().join("src/+loads/pulse.m"),
        "function y = pulse(t, magnitude)\narguments\n    t (:,1) double\n    magnitude (1,1) double\nend\ny = magnitude .* (1 + 0 .* t);\nend\n",
    )
    .unwrap();
    let fixture = repository.join("crates/runmat-cli/tests/fixtures/box_static.fea");
    let geometry = repository.join("crates/runmat-geometry/io/tests/fixtures/box.brep");
    let document = fs::read_to_string(fixture)
        .unwrap()
        .replace(
            "../../../runmat-geometry/io/tests/fixtures/box.brep",
            &geometry.display().to_string(),
        )
        .replace(
            "        constant:\n          scalar: { newtons: 1000.0 }",
            "        generator:\n          callable:\n            project_function: { qualified_name: loads.pulse }\n          arguments:\n            - real: 1000.0\n          domain: time_seconds\n          sampling:\n            uniform: { start: 0.0, end: 1.0, sample_count: 5 }\n          interpolation: linear\n          extrapolation: reject\n          effects: pure",
        )
        .replace("        sample_count: 1", "        sample_count: 0");
    fs::write(temp.path().join("study.fea"), document).unwrap();

    let output = runmat(&temp, &["check", "study.fea", "--json"]);
    assert!(
        output.status.success(),
        "callable-backed study failed check:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["validation"]["valid"], true);

    let output = runmat(&temp, &["run", "study.fea", "--json"]);
    assert!(
        output.status.success(),
        "callable-backed study failed run:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["study_id"], "canonical_box_static");
    let objects = fs::read_dir(temp.path().join("artifacts/generated-inputs/objects"))
        .unwrap()
        .count();
    assert!(objects >= 2, "expected a signal manifest and data chunk");

    let repeated = runmat(&temp, &["run", "study.fea", "--json"]);
    assert!(
        repeated.status.success(),
        "repeated callable-backed study failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&repeated.stdout),
        String::from_utf8_lossy(&repeated.stderr)
    );
    assert_eq!(
        fs::read_dir(temp.path().join("artifacts/generated-inputs/objects"))
            .unwrap()
            .count(),
        objects,
        "repeated materialization must reuse the content-addressed object inventory"
    );
}

#[test]
fn production_example_rejects_an_unqualified_formulation() {
    let temp = TempDir::new().unwrap();
    let path = "docs/fea/examples/acoustic-production-rejected.fea";
    let output = runmat_from_repository(&temp, &["run", path, "--json"]);

    assert!(
        !output.status.success(),
        "{path} must reject its explicit Production requirement"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("RM.FEA.RUN_ACOUSTIC.FORMULATION_NOT_ADMITTED"),
        "unexpected rejection for {path}:\nstdout={}\nstderr={stderr}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[test]
fn unresolved_function_is_a_warning_by_default_and_an_error_when_denied() {
    let temp = TempDir::new().unwrap();
    fs::write(
        temp.path().join("main.m"),
        "value = definitely_missing(1);\n",
    )
    .unwrap();

    let default = runmat(&temp, &["check", "main.m"]);
    assert!(default.status.success(), "{default:?}");
    let stdout = String::from_utf8(default.stdout).unwrap();
    assert!(stdout.contains("warning[RM-RES0001]: cannot find function `definitely_missing`"));
    assert!(stdout.contains("checked main.m: 0 error(s), 1 warning(s)"));

    let denied = runmat(&temp, &["check", "main.m", "-D", "warnings"]);
    assert!(!denied.status.success(), "{denied:?}");
    let stdout = String::from_utf8(denied.stdout).unwrap();
    assert!(stdout.contains("warning[RM-RES0001]"));
}

#[test]
fn json_output_is_stable_and_contains_source_coordinates() {
    let temp = TempDir::new().unwrap();
    fs::write(
        temp.path().join("runmat.toml"),
        "[package]\nname = \"check-json\"\n\n[sources]\nroots = [\".\"]\n",
    )
    .unwrap();
    fs::write(
        temp.path().join("main.m"),
        "value = definitely_missing(1);\n",
    )
    .unwrap();

    let output = runmat(&temp, &["check", "main.m", "--json"]);
    assert!(output.status.success(), "{output:?}");
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["schema_version"], 1);
    assert_eq!(payload["outcome"], "warnings");
    assert_eq!(payload["summary"]["warnings"], 1);
    assert_eq!(payload["diagnostics"][0]["code"], "RM-RES0001");
    assert_eq!(payload["diagnostics"][0]["severity"], "warning");
    assert_eq!(payload["diagnostics"][0]["primary"]["line"], 1);
    assert_eq!(payload["diagnostics"][0]["primary"]["column"], 9);
    assert_eq!(payload["analysis"]["name_resolution"], "complete");
    assert!(payload["project_revision"]["graph_digest"]
        .as_str()
        .is_some_and(|digest| digest.starts_with("sha256:")));
    assert!(payload["project_revision"]["source_revision"]
        .as_str()
        .is_some_and(|digest| digest.starts_with("sha256:")));
}

#[test]
fn human_diagnostics_support_forced_and_plain_color_modes() {
    let temp = TempDir::new().unwrap();
    fs::write(
        temp.path().join("main.m"),
        "value = definitely_missing(1);\n",
    )
    .unwrap();

    let styled = runmat_with_env(&temp, &["--color=always", "check", "main.m"], &[]);
    assert!(styled.status.success(), "{styled:?}");
    assert!(
        styled.stdout.windows(2).any(|bytes| bytes == b"\x1b["),
        "forced human output did not contain ANSI styling: {}",
        String::from_utf8_lossy(&styled.stdout)
    );

    let plain = runmat_with_env(
        &temp,
        &["check", "main.m", "--color=never"],
        &[("FORCE_COLOR", "3")],
    );
    assert!(plain.status.success(), "{plain:?}");
    assert!(!plain.stdout.windows(2).any(|bytes| bytes == b"\x1b["));
    assert_eq!(strip_ansi(&styled.stdout), plain.stdout);
}

#[test]
fn no_color_is_nonempty_and_wins_over_environment_force() {
    let temp = TempDir::new().unwrap();
    fs::write(
        temp.path().join("main.m"),
        "value = definitely_missing(1);\n",
    )
    .unwrap();

    let disabled = runmat_with_env(
        &temp,
        &["check", "main.m"],
        &[("NO_COLOR", "0"), ("FORCE_COLOR", "3")],
    );
    assert!(disabled.status.success(), "{disabled:?}");
    assert!(!disabled.stdout.windows(2).any(|bytes| bytes == b"\x1b["));

    let empty_is_unset = runmat_with_env(
        &temp,
        &["check", "main.m"],
        &[("NO_COLOR", ""), ("FORCE_COLOR", "1")],
    );
    assert!(empty_is_unset.status.success(), "{empty_is_unset:?}");
    assert!(empty_is_unset
        .stdout
        .windows(2)
        .any(|bytes| bytes == b"\x1b["));
}

#[test]
fn json_remains_plain_even_when_human_color_is_forced() {
    let temp = TempDir::new().unwrap();
    fs::write(
        temp.path().join("main.m"),
        "value = definitely_missing(1);\n",
    )
    .unwrap();

    let output = runmat_with_env(&temp, &["--color=always", "check", "main.m", "--json"], &[]);
    assert!(output.status.success(), "{output:?}");
    assert!(!output.stdout.windows(2).any(|bytes| bytes == b"\x1b["));
    assert!(!output.stderr.windows(2).any(|bytes| bytes == b"\x1b["));
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["schema_version"], 1);
}

#[test]
fn fea_human_path_uses_the_shared_color_policy() {
    let temp = TempDir::new().unwrap();
    fs::write(temp.path().join("invalid.fea"), "{}\n").unwrap();

    let styled = runmat_with_env(&temp, &["--color=always", "check", "invalid.fea"], &[]);
    assert!(!styled.status.success(), "{styled:?}");
    assert!(
        styled.stderr.windows(2).any(|bytes| bytes == b"\x1b["),
        "FEA human output did not contain ANSI styling: {}",
        String::from_utf8_lossy(&styled.stderr)
    );
    assert!(String::from_utf8_lossy(&styled.stderr).contains("Checking"));

    let plain = runmat_with_env(&temp, &["--color=never", "check", "invalid.fea"], &[]);
    assert!(!plain.status.success(), "{plain:?}");
    assert!(!plain.stderr.windows(2).any(|bytes| bytes == b"\x1b["));
}

#[test]
fn loose_sibling_and_explicit_search_path_functions_resolve() {
    let temp = TempDir::new().unwrap();
    fs::write(temp.path().join("main.m"), "value = helper(1);\n").unwrap();
    fs::write(
        temp.path().join("helper.m"),
        "function y = helper(x)\ny = x;\nend\n",
    )
    .unwrap();
    let sibling = runmat(&temp, &["check", "main.m", "--json"]);
    assert!(sibling.status.success(), "{sibling:?}");
    let payload: Value = serde_json::from_slice(&sibling.stdout).unwrap();
    assert_eq!(payload["outcome"], "clean");
    assert_eq!(payload["resolution"][0]["state"], "resolved");
    assert_eq!(payload["resolution"][0]["name"], "helper");
    assert!(payload["resolution"][0]["definition"]["source_path"]
        .as_str()
        .expect("source path")
        .ends_with("helper.m"));

    fs::remove_file(temp.path().join("helper.m")).unwrap();
    fs::create_dir(temp.path().join("toolbox")).unwrap();
    fs::write(
        temp.path().join("toolbox/helper.m"),
        "function y = helper(x)\ny = x;\nend\n",
    )
    .unwrap();
    let without_path = runmat(&temp, &["check", "main.m", "--json"]);
    let payload: Value = serde_json::from_slice(&without_path.stdout).unwrap();
    assert_eq!(payload["outcome"], "warnings");

    let with_path = runmat(&temp, &["check", "main.m", "--path", "toolbox", "--json"]);
    assert!(with_path.status.success(), "{with_path:?}");
    let payload: Value = serde_json::from_slice(&with_path.stdout).unwrap();
    assert_eq!(payload["outcome"], "clean");
}

#[test]
fn runtime_path_mutation_reports_the_causal_site() {
    let temp = TempDir::new().unwrap();
    fs::write(
        temp.path().join("main.m"),
        "addpath('plugins');\nvalue = selected_later(1);\n",
    )
    .unwrap();

    let output = runmat(&temp, &["check", "main.m", "--json"]);
    assert!(output.status.success(), "{output:?}");
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    let diagnostic = payload["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .find(|diagnostic| diagnostic["code"] == "RM-RES0002")
        .expect("runtime-dependent resolution diagnostic");
    assert_eq!(diagnostic["secondary"][0]["line"], 1);
    assert_eq!(diagnostic["primary"]["line"], 2);
    assert_eq!(payload["analysis"]["name_resolution"], "runtime_dependent");
}

#[test]
fn syntax_and_source_catalog_failures_return_structured_json() {
    let syntax = TempDir::new().unwrap();
    fs::write(syntax.path().join("main.m"), "value = ;\n").unwrap();
    let output = runmat(&syntax, &["check", "main.m", "--json"]);
    assert!(!output.status.success(), "{output:?}");
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["outcome"], "failed");
    assert_eq!(payload["diagnostics"][0]["code"], "RunMat:ParseError");
    assert_eq!(payload["analysis"]["name_resolution"], "unavailable");

    let catalog = TempDir::new().unwrap();
    fs::write(catalog.path().join("main.m"), "value = 1;\n").unwrap();
    let output = runmat(
        &catalog,
        &["check", "main.m", "--path", "missing-source-root", "--json"],
    );
    assert!(!output.status.success(), "{output:?}");
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(payload["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .any(|diagnostic| diagnostic["code"] == "RM-CAT0001"));
}

fn strip_ansi(bytes: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == 0x1b && bytes.get(index + 1) == Some(&b'[') {
            index += 2;
            while index < bytes.len() {
                let byte = bytes[index];
                index += 1;
                if byte.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            output.push(bytes[index]);
            index += 1;
        }
    }
    output
}
