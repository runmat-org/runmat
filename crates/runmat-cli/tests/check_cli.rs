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
    let artifact_root =
        toml::Value::String(temp.path().join("artifacts").to_string_lossy().into_owned());
    let study_artifact_root = toml::Value::String(
        temp.path()
            .join("study-artifacts")
            .to_string_lossy()
            .into_owned(),
    );
    fs::write(
        &config,
        format!(
            "[runtime.accelerate]\nenabled = false\nprovider = \"inprocess\"\n\n[runtime.fea]\nartifact_store = \"filesystem\"\nartifact_root = {artifact_root}\nstudy_artifact_root = {study_artifact_root}\n"
        ),
    )
    .unwrap();
    Command::new(env!("CARGO_BIN_EXE_runmat"))
        .current_dir(repository)
        .args(args)
        .env("RUNMAT_CONFIG", config)
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

fn fea_plan(temp: &TempDir, path: &str) -> Value {
    let output = runmat_from_repository(temp, &["fea", "plan", path, "--json"]);
    assert!(
        output.status.success(),
        "{path} failed plan:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn assert_repeated_result_identity(temp: &TempDir, path: &str, first: &Value) {
    let repeated = runmat_from_repository(temp, &["run", path, "--json"]);
    assert!(
        repeated.status.success(),
        "{path} failed repeated run:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&repeated.stdout),
        String::from_utf8_lossy(&repeated.stderr)
    );
    let repeated: Value = serde_json::from_slice(&repeated.stdout).unwrap();
    for field in [
        "study_definition_digest",
        "mesh_digest",
        "nodes",
        "elements",
        "boundary_faces",
        "step_run_ids",
    ] {
        assert_eq!(
            repeated[field], first[field],
            "{path} changed verified result field `{field}` on repeat"
        );
    }
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
fn representative_documentation_plans_expose_the_complete_resolved_contract() {
    struct Expected<'a> {
        file: &'a str,
        root: &'a str,
        model: &'a str,
        steps: &'a [(&'a str, &'a str)],
        outputs: &'a [&'a str],
        material_assignments: usize,
        response_spectra: usize,
        random_excitations: usize,
    }

    let cases = [
        Expected {
            file: "structural-static-experimental.fea",
            root: "structural_static_experimental",
            model: "loaded_bracket",
            steps: &[("static", "static")],
            outputs: &["displacement", "von_mises"],
            material_assignments: 2,
            response_spectra: 0,
            random_excitations: 0,
        },
        Expected {
            file: "structural-transient-experimental.fea",
            root: "structural_transient_experimental",
            model: "impulsive_bracket",
            steps: &[("response", "transient_response")],
            outputs: &["displacement"],
            material_assignments: 1,
            response_spectra: 0,
            random_excitations: 0,
        },
        Expected {
            file: "modal-response-spectrum-experimental.fea",
            root: "modal_response_spectrum_experimental",
            model: "launch_bracket",
            steps: &[
                ("modes", "modal_extraction"),
                ("response_spectrum", "modal_response_spectrum"),
            ],
            outputs: &["displacement"],
            material_assignments: 1,
            response_spectra: 1,
            random_excitations: 0,
        },
        Expected {
            file: "modal-random-response-experimental.fea",
            root: "modal_random_response_experimental",
            model: "random_environment",
            steps: &[
                ("modes", "modal_extraction"),
                ("random_response", "random_response"),
            ],
            outputs: &["displacement"],
            material_assignments: 1,
            response_spectra: 0,
            random_excitations: 1,
        },
        Expected {
            file: "acoustic-production-rejected.fea",
            root: "acoustic_production_rejected",
            model: "cavity",
            steps: &[("harmonic", "harmonic_response")],
            outputs: &["pressure"],
            material_assignments: 1,
            response_spectra: 0,
            random_excitations: 0,
        },
    ];
    let temp = TempDir::new().unwrap();
    for expected in cases {
        let path = format!("docs/fea/examples/{}", expected.file);
        let plan = fea_plan(&temp, &path);
        assert_eq!(plan["schema_version"], 1, "{path}");
        assert_eq!(plan["root_kind"], "study", "{path}");
        assert_eq!(plan["root_id"], expected.root, "{path}");
        assert!(plan["geometry_dependencies"].as_array().is_some(), "{path}");

        let models = plan["model_closure"].as_array().expect("model closure");
        assert_eq!(models.len(), 1, "{path}");
        let model = &models[0];
        assert_eq!(model["model_id"], expected.model, "{path}");
        assert_eq!(
            model["response_spectra"].as_array().unwrap().len(),
            expected.response_spectra,
            "{path}"
        );
        assert_eq!(
            model["random_excitations"].as_array().unwrap().len(),
            expected.random_excitations,
            "{path}"
        );
        assert_eq!(
            plan["material_catalog"].as_array().unwrap().len(),
            1,
            "{path}"
        );
        assert_eq!(
            model["material_assignments"].as_array().unwrap().len(),
            expected.material_assignments,
            "{path}"
        );

        let steps = plan["study"]["sequence"]["steps"]
            .as_array()
            .expect("procedure steps");
        assert_eq!(steps.len(), expected.steps.len(), "{path}");
        for (step, (step_id, procedure)) in steps.iter().zip(expected.steps) {
            assert_eq!(step["step_id"], *step_id, "{path}");
            assert_eq!(step["procedure"], *procedure, "{path}");
        }
        let outputs = plan["study"]["outputs"]
            .as_array()
            .expect("authored outputs")
            .iter()
            .map(|output| output["output_id"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(outputs, expected.outputs, "{path}");
        assert_eq!(
            plan["mesh"]["geometry"], plan["study"]["geometry"]["definition"],
            "{path}"
        );
    }
}

#[test]
fn structural_static_step_documentation_example_runs_and_publishes_fields() {
    let temp = TempDir::new().unwrap();
    let path = "docs/fea/examples/structural-static-experimental.fea";
    let output = runmat_from_repository(&temp, &["run", path, "--json"]);
    assert!(
        output.status.success(),
        "{path} failed run:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["root_id"], "structural_static_experimental");
    assert_eq!(payload["procedure"], "structural_linear_sequence");
    let step_run_id = payload["step_run_ids"]
        .as_array()
        .and_then(|steps| steps.first())
        .and_then(Value::as_str)
        .expect("static step run id");
    assert_repeated_result_identity(&temp, path, &payload);

    let script_path = temp.path().join("static-fields.m");
    fs::write(
        &script_path,
        format!(
            "results = fea.results(\"{step_run_id}\");\ndisplacement = fea.field(results, fea.OutputField.named(\"displacement\"));\ndisplacementPage = fea.fieldPage(displacement, 1, 3);\nstress = fea.field(results, fea.OutputField.named(\"von_mises\"));\nstressPage = fea.fieldPage(stress, 1, 3);\ndisp(displacement.field_id);\ndisp(displacementPage.count);\ndisp(stress.field_id);\ndisp(stressPage.count);\n"
        ),
    )
    .unwrap();
    let fields = runmat_from_repository(
        &temp,
        &["run", script_path.to_str().expect("UTF-8 script path")],
    );
    assert!(
        fields.status.success(),
        "STEP-backed result field smoke failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&fields.stdout),
        String::from_utf8_lossy(&fields.stderr)
    );
    let stdout = String::from_utf8_lossy(&fields.stdout);
    assert!(stdout.contains("displacement"));
    assert!(stdout.contains("von_mises"));
}

#[test]
fn response_spectrum_documentation_example_runs_its_two_step_study() {
    let temp = TempDir::new().unwrap();
    let path = "docs/fea/examples/modal-response-spectrum-experimental.fea";
    let output = runmat_from_repository(&temp, &["run", path, "--json"]);
    assert!(
        output.status.success(),
        "{path} failed run:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["root_id"], "modal_response_spectrum_experimental");
    assert_eq!(payload["procedure"], "structural_modal_response_spectrum");
    let step_run_ids = payload["step_run_ids"]
        .as_array()
        .expect("study step run ids");
    assert_eq!(step_run_ids.len(), 2);
    let response_run_id = step_run_ids[1]
        .as_str()
        .expect("response-spectrum step run id");

    let script_path = temp.path().join("response-spectrum-field.m");
    fs::write(
        &script_path,
        format!(
            "response = fea.results(\"{response_run_id}\");\nfield = fea.field(response, fea.ResponseSpectrumField.pseudoAcceleration());\npage = fea.fieldPage(field, 1, 3);\ndisp(field.field_id);\ndisp(field.statistic);\ndisp(page.count);\n"
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
    assert!(stdout.contains("structural.response_spectrum.nodal.pseudo_acceleration"));
    assert!(stdout.contains("component_magnitude_envelope"));
    assert!(stdout.contains("3"));
}

#[test]
fn named_response_spectrum_generator_materializes_before_the_solve() {
    let temp = TempDir::new().unwrap();
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    fs::write(
        temp.path().join("runmat.toml"),
        "[package]\nname = \"fea-spectrum-generator\"\n\n[sources]\nroots = [\".\"]\n",
    )
    .unwrap();
    fs::write(
        temp.path().join("design_input.m"),
        "function y = design_input(frequency, scale)\narguments\n    frequency (:,1) double\n    scale (1,1) double\nend\ny = scale .* (1 + 0 .* frequency);\nend\n",
    )
    .unwrap();
    let source = repository.join("docs/fea/examples/modal-response-spectrum-experimental.fea");
    fs::create_dir_all(temp.path().join("geometry")).unwrap();
    fs::copy(
        repository.join("crates/runmat-geometry/io/tests/fixtures/box.brep"),
        temp.path().join("geometry/box.brep"),
    )
    .unwrap();
    let document = format!(
        "import function design_input from \"./design_input.m\"\n\n{}",
        fs::read_to_string(source).unwrap().replace(
            "        ordinates = [1.2, 1.2, 2.5, 1.8, 0.7] g",
            "        generated by function.design_input { arguments { scale = 1.5 } }",
        )
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
        serde_json::from_slice::<Value>(&run.stdout).unwrap()["root_id"],
        "modal_response_spectrum_experimental"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&run.stdout).unwrap()["procedure"],
        "structural_modal_response_spectrum"
    );
}

#[test]
fn random_response_documentation_example_exposes_paged_statistical_fields() {
    let temp = TempDir::new().unwrap();
    let path = "docs/fea/examples/modal-random-response-experimental.fea";
    let run = runmat_from_repository(&temp, &["run", path, "--json"]);
    assert!(
        run.status.success(),
        "{path} failed run:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let payload: Value = serde_json::from_slice(&run.stdout).unwrap();
    assert_eq!(payload["procedure"], "structural_random_response");
    let response_run_id = payload["step_run_ids"]
        .as_array()
        .and_then(|steps| steps.get(1))
        .and_then(Value::as_str)
        .expect("random-response step run id");
    let script_path = temp.path().join("random-response-fields.m");
    fs::write(
        &script_path,
        format!(
            "response = fea.results(\"{response_run_id}\");\nrms = fea.field(response, fea.RandomResponseField.rms(fea.StructuralField.displacement()));\nrmsPage = fea.fieldPage(rms, 1, 3);\nvariance = fea.field(response, fea.RandomResponseField.variance(fea.StructuralField.stress()));\nvariancePage = fea.fieldPage(variance, 1, 3);\ndisp(rms.field_id);\ndisp(rms.statistic);\ndisp(rmsPage.count);\ndisp(variance.field_id);\ndisp(variance.statistic);\ndisp(variancePage.count);\n"
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
    fs::write(
        temp.path().join("runmat.toml"),
        "[package]\nname = \"fea-callable-check\"\n\n[sources]\nroots = [\".\"]\n",
    )
    .unwrap();
    fs::write(
        temp.path().join("pulse.m"),
        "function y = pulse(t, magnitude)\narguments\n    t (:,1) double\n    magnitude (1,1) double\nend\ny = magnitude .* (1 + 0 .* t);\nend\n",
    )
    .unwrap();
    let fixture = repository.join("docs/fea/examples/structural-transient-experimental.fea");
    fs::create_dir_all(temp.path().join("geometry")).unwrap();
    fs::copy(
        repository.join("crates/runmat-geometry/io/tests/fixtures/box.brep"),
        temp.path().join("geometry/box.brep"),
    )
    .unwrap();
    let document = format!(
        "import function pulse from \"./pulse.m\"\n\n{}",
        fs::read_to_string(fixture)
            .unwrap()
        .replace("            end_time = 0.1 s", "            end_time = 0.003 s")
        .replace(
            "        samples = [\n            { at = 0 s, value = 0 N },\n            { at = 0.01 s, value = 1 kN },\n            { at = 0.02 s, value = 0 N },\n            { at = 0.1 s, value = 0 N },\n        ]\n        interpolation = linear\n        extrapolation = reject",
            "        generated by function.pulse {\n            arguments { magnitude = 1000 }\n            sample from 0 s to 0.003 s at 1 kHz\n            interpolation = linear\n            extrapolation = reject\n            tolerance = 1e-6 N\n        }",
        )
    );
    fs::write(temp.path().join("study.fea"), document).unwrap();

    let output = runmat(&temp, &["check", "study.fea", "--json"]);
    assert!(
        output.status.success(),
        "callable-backed study failed check:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["status"], "valid");

    let output = runmat(&temp, &["run", "study.fea", "--json"]);
    assert!(
        output.status.success(),
        "callable-backed study failed run:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["root_id"], "structural_transient_experimental");
    assert_eq!(payload["procedure"], "structural_transient");
    assert!(payload["study_definition_digest"].as_str().is_some());
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
        stderr.contains("RM.FEA.TYPED_EXECUTION.UNSUPPORTED_PHYSICS_FAMILY"),
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
