//! `runmat check` and `runmat run` are the single entry points for `.fea` documents. A run
//! applies every static rule `check` reports, so both report the same diagnostic id, before any
//! geometry, mesh or solve work. Every published example checks clean.

use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn runmat(temp: &TempDir, cwd: &Path, args: &[&str]) -> Output {
    let config = temp.path().join("runmat.toml");
    let artifact_root =
        toml::Value::String(temp.path().join("artifacts").to_string_lossy().into_owned());
    fs::write(
        &config,
        format!(
            "[runtime.accelerate]\nenabled = false\nprovider = \"inprocess\"\n\n[runtime.fea]\nartifact_store = \"filesystem\"\nartifact_root = {artifact_root}\nstudy_artifact_root = {artifact_root}\n"
        ),
    )
    .unwrap();
    Command::new(env!("CARGO_BIN_EXE_runmat"))
        .current_dir(cwd)
        .arg("--offline")
        .args(args)
        .env("RUNMAT_CONFIG", config)
        .env(
            "RUNMAT_PACKAGE_CACHE_DIR",
            temp.path().join("package-cache"),
        )
        .env(
            "RUNMAT_EXECUTION_STATE_DIR",
            temp.path().join("execution-state"),
        )
        .output()
        .expect("run runmat")
}

fn json(output: &Output) -> Value {
    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(stdout.trim()).unwrap_or_else(|error| {
        panic!(
            "not JSON ({error}):\nstdout={stdout}\nstderr={}",
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

fn error_codes(payload: &Value) -> Vec<String> {
    payload["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|diagnostic| {
            diagnostic["severity"]
                .as_str()
                .is_some_and(|severity| severity.eq_ignore_ascii_case("error"))
        })
        .map(|diagnostic| diagnostic["code"].as_str().unwrap_or_default().to_string())
        .collect()
}

fn fea_examples() -> Vec<PathBuf> {
    let examples = repository().join("docs/simulation/examples");
    let mut found = Vec::new();
    let mut directories = vec![examples];
    while let Some(directory) = directories.pop() {
        for entry in fs::read_dir(&directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                directories.push(path);
            } else if path.extension().is_some_and(|extension| extension == "fea") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

// witness: FEA-PUBLIC-LANGUAGE-001/check_matches_run @ shared_infrastructure/mesh_independent/source_check; coverage: partial; boundary: Every published .fea example, including the finished bracket_strength study, passes `runmat check` with its static run admission; it does not run the studies.
#[test]
fn every_published_example_checks_clean() {
    let temp = TempDir::new().unwrap();
    let examples = fea_examples();
    assert!(examples.len() > 20, "{examples:?}");
    let failures = examples
        .iter()
        .filter_map(|path| {
            let output = runmat(
                &temp,
                &repository(),
                &["check", path.to_str().unwrap(), "--json"],
            );
            let payload = json(&output);
            (!output.status.success() || payload["status"] != "valid").then(|| {
                format!(
                    "{}: {}",
                    path.display(),
                    serde_json::to_string(&payload["diagnostics"]).unwrap()
                )
            })
        })
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty(),
        "examples fail check:\n{}",
        failures.join("\n")
    );
}

fn project_with(temp: &TempDir, name: &str, source: &str) -> PathBuf {
    let project = temp.path().join("project");
    fs::create_dir_all(project.join("geometry")).unwrap();
    for entry in fs::read_dir(repository().join("docs/simulation/examples/geometry")).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            fs::copy(
                &path,
                project.join("geometry").join(path.file_name().unwrap()),
            )
            .unwrap();
        }
    }
    fs::write(project.join(name), source).unwrap();
    project
}

fn example(name: &str) -> String {
    fs::read_to_string(repository().join("docs/simulation/examples").join(name)).unwrap()
}

/// Asserts `check` and `run` both refuse the source with `code`, and that the run stops before
/// any geometry or mesh work.
fn assert_check_and_run_agree(source: &str, code: &str, pending: Option<&str>) {
    let temp = TempDir::new().unwrap();
    let project = project_with(&temp, "study.fea", source);
    let checked = runmat(&temp, &project, &["check", "study.fea", "--json"]);
    assert!(!checked.status.success(), "{checked:?}");
    let checked = json(&checked);
    assert!(
        error_codes(&checked).contains(&code.to_string()),
        "{checked:#}"
    );
    let run = runmat(&temp, &project, &["run", "study.fea", "--json"]);
    assert!(!run.status.success(), "{run:?}");
    let run = json(&run);
    assert!(error_codes(&run).contains(&code.to_string()), "{run:#}");
    if let Some(pending) = pending {
        let text = serde_json::to_string(&checked["diagnostics"]).unwrap();
        assert!(text.contains(&format!("pending:{pending}")), "{text}");
    }
    assert!(
        !temp.path().join("artifacts").exists()
            || fs::read_dir(temp.path().join("artifacts"))
                .unwrap()
                .next()
                .is_none(),
        "the refused run produced artifacts"
    );
}

#[test]
fn pending_family_procedures_are_refused_by_check_and_run_alike() {
    let acoustic_static = example("acoustic-cavity-harmonic.fea").replace(
        "acoustic harmonic { frequencies = [100, 200, 400] Hz }",
        "acoustic static {}",
    );
    assert_check_and_run_agree(
        &acoustic_static,
        "fea.admission.family_procedure",
        Some("acoustic.static"),
    );
    // A harmonic field is complex and has no spatial maximum, so the steady example's
    // extrema outputs leave; the study still asks for the pending procedure.
    let fluid_harmonic = example("fluid-plane-poiseuille.fea")
        .replace(
            "analysis flow { fluid steady {} }",
            "analysis flow { fluid harmonic { frequencies = [10] Hz } }",
        )
        .lines()
        .filter(|line| !line.contains("= max(") && !line.contains("= pressure_drop("))
        .collect::<Vec<_>>()
        .join("\n");
    assert_check_and_run_agree(
        &fluid_harmonic,
        "fea.admission.family_procedure",
        Some("fluid.harmonic"),
    );
}

#[test]
fn several_runnable_roots_list_the_choices_and_select_alike() {
    let temp = TempDir::new().unwrap();
    // The sweep owns its parameterized study's default entry. Add an independent
    // study so this fixture still exercises ambiguous default-root selection.
    let source = format!(
        "{}\nstudy standalone {{
    geometry = geometry.bar
    mesh = mesh.bar
    model = model.cooled_bar(film = 35 W/m^2/K)
    analysis heat {{ thermal steady {{}} }}
}}\n",
        example("bar-cooling-sweep.fea")
    );
    let project = project_with(&temp, "study.fea", &source);
    let checked = runmat(&temp, &project, &["check", "study.fea", "--json"]);
    assert!(checked.status.success(), "{checked:?}");
    let checked = json(&checked);
    assert_eq!(checked["roots"].as_array().unwrap().len(), 2, "{checked:#}");
    let run = runmat(&temp, &project, &["run", "study.fea"]);
    assert!(!run.status.success());
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains("--study standalone") && stderr.contains("--sweep cooling"),
        "{stderr}"
    );
    let unknown = runmat(
        &temp,
        &project,
        &["check", "study.fea", "--study", "missing", "--json"],
    );
    assert!(!unknown.status.success());
    assert!(json(&unknown)["selection_error"].is_string());
    let selected = runmat(
        &temp,
        &project,
        &["check", "study.fea", "--sweep", "cooling", "--json"],
    );
    assert!(selected.status.success(), "{selected:?}");
    assert_eq!(json(&selected)["selected"]["id"], "cooling");
}

#[test]
fn root_selection_flags_belong_to_fea_documents() {
    let temp = TempDir::new().unwrap();
    fs::write(temp.path().join("main.m"), "x = 1;\n").unwrap();
    for command in ["check", "run"] {
        let output = runmat(&temp, temp.path(), &[command, "main.m", "--study", "a"]);
        assert!(!output.status.success(), "{command}: {output:?}");
        assert!(String::from_utf8_lossy(&output.stderr).contains(".fea"));
    }
}

const NAMED_STUDY: &str = r#"
geometry pair {
    file = "./geometry/two-box-assembly.step"
    units = m
    region left_body : solid { select name "left_body"; expect exactly_one }
    region extra : face { select name "operator_saved_face"; expect exactly_one }
}
"#;

// witness: FEA-MESH-GEOMETRY-001/geometry_load_units_option @ shared_infrastructure/mesh_independent/source_check; coverage: partial; boundary: `select name` resolves only names authored in the STEP file; a selections file under .runmat/geometry is never read.
#[test]
fn only_names_authored_in_the_cad_file_resolve() {
    let temp = TempDir::new().unwrap();
    let project = project_with(&temp, "study.fea", NAMED_STUDY);
    // A former operator-local selections file names the face; it must have no effect.
    fs::create_dir_all(project.join("geometry/.runmat/geometry")).unwrap();
    fs::write(
        project.join("geometry/.runmat/geometry/two-box-assembly.step.json"),
        r#"{"schema_version":1,"source_digest":[1],"regions":[{"region_id":"operator_saved_face","role":"face","selector":{"kind":"persistent_id","value":"persistent-entity:sha256:00"},"cardinality":"exactly_one"}]}"#,
    )
    .unwrap();
    let output = runmat(&temp, &project, &["check", "study.fea", "--json"]);
    assert!(!output.status.success(), "{output:?}");
    let text = serde_json::to_string(&json(&output)["diagnostics"]).unwrap();
    assert!(text.contains("operator_saved_face"), "{text}");
    assert!(!text.contains("left_body\\\""), "{text}");
    assert!(
        text.contains("left_support"),
        "lists the authored face names: {text}"
    );
}

/// A structural modal study whose assigned material has no mass_density is refused by check
/// and run with the same static admission code, before meshing.
#[test]
fn modal_study_without_mass_density_is_refused_by_check_and_run_alike() {
    let source = example("cantilever-modal.fea").replace("    mass_density = 7850 kg/m^3\n", "");
    assert!(!source.contains("mass_density"));
    assert_check_and_run_agree(&source, "fea.admission.material_mass_density", None);
}

#[test]
fn mode_located_field_output_is_refused_with_its_pending_id() {
    let source = example("cantilever-modal.fea").replace(
        "output shapes = analysis.modes.field.node.displacement",
        "output shapes = analysis.modes.field.mode.modal_frequency",
    );
    let temp = TempDir::new().unwrap();
    let project = project_with(&temp, "study.fea", &source);
    let checked = runmat(&temp, &project, &["check", "study.fea", "--json"]);
    assert!(!checked.status.success(), "{checked:?}");
    let text = serde_json::to_string(&json(&checked)["diagnostics"]).unwrap();
    assert!(text.contains("pending:output.mode-location"), "{text}");
}
