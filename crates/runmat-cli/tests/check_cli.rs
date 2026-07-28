use serde_json::Value;
use std::fs;
use std::process::{Command, Output};
use tempfile::TempDir;

fn runmat(temp: &TempDir, args: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_runmat"));
    command.current_dir(temp.path()).args(args);
    command.env_remove("RUNMAT_CONFIG");
    command.output().expect("run runmat check")
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
