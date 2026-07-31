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
