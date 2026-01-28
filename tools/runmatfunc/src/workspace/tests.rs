use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::app::config::AppConfig;
use crate::context::types::AuthoringContext;
use anyhow::{Context, Result};

#[derive(Debug)]
pub struct TestCommandReport {
    pub label: String,
    pub command: Vec<String>,
    pub success: bool,
    pub skipped: bool,
    pub note: Option<String>,
    pub stdout: String,
    pub stderr: String,
    pub stdout_path: Option<PathBuf>,
    pub stderr_path: Option<PathBuf>,
}

#[derive(Debug)]
pub struct TestOutcome {
    pub success: bool,
    pub reports: Vec<TestCommandReport>,
    pub log_dir: PathBuf,
}

struct TestCommandPlan {
    label: String,
    args: Vec<String>,
    requires_features: Vec<String>,
}

/// Run targeted cargo tests for a builtin and capture output.
pub fn run_builtin_tests(ctx: &AuthoringContext, config: &AppConfig) -> Result<TestOutcome> {
    let log_dir = tests_log_dir(config, &ctx.builtin.name);
    fs::create_dir_all(&log_dir).with_context(|| format!("creating {}", log_dir.display()))?;

    let plans = build_test_plan(ctx, None);

    let mut reports = Vec::new();
    let mut overall_success = true;

    let workspace_root = std::env::current_dir()?;

    for (index, plan) in plans.into_iter().enumerate() {
        let (should_run, note) = should_run_command(&plan);
        if !should_run {
            reports.push(TestCommandReport {
                label: plan.label,
                command: plan.args,
                success: true,
                skipped: true,
                note,
                stdout: String::new(),
                stderr: String::new(),
                stdout_path: None,
                stderr_path: None,
            });
            continue;
        }

        let output = Command::new("cargo")
            .args(&plan.args)
            .current_dir(&workspace_root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| format!("executing `{}`", plan.args.join(" ")))?;

        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let success = output.status.success();
        if !success {
            overall_success = false;
        }

        let stdout_path = write_log(&log_dir, index, &plan.label, "stdout", &stdout)?;
        let stderr_path = write_log(&log_dir, index, &plan.label, "stderr", &stderr)?;

        reports.push(TestCommandReport {
            label: plan.label,
            command: plan.args,
            success,
            skipped: false,
            note,
            stdout,
            stderr,
            stdout_path,
            stderr_path,
        });
    }

    Ok(TestOutcome {
        success: overall_success,
        reports,
        log_dir,
    })
}

fn build_test_plan(ctx: &AuthoringContext, feature_flag: Option<&str>) -> Vec<TestCommandPlan> {
    let mut plans = Vec::new();
    let feature_vec = parse_feature_list(feature_flag);

    plans.push(TestCommandPlan {
        label: "cargo test (unit)".to_string(),
        args: build_cargo_args(TestScope::Unit, feature_flag, &ctx.builtin.name),
        requires_features: feature_vec.clone(),
    });

    if has_integration_tests(ctx) {
        plans.push(TestCommandPlan {
            label: "cargo test (integration)".to_string(),
            args: build_cargo_args(TestScope::Integration, feature_flag, &ctx.builtin.name),
            requires_features: feature_vec,
        });
    }

    plans
}

fn parse_feature_list(flag: Option<&str>) -> Vec<String> {
    flag.map(|f| {
        f.split(',')
            .filter_map(|item| {
                let trimmed = item.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            })
            .collect()
    })
    .unwrap_or_default()
}

fn should_run_command(plan: &TestCommandPlan) -> (bool, Option<String>) {
    if plan.requires_features.is_empty() {
        return (true, None);
    }

    let env_flag = std::env::var("RUNMATFUNC_ENABLE_OPTIONAL_FEATURES").unwrap_or_default();
    if matches!(env_flag.as_str(), "1" | "true" | "yes" | "on") {
        return (
            true,
            Some(format!(
                "running with optional feature(s): {}",
                plan.requires_features.join(", ")
            )),
        );
    }

    (
        false,
        Some(format!(
            "skipped (requires features: {} – set RUNMATFUNC_ENABLE_OPTIONAL_FEATURES=1 to enable)",
            plan.requires_features.join(", ")
        )),
    )
}

#[derive(Clone, Copy)]
enum TestScope {
    Unit,
    Integration,
}

fn build_cargo_args(scope: TestScope, feature_flag: Option<&str>, filter: &str) -> Vec<String> {
    let mut args = vec![
        "test".to_string(),
        "-p".to_string(),
        "runmat-runtime".to_string(),
    ];

    match scope {
        TestScope::Unit => args.push("--lib".to_string()),
        TestScope::Integration => args.push("--tests".to_string()),
    }

    if let Some(features) = feature_flag {
        args.push("--features".to_string());
        args.push(features.to_string());
    }

    args.push("--".to_string());
    args.push(filter.to_string());
    args
}

fn tests_log_dir(config: &AppConfig, builtin_name: &str) -> PathBuf {
    let mut dir = config.artifacts_dir().to_path_buf();
    dir.push("tests");
    dir.push(slugify(builtin_name));
    dir
}

fn slugify(input: &str) -> String {
    input
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn write_log(
    log_dir: &Path,
    index: usize,
    label: &str,
    kind: &str,
    content: &str,
) -> Result<Option<PathBuf>> {
    if content.trim().is_empty() {
        return Ok(None);
    }
    let file_name = format!("{:02}_{}_{}.log", index, slugify(label), kind);
    let path = log_dir.join(file_name);
    fs::write(&path, content).with_context(|| format!("writing {}", path.display()))?;
    Ok(Some(path))
}

fn has_integration_tests(ctx: &AuthoringContext) -> bool {
    ctx.source_paths
        .iter()
        .any(|path| path.to_string_lossy().contains("tests"))
}
