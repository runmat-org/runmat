use std::env;
use std::process::{Command, Stdio};

use anyhow::{Context, Result};

use crate::context::types::AuthoringContext;

#[derive(Debug)]
pub struct TestCommandReport {
    pub label: String,
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Debug)]
pub struct TestOutcome {
    pub success: bool,
    pub reports: Vec<TestCommandReport>,
}

/// Run targeted cargo tests for a builtin and capture output.
pub fn run_builtin_tests(ctx: &AuthoringContext) -> Result<TestOutcome> {
    let current_dir = env::current_dir()?;
    let commands = vec![
        (
            "cargo test (lib)",
            vec![
                "test",
                "-p",
                "runmat-runtime",
                "--lib",
                "--",
                &ctx.builtin.name,
            ],
        ),
        (
            "cargo test (integration)",
            vec![
                "test",
                "-p",
                "runmat-runtime",
                "--tests",
                "--",
                &ctx.builtin.name,
            ],
        ),
    ];

    let mut reports = Vec::new();
    let mut overall_success = true;

    for (label, args) in commands {
        let output = Command::new("cargo")
            .args(&args)
            .current_dir(&current_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| format!("executing `{}`", args.join(" ")))?;

        let success = output.status.success();
        if !success {
            overall_success = false;
        }

        reports.push(TestCommandReport {
            label: label.to_string(),
            success,
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }

    Ok(TestOutcome {
        success: overall_success,
        reports,
    })
}
