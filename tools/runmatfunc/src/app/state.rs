use std::path::Path;

use anyhow::Result;

use crate::app::config::AppConfig;
use crate::cli::CliArgs;
use crate::context::{manifest, serialize};
use crate::workspace::tests as workspace_tests;

/// Central application context carried throughout the CLI/TUI lifecycle.
pub struct AppContext {
    pub config: AppConfig,
}

impl AppContext {
    pub fn new(args: &CliArgs) -> Result<Self> {
        let config = AppConfig::load(args)?;
        Ok(Self { config })
    }

    pub fn launch_tui(&mut self) -> Result<()> {
        let manifest = manifest::build_manifest()?;
        crate::tui::run(manifest)
    }

    pub fn print_manifest(&mut self) -> Result<()> {
        let manifest = manifest::build_manifest()?;
        println!("RunMat builtins: {} entries", manifest.builtins.len());
        for record in &manifest.builtins {
            println!(
                "- {:<20} | {:<20} | sink: {} | accel: {}",
                record.name,
                record.category.as_deref().unwrap_or(""),
                record.is_sink,
                record.accel_tags.join(",")
            );
        }
        Ok(())
    }

    pub fn emit_docs(&mut self, out_dir: &str) -> Result<()> {
        let manifest = manifest::build_manifest()?;
        let out_path = Path::new(out_dir);
        serialize::write_manifest_files(&manifest, out_path)?;
        println!(
            "[runmatfunc] wrote documentation manifest to {}",
            out_path.display()
        );
        Ok(())
    }

    pub fn run_builtin(
        &mut self,
        name: &str,
        model: Option<String>,
        use_codex: bool,
    ) -> Result<()> {
        let ctx = crate::context::gather::build_authoring_context(name)?;
        println!(
            "Preparing builtin '{}' (model: {})",
            ctx.builtin.name,
            model.as_deref().unwrap_or("default")
        );
        println!("\nPrompt:\n{}", ctx.prompt);
        if let Some(doc) = &ctx.doc_markdown {
            println!("\nDocumentation (excerpt):\n{}", truncate(doc, 600));
        }
        println!("\nRelevant sources:");
        for path in &ctx.source_paths {
            println!("- {}", path.display());
        }
        if use_codex {
            match crate::codex::session::run_authoring(&ctx) {
                Ok(Some(note)) => {
                    println!("\n[runmatfunc] Codex response:\n{}", note);
                }
                Ok(None) => {
                    println!("\n[runmatfunc] Codex integration not enabled (requires embedded-codex feature).");
                }
                Err(err) => {
                    println!("\n[runmatfunc] Codex error: {err}");
                }
            }
        } else {
            println!("\n[runmatfunc] Codex integration skipped (pass --codex to enable).");
        }
        match workspace_tests::run_builtin_tests(&ctx) {
            Ok(outcome) => {
                if outcome.success {
                    println!("[runmatfunc] tests passed for {}", ctx.builtin.name);
                } else {
                    println!("[runmatfunc] tests failed for {}", ctx.builtin.name);
                    for report in outcome.reports {
                        println!(
                            "  {} -> {}",
                            report.label,
                            if report.success { "ok" } else { "FAILED" }
                        );
                        if !report.stdout.trim().is_empty() {
                            println!("    stdout:\n{}", indent(&report.stdout, 6));
                        }
                        if !report.stderr.trim().is_empty() {
                            println!("    stderr:\n{}", indent(&report.stderr, 6));
                        }
                    }
                }
            }
            Err(err) => {
                println!("[runmatfunc] test execution error: {err}");
            }
        }
        Ok(())
    }
}

fn truncate(input: &str, max: usize) -> String {
    if input.len() <= max {
        input.to_string()
    } else {
        format!("{}…", &input[..max])
    }
}

fn indent(text: &str, spaces: usize) -> String {
    let pad = " ".repeat(spaces);
    text.lines()
        .map(|line| format!("{}{}", pad, line))
        .collect::<Vec<_>>()
        .join("\n")
}
