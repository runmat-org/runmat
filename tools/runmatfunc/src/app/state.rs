use std::path::Path;

use anyhow::Result;

use crate::app::config::AppConfig;
use crate::cli::CliArgs;
use crate::context::{manifest, serialize};

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

    pub fn run_builtin(&mut self, name: &str, model: Option<String>) -> Result<()> {
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
        for path in ctx.source_paths {
            println!("- {}", path.display());
        }
        println!("\n[runmatfunc] Codex integration not yet implemented.");
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
