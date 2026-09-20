use anyhow::Result;
use runmat_config::runtime::RunMatRuntimeConfig;

use crate::cli::{Cli, FeaCommand};
use crate::commands::script::resolve_script_input;
use crate::presentation;
use crate::AlreadyReportedCliError;

use super::selection::select_root;

pub(super) async fn execute(
    command: FeaCommand,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
) -> Result<()> {
    match command {
        FeaCommand::Check { file, root, json } => {
            let file = resolve_script_input(file)?;
            let checked = super::source::check_source(&file, cli, config).await?;
            if !checked.program.is_valid() || !checked.program.diagnostics.is_empty() {
                return print_diagnostics(&checked, json);
            }
            let selected = (root.study.is_some() || root.sweep.is_some())
                .then(|| select_root(&checked, &root))
                .transpose()?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "kind": "fea",
                        "status": "valid",
                        "root_module": checked.root_module,
                        "selected_kind": selected.as_ref().map(|root| root.kind()),
                        "selected_id": selected.as_ref().map(|root| root.id()),
                        "studies": checked.program.studies.len(),
                        "sweeps": checked.program.sweeps.len(),
                        "diagnostics": checked.program.diagnostics,
                    }))?
                );
            } else {
                let styles = presentation::stdout();
                println!("{}", styles.success("FEA source is valid"));
                println!(
                    "  {} {}",
                    styles.label("studies:"),
                    checked.program.studies.len()
                );
                println!(
                    "  {} {}",
                    styles.label("sweeps:"),
                    checked.program.sweeps.len()
                );
                if let Some(root) = selected {
                    println!(
                        "  {} {} {}",
                        styles.label("selected:"),
                        root.kind(),
                        styles.identifier(root.id())
                    );
                }
            }
            Ok(())
        }
        FeaCommand::Plan { file, root, json } => {
            let file = resolve_script_input(file)?;
            let checked = super::source::check_source(&file, cli, config).await?;
            require_valid(&checked, json)?;
            super::plan::print(&select_root(&checked, &root)?, json)
        }
        FeaCommand::Run {
            file,
            root,
            execution,
            json,
        } => {
            let file = resolve_script_input(file)?;
            let checked = super::source::check_source(&file, cli, config).await?;
            require_valid(&checked, json)?;
            let selected = select_root(&checked, &root)?;
            super::run::execute(file, checked, selected, execution, cli, config, json).await
        }
    }
}

fn require_valid(source: &super::source::NativeCheckedFeaSource, json: bool) -> Result<()> {
    if source.program.is_valid() && source.program.diagnostics.is_empty() {
        Ok(())
    } else {
        print_diagnostics(source, json)
    }
}

fn print_diagnostics(source: &super::source::NativeCheckedFeaSource, json: bool) -> Result<()> {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "kind": "fea",
                "status": "invalid",
                "root_module": source.root_module,
                "diagnostics": source.program.diagnostics,
            }))?
        );
    } else {
        for diagnostic in &source.program.diagnostics {
            eprintln!(
                "{:?} {}: {}",
                diagnostic.severity, diagnostic.code, diagnostic.message
            );
        }
    }
    Err(AlreadyReportedCliError.into())
}
