use anyhow::Result;
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_runtime::analysis::{FeaRootSelector, FeaSourceCheckResult, ResolvedFeaRoot};

use crate::cli::{Cli, FeaCommand, FeaRootArgs};
use crate::commands::script::resolve_script_input;
use crate::presentation;
use crate::AlreadyReportedCliError;

pub(super) async fn execute(
    command: FeaCommand,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
) -> Result<()> {
    match command {
        FeaCommand::Check { file, root, json } => {
            let file = resolve_script_input(file)?;
            let checked = super::source::check_source(&file, cli, config).await?;
            let report = checked.check_result();
            if !report.status.is_valid() {
                return print_check_failure(&report, json);
            }
            let selector = root_selector(&root);
            let selected = selector
                .as_ref()
                .map(|selector| checked.resolve_root(Some(selector)))
                .transpose()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                let styles = presentation::stdout();
                println!("{}", styles.success("FEA source is valid"));
                println!(
                    "  {} {}",
                    styles.label("studies:"),
                    report
                        .roots
                        .iter()
                        .filter(|root| matches!(root, FeaRootSelector::Study { .. }))
                        .count()
                );
                println!(
                    "  {} {}",
                    styles.label("sweeps:"),
                    report
                        .roots
                        .iter()
                        .filter(|root| matches!(root, FeaRootSelector::Sweep { .. }))
                        .count()
                );
                if let Some(root) = selected {
                    println!(
                        "  {} {} {}",
                        styles.label("selected:"),
                        root.kind(),
                        styles.identifier(root.id())
                    );
                }
                print_diagnostics(&report);
            }
            Ok(())
        }
        FeaCommand::Plan { file, root, json } => {
            let file = resolve_script_input(file)?;
            let checked = super::source::check_source(&file, cli, config).await?;
            require_valid(&checked, json)?;
            super::plan::print(&resolve_root(&checked, &root)?, json)
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
            let selected = resolve_root(&checked, &root)?;
            super::run::execute(file, checked, selected, execution, cli, config, json).await
        }
    }
}

fn require_valid(source: &super::source::NativeCheckedFeaSource, json: bool) -> Result<()> {
    let report = source.check_result();
    if report.status.is_valid() {
        Ok(())
    } else {
        print_check_failure(&report, json)
    }
}

fn print_check_failure(report: &FeaSourceCheckResult, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(report)?);
    } else {
        eprintln!("FEA source check is {}", report.status);
        print_diagnostics(report);
    }
    Err(AlreadyReportedCliError.into())
}

fn print_diagnostics(report: &FeaSourceCheckResult) {
    for diagnostic in &report.diagnostics {
        eprintln!(
            "{:?} {}: {}",
            diagnostic.severity, diagnostic.code, diagnostic.message
        );
    }
}

fn root_selector(args: &FeaRootArgs) -> Option<FeaRootSelector> {
    args.study
        .as_ref()
        .map(|id| FeaRootSelector::Study { id: id.clone() })
        .or_else(|| {
            args.sweep
                .as_ref()
                .map(|id| FeaRootSelector::Sweep { id: id.clone() })
        })
}

fn resolve_root(
    source: &super::source::NativeCheckedFeaSource,
    args: &FeaRootArgs,
) -> Result<ResolvedFeaRoot> {
    let selector = root_selector(args);
    source
        .resolve_root(selector.as_ref())
        .map_err(anyhow::Error::new)
}
