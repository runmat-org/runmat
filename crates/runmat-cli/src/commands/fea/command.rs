use anyhow::Result;
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_runtime::analysis::{FeaRootSelector, FeaSourceCheckResult, ResolvedFeaRoot};
use std::path::PathBuf;

use crate::cli::{Cli, ExecutionTargetArgs, FeaRootArgs};
use crate::commands::script::resolve_script_input;
use crate::presentation;
use crate::AlreadyReportedCliError;

/// `runmat plan <file>.fea`
pub async fn plan(
    file: PathBuf,
    root: FeaRootArgs,
    json: bool,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
) -> Result<()> {
    let file = resolve_script_input(file)?;
    let checked = super::source::check_source(&file, cli, config).await?;
    require_valid(&checked, &file, json)?;
    super::plan::print(&resolve_root(&checked, &root)?, json)
}

/// `runmat run <file>.fea`: the run applies exactly the checks `runmat check` reports, then
/// prepares, meshes and solves the selected root.
pub(crate) async fn run(
    file: PathBuf,
    root: FeaRootArgs,
    execution: ExecutionTargetArgs,
    json: bool,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
) -> Result<()> {
    let file = resolve_script_input(file)?;
    let checked = super::source::check_source(&file, cli, config).await?;
    require_valid(&checked, &file, json)?;
    let selected = resolve_root(&checked, &root)?;
    super::run::execute(file, checked, selected, execution, cli, config, json).await
}

/// `runmat snapshot <file>.fea --out <metadata>`
pub async fn snapshot(
    file: PathBuf,
    study: Option<String>,
    step: Option<String>,
    case: Option<String>,
    views: Vec<String>,
    out: PathBuf,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
) -> Result<()> {
    let file = resolve_script_input(file)?;
    let checked = super::source::check_source(&file, cli, config).await?;
    require_valid(&checked, &file, true)?;
    let selected = resolve_root(&checked, &FeaRootArgs { study, sweep: None })?;
    super::snapshot::capture(
        &checked,
        selected,
        step.as_deref(),
        case.as_deref(),
        &views,
        &out,
    )
    .await
}

pub async fn render_snapshot(snapshot: PathBuf, out: PathBuf) -> Result<()> {
    super::snapshot::render::render_saved(&snapshot, &out).await
}

pub fn verify_snapshot(snapshot: PathBuf, out: PathBuf) -> Result<()> {
    super::snapshot::verify::verify_saved_fields(&snapshot, &out)
}

/// `runmat reassess <file>.fea`
pub async fn reassess(
    file: PathBuf,
    study: Option<String>,
    original_definition: runmat_execution::Digest,
    result_manifest: PathBuf,
    json: bool,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
) -> Result<()> {
    let file = resolve_script_input(file)?;
    let checked = super::source::check_source(&file, cli, config).await?;
    require_valid(&checked, &file, json)?;
    let selected = resolve_root(&checked, &FeaRootArgs { study, sweep: None })?;
    let ResolvedFeaRoot::Study(study) = selected else {
        anyhow::bail!("reassessment requires one study; select it with --study");
    };
    let result = runmat_execution_runner_native::reassess_saved_fea_locally(
        &checked.session_source,
        original_definition,
        &result_manifest,
        &study.study,
        runmat_runtime::analysis::assessment::AssessmentEvaluationLimits::default(),
        64 * 1024 * 1024,
    )
    .await
    .map_err(anyhow::Error::msg)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        for assessment in &result.assessments {
            println!(
                "{}: {:?}",
                assessment.definition.assessment_id, assessment.result.status
            );
            if let Some(issue) = assessment.result.issue {
                println!("  evidence: {issue:?}");
            }
            if let Some(diagnostic) = &assessment.function_execution_diagnostic {
                println!("  execution: {diagnostic}");
            }
        }
        println!("Reassessment evidence: {}", result.evidence_artifact_path);
    }
    Ok(())
}

fn require_valid(
    source: &super::source::NativeCheckedFeaSource,
    file: &std::path::Path,
    json: bool,
) -> Result<()> {
    let report = source.check_result();
    if report.status.is_valid() {
        Ok(())
    } else {
        print_check_failure(source, file, &report, json)
    }
}

fn print_check_failure(
    source: &super::source::NativeCheckedFeaSource,
    file: &std::path::Path,
    report: &FeaSourceCheckResult,
    json: bool,
) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(report)?);
    } else {
        print_diagnostics(source, file, report);
    }
    Err(AlreadyReportedCliError.into())
}

/// Prints located diagnostics to stderr in the `runmat check` form, with its summary line.
fn print_diagnostics(
    source: &super::source::NativeCheckedFeaSource,
    file: &std::path::Path,
    report: &FeaSourceCheckResult,
) {
    if report.diagnostics.is_empty() {
        return;
    }
    let styles = presentation::stderr();
    let sources = crate::commands::check::FeaDiagnosticSources::new(
        file,
        &source.root_module,
        &source.source_root,
    );
    let counts =
        crate::commands::check::print_fea_diagnostics(&report.diagnostics, &sources, &styles, true);
    eprintln!("{}", counts.summary(file, &styles));
}

pub(crate) fn root_selector(args: &FeaRootArgs) -> Option<FeaRootSelector> {
    args.study
        .as_ref()
        .map(|id| FeaRootSelector::Study { id: id.clone() })
        .or_else(|| {
            args.sweep
                .as_ref()
                .map(|id| FeaRootSelector::Sweep { id: id.clone() })
        })
}

pub(crate) fn resolve_root(
    source: &super::source::NativeCheckedFeaSource,
    args: &FeaRootArgs,
) -> Result<ResolvedFeaRoot> {
    let selector = root_selector(args);
    source
        .resolve_root(selector.as_ref())
        .map_err(anyhow::Error::new)
}
