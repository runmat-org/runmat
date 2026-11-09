//! Native artifact production for documentation examples. Uses the admitted public
//! execution route and original persisted fields; playback never reruns this work.
use anyhow::{anyhow, bail, Context, Result};
use runmat_runtime::analysis::{
    AnalysisResultsData, AnalysisStudyRunData, AnalysisStudyStepRunData, FeaRootSelector,
    ResolvedFeaRoot,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub(super) mod render;
pub(super) mod verify;
mod views;

/// One captured analysis step: its run, the study publication it belongs to and its original
/// results. A study whose execution publishes several cases (electrothermal conduction and
/// heat, one harmonic case per frequency) captures one run per selected case.
#[derive(Serialize, Deserialize, PartialEq)]
struct CapturedRun {
    /// The execution case, when the study publishes several.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    case: Option<String>,
    /// The captured analysis step (the final step unless one was selected).
    step: String,
    run_id: String,
    publication: AnalysisStudyRunData,
    results: AnalysisResultsData,
}

#[derive(Serialize)]
struct Snapshot<'a> {
    schema_version: u16,
    /// The unit each of the study's assessments writes its bound in, by assessment id: legends
    /// of an assessed quantity are written in it. A render of this snapshot reuses them.
    #[serde(skip_serializing_if = "no_units")]
    display_units: &'a std::collections::BTreeMap<String, String>,
    /// The primary run (the first captured case), at the top level.
    #[serde(flatten)]
    primary: &'a CapturedRun,
    views: Vec<views::View>,
    /// The other captured runs; each view names its run.
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    runs: &'a [CapturedRun],
}

fn no_units(units: &&std::collections::BTreeMap<String, String>) -> bool {
    units.is_empty()
}

fn step_list<'a>(ids: impl Iterator<Item = &'a str>) -> String {
    ids.collect::<Vec<_>>().join(", ")
}

/// The published step a snapshot captures: the named step, or the final step.
fn captured_step<'a>(
    publication: &'a AnalysisStudyRunData,
    step: Option<&str>,
) -> Result<Option<&'a AnalysisStudyStepRunData>> {
    let mut published = publication.steps.iter();
    Ok(match step {
        Some(id) => published.find(|candidate| candidate.procedure_step.step_id.as_str() == id),
        None => Some(
            published
                .max_by_key(|candidate| candidate.ordinal)
                .ok_or_else(|| anyhow!("snapshot has no step"))?,
        ),
    })
}

/// The step's original result metadata as the public results query returns it: descriptors,
/// summary, diagnostics and provenance. Field values, mode bases and histories stay in their
/// paged artifacts, so the metadata stays small at any mesh size; views read values page by
/// page.
fn step_results(run_id: &str) -> Result<AnalysisResultsData> {
    Ok(runmat_runtime::analysis::analysis_results_by_run_id_op(
        run_id,
        runmat_runtime::analysis::AnalysisResultsQuery::metadata_only(),
        runmat_runtime::operations::OperationContext::new(None, None),
    )
    .map_err(|error| anyhow!("{}: {}", error.error_code, error.message))?
    .data)
}

fn selects_case(case_id: &str, selected: Option<&str>) -> bool {
    selected.is_none_or(|selected| {
        case_id == selected
            || case_id
                .strip_prefix(selected)
                .is_some_and(|rest| rest.starts_with('/'))
    })
}

/// Views of the captured runs (the named ones, or every view the runs offer), written beside
/// `output`, then the snapshot metadata.
async fn write_snapshot(
    runs: &[CapturedRun],
    selected: &[String],
    output: &Path,
    presentation: &views::Presentation,
    display_units: &std::collections::BTreeMap<String, String>,
) -> Result<()> {
    let (primary, others) = runs
        .split_first()
        .ok_or_else(|| anyhow!("snapshot captured no run"))?;
    let parent = output
        .parent()
        .context("snapshot output requires a parent directory")?;
    std::fs::create_dir_all(parent)?;
    let profile = views::CaptureProfile::from_runs(
        &runs
            .iter()
            .map(|run| (run.run_id.as_str(), &run.results))
            .collect::<Vec<_>>(),
    )?
    .select(selected)?;
    let views = views::capture(
        parent,
        profile,
        |run_id| {
            runs.iter()
                .find(|run| run.run_id == run_id)
                .and_then(|run| run.publication.solver_mesh_artifact_path.clone())
        },
        presentation,
        &views::LegendTerms {
            display_units: display_units.clone(),
            assessments: runs
                .iter()
                .map(|run| (run.run_id.clone(), run.publication.assessments.clone()))
                .collect(),
        },
    )
    .await?;
    std::fs::write(
        output,
        serde_json::to_vec_pretty(&Snapshot {
            schema_version: 1,
            display_units,
            primary,
            views,
            runs: others,
        })?,
    )?;
    println!("{}", output.display());
    Ok(())
}

pub(super) async fn capture(
    source: &super::source::NativeCheckedFeaSource,
    root: ResolvedFeaRoot,
    step: Option<&str>,
    case: Option<&str>,
    selected_views: &[String],
    output: &Path,
    presentation: &crate::cli::SnapshotPresentationArgs,
) -> Result<()> {
    // Load context geometry and check the options before solving anything.
    let presentation = views::Presentation::from_args(presentation)?;
    let ResolvedFeaRoot::Study(study) = root else {
        bail!("snapshot requires one study");
    };
    let steps = &study.study.sequence.steps;
    if steps.is_empty() {
        bail!("snapshot requires at least one analysis step");
    }
    // Refuse an unknown step before solving anything.
    if let Some(id) = step {
        if !steps.iter().any(|declared| declared.step_id.as_str() == id) {
            bail!(
                "study has no analysis step '{id}'; steps: {}",
                step_list(steps.iter().map(|declared| declared.step_id.as_str()))
            );
        }
    }
    let result = runmat_execution_runner_native::execute_checked_fea_locally(
        &source.session_source,
        runmat_execution_runner_native::LocalFeaExecutionOptions {
            root: Some(FeaRootSelector::Study {
                id: study.study.study_id,
            }),
            ..Default::default()
        },
    )
    .await
    .map_err(anyhow::Error::msg)?;
    log::info!(
        "snapshot solve finished; peak resident {}",
        views::peak_resident_memory()
    );
    let completed = match result {
        runmat_execution_runner_native::LocalFeaExecution::Study(completed) => {
            vec![(None, completed.publication)]
        }
        runmat_execution_runner_native::LocalFeaExecution::Sweep { cases, .. } => cases
            .into_iter()
            .map(|case| match case.outcome {
                Ok(completed) => Ok((Some(case.case_id), completed.publication)),
                Err(message) => Err(anyhow!("case '{}' failed: {message}", case.case_id)),
            })
            .collect::<Result<Vec<_>>>()?,
    };
    let case_ids = completed
        .iter()
        .filter_map(|(case, _)| case.as_deref())
        .collect::<Vec<_>>()
        .join(", ");
    let mut runs = Vec::new();
    for (case_id, publication) in &completed {
        if !selects_case(case_id.as_deref().unwrap_or_default(), case) {
            continue;
        }
        if !publication.numerically_acceptable {
            bail!(
                "snapshot execution did not produce an acceptable numerical result{}",
                case_id
                    .as_deref()
                    .map(|case| format!(" in case '{case}'"))
                    .unwrap_or_default()
            );
        }
        // A case that does not run the selected step contributes nothing to its figure.
        let Some(captured) = captured_step(publication, step)? else {
            continue;
        };
        runs.push(CapturedRun {
            case: case_id.clone(),
            step: captured.procedure_step.step_id.as_str().to_string(),
            run_id: captured.run_id.clone(),
            publication: publication.clone(),
            results: step_results(&captured.run_id)?,
        });
    }
    if runs.is_empty() {
        bail!("no executed case matches the selection; cases: {case_ids}");
    }
    drop(completed);
    log::info!(
        "snapshot result metadata read; peak resident {}",
        views::peak_resident_memory()
    );
    let parent = output
        .parent()
        .context("snapshot output requires a parent directory")?;
    std::fs::create_dir_all(parent)?;
    let (primary, others) = runs.split_first().expect("runs are not empty");
    // Preserve the original accepted numerical checkpoint before expensive scene
    // export. The coordinator keeps failed projects diagnostic-only; this file
    // cannot make an incomplete figure capture publishable.
    let display_units = views::authored_bound_units(
        &source.session_source.checked().program,
        &primary.publication.study_id,
    );
    let accepted_checkpoint = parent.join("accepted-solve.json");
    std::fs::write(
        &accepted_checkpoint,
        serde_json::to_vec_pretty(&Snapshot {
            schema_version: 1,
            display_units: &display_units,
            primary,
            views: Vec::new(),
            runs: others,
        })?,
    )?;
    write_snapshot(&runs, selected_views, output, &presentation, &display_units).await?;
    std::fs::remove_file(accepted_checkpoint)?;
    Ok(())
}
