//! Native artifact production for documentation examples. Uses the admitted public
//! execution route and original persisted fields; playback never reruns this work.
use anyhow::{anyhow, bail, Context, Result};
use runmat_runtime::analysis::{FeaRootSelector, ResolvedFeaRoot};
use serde::Serialize;
use std::path::Path;

pub(super) mod render;
pub(super) mod verify;
mod views;

#[derive(Serialize)]
struct Snapshot<'a> {
    schema_version: u16,
    run_id: &'a str,
    publication: &'a runmat_runtime::analysis::AnalysisStudyRunData,
    views: Vec<views::View>,
    results: &'a runmat_runtime::analysis::AnalysisResultsData,
}

pub(super) async fn capture(
    source: &super::source::NativeCheckedFeaSource,
    root: ResolvedFeaRoot,
    output: &Path,
) -> Result<()> {
    let ResolvedFeaRoot::Study(study) = root else {
        bail!("snapshot requires one study");
    };
    let steps = &study.study.sequence.steps;
    if steps.len() != 1 {
        bail!("snapshot requires one standalone analysis step");
    }
    let profile = views::CaptureProfile::for_step(&steps[0])?;
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
    let runmat_execution_runner_native::LocalFeaExecution::Study(completed) = result else {
        bail!("snapshot execution returned a sweep");
    };
    if !completed.publication.numerically_acceptable {
        bail!("snapshot execution did not produce an acceptable numerical result");
    }
    let run_id = &completed
        .publication
        .steps
        .first()
        .ok_or_else(|| anyhow!("snapshot has no step"))?
        .run_id;
    let parent = output
        .parent()
        .context("snapshot output requires a parent directory")?;
    std::fs::create_dir_all(parent)?;
    let results = runmat_runtime::analysis::analysis_results_by_run_id_op(
        run_id,
        runmat_runtime::analysis::AnalysisResultsQuery {
            // The original modal basis is field-bearing query data. Request it
            // for frequency labels rather than inferring modes from a summary.
            include_field_values: matches!(&profile, views::CaptureProfile::Modal { .. }),
            ..Default::default()
        },
        runmat_runtime::operations::OperationContext::new(None, None),
    )
    .map_err(|error| anyhow!("{}: {}", error.error_code, error.message))?
    .data;
    // Preserve the original accepted numerical checkpoint before expensive scene
    // export. The coordinator keeps failed projects diagnostic-only; this file
    // cannot make an incomplete figure capture publishable.
    let accepted_checkpoint = parent.join("accepted-solve.json");
    std::fs::write(
        &accepted_checkpoint,
        serde_json::to_vec_pretty(&Snapshot {
            schema_version: 1,
            run_id,
            publication: &completed.publication,
            views: Vec::new(),
            results: &results,
        })?,
    )?;
    let views = views::capture(run_id, parent, profile, &results).await?;
    let bytes = serde_json::to_vec_pretty(&Snapshot {
        schema_version: 1,
        run_id,
        publication: &completed.publication,
        views,
        results: &results,
    })?;
    std::fs::write(output, bytes)?;
    std::fs::remove_file(accepted_checkpoint)?;
    println!("{}", output.display());
    Ok(())
}
