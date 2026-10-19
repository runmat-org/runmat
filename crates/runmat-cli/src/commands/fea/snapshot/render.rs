//! Presentation-only resume of authenticated original fields. A saved solve
//! remains an original solve, never a fresh qualification of changed numerics.
use anyhow::{anyhow, ensure, Context, Result};
use runmat_runtime::analysis::{
    analysis_results_by_run_id_op, AnalysisResultsData, AnalysisResultsQuery, AnalysisStudyRunData,
};
use serde::Deserialize;
use std::{io::Read, path::Path};

#[derive(Deserialize)]
struct SavedSnapshot {
    schema_version: u16,
    run_id: String,
    publication: AnalysisStudyRunData,
    results: AnalysisResultsData,
}
pub(in crate::commands::fea) async fn render_saved(input: &Path, output: &Path) -> Result<()> {
    let mut bytes = Vec::new();
    std::fs::File::open(input)?
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= 16 * 1024 * 1024,
        "saved snapshot metadata exceeds project limit"
    );
    let saved: SavedSnapshot = serde_json::from_slice(&bytes)?;
    ensure!(
        saved.schema_version == 1 && saved.publication.steps.len() == 1,
        "saved snapshot must contain one original study step"
    );
    let definition: runmat_execution::Digest = saved
        .publication
        .study_fingerprint
        .parse()
        .map_err(|error| anyhow!("invalid saved definition: {error}"))?;
    let manifest = saved
        .publication
        .result_manifest_artifact_path
        .as_ref()
        .context("saved study lacks its authenticated publication manifest")?;
    let publication =
        runmat_runtime::analysis::study_result_manifest::read_saved_study_publication(
            definition,
            Path::new(manifest),
            16 * 1024 * 1024,
        )
        .map_err(anyhow::Error::msg)?;
    ensure!(
        publication == saved.publication && publication.numerically_acceptable,
        "saved snapshot differs from original accepted study publication"
    );
    ensure!(
        publication.steps[0].run_id == saved.run_id,
        "saved run does not belong to the original study"
    );
    let profile = super::views::CaptureProfile::for_step(&publication.steps[0].procedure_step)?;
    let results = analysis_results_by_run_id_op(
        &saved.run_id,
        AnalysisResultsQuery {
            include_field_values: matches!(&profile, super::views::CaptureProfile::Modal { .. }),
            ..Default::default()
        },
        runmat_runtime::operations::OperationContext::new(None, None),
    )
    .map_err(|error| anyhow!("{}: {}", error.error_code, error.message))?
    .data;
    ensure!(
        results == saved.results && results.numerically_acceptable,
        "saved presentation metadata differs from authenticated original results"
    );
    let parent = output
        .parent()
        .context("snapshot output requires a parent directory")?;
    std::fs::create_dir_all(parent)?;
    let views = super::views::capture(&saved.run_id, parent, profile, &results).await?;
    std::fs::write(
        output,
        serde_json::to_vec_pretty(&super::Snapshot {
            schema_version: 1,
            run_id: &saved.run_id,
            publication: &publication,
            views,
            results: &results,
        })?,
    )?;
    println!("{}", output.display());
    Ok(())
}
