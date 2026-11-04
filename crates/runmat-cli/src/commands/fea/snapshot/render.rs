//! Presentation-only resume of authenticated original fields. A saved solve
//! remains an original solve, never a fresh qualification of changed numerics.
use anyhow::{anyhow, ensure, Context, Result};
use serde::Deserialize;
use std::{io::Read, path::Path};

#[derive(Deserialize)]
struct SavedSnapshot {
    schema_version: u16,
    #[serde(flatten)]
    primary: super::CapturedRun,
    #[serde(default)]
    runs: Vec<super::CapturedRun>,
    /// The saved views; a render redraws the same ones.
    views: Vec<SavedView>,
}

#[derive(Deserialize)]
struct SavedView {
    id: String,
}

/// Check one saved run against its authenticated publication manifest and original results.
fn authenticate(saved: &super::CapturedRun) -> Result<()> {
    ensure!(
        !saved.publication.steps.is_empty(),
        "saved snapshot must contain its original study steps"
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
    let step = publication
        .steps
        .iter()
        .find(|step| step.run_id == saved.run_id)
        .context("saved run does not belong to the original study")?;
    ensure!(
        step.procedure_step.step_id.as_str() == saved.step,
        "saved step differs from the original study step of its run"
    );
    let results = super::step_results(&saved.run_id)?;
    ensure!(
        results == saved.results && results.numerically_acceptable,
        "saved presentation metadata differs from authenticated original results"
    );
    Ok(())
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
    ensure!(saved.schema_version == 1, "unsupported snapshot schema");
    let mut runs = vec![saved.primary];
    runs.extend(saved.runs);
    for run in &runs {
        authenticate(run)?;
    }
    let views: Vec<String> = saved.views.into_iter().map(|view| view.id).collect();
    super::write_snapshot(&runs, &views, output).await
}
