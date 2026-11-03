//! Cold portable field closure check. Streams the existing paged reader; never
//! imports geometry, meshes, solves, or materializes all field values together.
use anyhow::{anyhow, ensure, Context, Result};
use runmat_runtime::analysis::{
    analysis_field_page_by_run_id_op, analysis_results_by_run_id_op, AnalysisFieldPageValues,
    AnalysisFieldRequestOptions, AnalysisFieldSelection, AnalysisResultsData, AnalysisResultsQuery,
};
use runmat_runtime::operations::OperationContext;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

mod geometry;

#[derive(Deserialize)]
struct SavedRun {
    run_id: String,
    results: AnalysisResultsData,
}

#[derive(Deserialize)]
struct Snapshot {
    schema_version: u16,
    #[serde(flatten)]
    primary: SavedRun,
    /// Further captured runs of a study that publishes several cases.
    #[serde(default)]
    runs: Vec<SavedRun>,
}

#[derive(Serialize)]
struct FieldReplay {
    field_id: String,
    scalar_count: usize,
    pages: usize,
    scalar_bytes_sha256: String,
}

pub(in crate::commands::fea) fn verify_saved_fields(snapshot: &Path, output: &Path) -> Result<()> {
    // Match the existing per-file docs project limit. Decode cannot grow an
    // unbounded metadata file before the public artifact reader applies limits.
    const MAXIMUM_METADATA_BYTES: u64 = 16 * 1024 * 1024;
    let mut bytes = Vec::new();
    std::fs::File::open(snapshot)?
        .take(MAXIMUM_METADATA_BYTES + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAXIMUM_METADATA_BYTES,
        "snapshot metadata exceeds project file limit"
    );
    let snapshot: Snapshot = serde_json::from_slice(&bytes)?;
    ensure!(snapshot.schema_version == 1, "unsupported snapshot schema");
    let runs = std::iter::once(&snapshot.primary)
        .chain(&snapshot.runs)
        .map(verify_run)
        .collect::<Result<Vec<_>>>()?;
    let mut receipt = runs.into_iter();
    let (run_id, fields, geometry) = receipt.next().context("snapshot has no run")?;
    let others = receipt
        .map(|(run_id, fields, geometry)| {
            serde_json::json!({ "runId": run_id, "fields": fields, "geometry": geometry })
        })
        .collect::<Vec<_>>();
    std::fs::write(
        output,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schemaVersion": 1, "runId": run_id, "fields": fields,
            "geometry": geometry, "runs": others,
            "method": "fresh native process, original authenticated artifact readers, all bounded field pages and saved geometry tiles; no solve"
        }))?,
    )?;
    Ok(())
}

fn verify_run(snapshot: &SavedRun) -> Result<(String, Vec<FieldReplay>, serde_json::Value)> {
    let cold = analysis_results_by_run_id_op(
        &snapshot.run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .map_err(|e| anyhow!("{}: {}", e.error_code, e.message))?
    .data;
    ensure!(
        cold.field_descriptors == snapshot.results.field_descriptors,
        "cold descriptors differ from original capture"
    );
    ensure!(
        !cold.field_descriptors.is_empty(),
        "snapshot has no physical fields"
    );
    let geometry = geometry::verify_saved_tiles(&snapshot.run_id)?;
    let mut replay = Vec::new();
    for descriptor in cold.field_descriptors {
        let mut offset = 0;
        let mut pages = 0;
        let mut hash = Sha256::new();
        while offset < descriptor.value_count {
            let count = (descriptor.value_count - offset).min(4096);
            let page = analysis_field_page_by_run_id_op(
                &snapshot.run_id,
                AnalysisFieldSelection::new(&descriptor.field_id, descriptor.sample),
                AnalysisFieldRequestOptions {
                    offset: Some(offset),
                    limit: Some(count),
                },
                OperationContext::new(None, None),
            )
            .map_err(|e| anyhow!("{}: {}", e.error_code, e.message))?
            .data;
            ensure!(
                page.offset == offset
                    && page.count == count
                    && page.total_count == descriptor.value_count
                    && page.run_id == snapshot.run_id
                    && page.field_id == descriptor.field_id,
                "cold field page differs from original descriptor scope"
            );
            let (values, components) = match page.values {
                AnalysisFieldPageValues::Real(values) => (values, 1),
                AnalysisFieldPageValues::ComplexInterleaved(values) => (values, 2),
            };
            ensure!(
                values.len()
                    == count
                        .checked_mul(components)
                        .context("field storage count overflow")?,
                "cold field storage differs from logical page count"
            );
            ensure!(
                values.iter().all(|value| value.is_finite()),
                "cold field contains nonfinite values"
            );
            for value in values {
                hash.update(value.to_bits().to_le_bytes());
            }
            offset = offset
                .checked_add(count)
                .context("field page offset overflow")?;
            pages += 1;
        }
        replay.push(FieldReplay {
            field_id: descriptor.field_id,
            scalar_count: offset,
            pages,
            scalar_bytes_sha256: hex::encode(hash.finalize()),
        });
    }
    Ok((snapshot.run_id.clone(), replay, geometry))
}
