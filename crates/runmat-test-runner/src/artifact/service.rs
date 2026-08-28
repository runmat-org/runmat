use runmat_test::identity::RunId;

use crate::reporter::RenderedReport;
use crate::RunnerResult;

use super::{safe_artifact_name, ArtifactManifest, ArtifactStore};

/// Persists already-rendered reports without giving reporters storage
/// authority. Names are validated before the first store mutation.
pub async fn persist_reports<S: ArtifactStore>(
    store: &S,
    run_id: &RunId,
    reports: &[RenderedReport],
) -> RunnerResult<ArtifactManifest> {
    let names = reports
        .iter()
        .map(|report| safe_artifact_name(&report.name))
        .collect::<RunnerResult<Vec<_>>>()?;
    let mut artifacts = Vec::with_capacity(reports.len());
    for (report, name) in reports.iter().zip(names) {
        artifacts.push(
            store
                .put(run_id, &name, &report.media_type, report.bytes.as_slice())
                .await?,
        );
    }
    Ok(ArtifactManifest { artifacts })
}
