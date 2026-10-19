//! Simulate transport loss/corruption without modifying original immutable artifacts.
use super::*;
use runmat_analysis_execution::{import_analysis_output_set, AnalysisHostWorkload};
use runmat_analysis_fea::execution::AnalysisObjectManifest;
use runmat_execution_artifact::{
    cache::CacheImport, ArtifactResult, ProgramExecutionRequest, ProgramExecutionResponse,
};

pub(super) struct OriginalStore<'a> {
    source: &'a FilesystemObjectStore,
    selected: Option<Digest>,
    admission: SourceAdmissionCase,
}
impl OriginalStore<'_> {
    fn fault(&self, digest: Digest, bytes: Option<Vec<u8>>) -> Option<Vec<u8>> {
        if Some(digest) != self.selected {
            return bytes;
        }
        match self.admission {
            SourceAdmissionCase::MissingSourceChunk => None,
            SourceAdmissionCase::CorruptSourceChunk => bytes.map(|mut bytes| {
                bytes[0] ^= 1;
                bytes
            }),
            _ => bytes,
        }
    }
}
impl CacheImport for OriginalStore<'_> {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        Ok(self.fault(digest, self.source.read_verified(digest)?))
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        Ok(self.fault(
            digest,
            self.source.read_verified_bounded(digest, maximum_bytes)?,
        ))
    }
}
pub(super) fn original_store<'a>(
    store: &'a FilesystemObjectStore,
    request: &ProgramExecutionRequest,
    response: &ProgramExecutionResponse,
    admission: SourceAdmissionCase,
) -> AnalysisBridgeResult<OriginalStore<'a>> {
    let selected = if matches!(
        admission,
        SourceAdmissionCase::MissingSourceChunk | SourceAdmissionCase::CorruptSourceChunk
    ) {
        let host = AnalysisHostWorkload::from_program_request(request, store)?;
        let completed = import_analysis_output_set(store, response.clone(), &host)?;
        let root = completed
            .manifest
            .outputs
            .iter()
            .find(|o| o.kind == AnalysisNumericalObjectKind::SolverResult)
            .unwrap();
        let bytes = store
            .read_verified_bounded(root.root.logical_digest, root.root.encoded_length)?
            .unwrap();
        let manifest = AnalysisObjectManifest::from_canonical_bytes(&bytes)?;
        Some(manifest.chunks.last().unwrap().digest)
    } else {
        None
    };
    Ok(OriginalStore {
        source: store,
        selected,
        admission,
    })
}
