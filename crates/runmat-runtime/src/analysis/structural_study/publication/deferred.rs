//! Private completed steps remain unpublished until their whole-study root is committed.
use super::*;
use std::sync::Arc;

pub(in crate::analysis) struct DeferredStudyRun {
    pub(in crate::analysis) published: OperationEnvelope<AnalysisStudyRunData>,
    pub(in crate::analysis) staging: storage::AnalysisPublicationBuilder,
    pub(in crate::analysis) store: Arc<dyn storage::AnalysisArtifactStore>,
    pub(in crate::analysis) definition: runmat_execution::Digest,
}
impl DeferredStudyRun {
    pub(in crate::analysis) fn publish_legacy(
        self,
    ) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
        let context = OperationContext::new(
            self.published.trace_id.clone(),
            self.published.request_id.clone(),
        );
        #[derive(serde::Serialize)]
        struct LegacyManifest<'a> {
            schema_version: &'static str,
            study_definition_digest: runmat_execution::Digest,
            public_run: &'a AnalysisStudyRunData,
        }
        let manifest = self
            .staging
            .serialize_manifest(&LegacyManifest {
                schema_version: "fea_study_sequence_publication/v1",
                study_definition_digest: self.definition,
                public_run: &self.published.data,
            })
            .map_err(|error| {
                completion_storage_error(&context, "RM.FEA.RUN_STUDY.ARTIFACT_STORE_FAILED", error)
            })?;
        self.commit(manifest).map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error,
                &context,
                |message| {
                    completion_storage_error(
                        &context,
                        "RM.FEA.RUN_STUDY.ARTIFACT_STORE_FAILED",
                        message,
                    )
                },
            )
        })
    }
    pub(in crate::analysis) fn commit(
        mut self,
        manifest: serde_json::Value,
    ) -> Result<OperationEnvelope<AnalysisStudyRunData>, storage::AnalysisPublicationError> {
        if crate::interrupt::is_cancelled() {
            return Err("study publication was cancelled".into());
        }
        let prepared = self.staging.finish(manifest)?;
        let receipt = self.store.commit_publication(prepared)?;
        self.published.data.result_manifest_artifact_path = receipt
            .manifest_artifact_path
            .map(|path| path.to_string_lossy().into_owned());
        Ok(self.published)
    }
}
