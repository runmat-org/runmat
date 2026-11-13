use super::*;
pub trait AnalysisArtifactStore: Send + Sync {
    fn filesystem_scope(&self) -> Option<(PathBuf, Arc<dyn runmat_filesystem::FsProvider>)> {
        None
    }
    /// Export one authenticated original field under the caller's cumulative read ceiling.
    fn export_initial_field(
        &self,
        run_id: &str,
        field_id: &str,
        maximum_bytes: u64,
    ) -> Result<Option<runmat_analysis_fea::execution::InitialFieldBinding>, AnalysisPublicationError>;

    /// Publish one privately prepared closed inventory; no child is made visible first.
    fn commit_publication(
        &self,
        publication: PreparedAnalysisPublication,
    ) -> Result<AnalysisPublicationReceipt, AnalysisPublicationError>;
    fn load_publication(
        &self,
        identity: Digest,
        maximum_bytes: u64,
    ) -> Result<Option<serde_json::Value>, String>;
    /// Resolve only a committed manifest owned by this backend; charge all proof metadata.
    fn load_publication_manifest_path(
        &self,
        path: &Path,
        maximum_bytes: u64,
    ) -> Result<serde_json::Value, AssessmentStoreError>;
    fn load_observation_publication(
        &self,
        run_id: &str,
    ) -> Result<Option<crate::analysis::ThermalObservationPublication>, String>;
    fn persist_thermal_observations(
        &self,
        run_id: &str,
        completed: &crate::analysis::ThermalStudyCompletion,
        source: &crate::analysis::ThermalStudyObjectSource,
    ) -> Result<(), String>;
    fn load_observation_history_sample(
        &self,
        run_id: &str,
        observation_id: &runmat_analysis_core::ObservationId,
        sample: runmat_analysis_fea::results::field::RecoveredFieldSample,
    ) -> Result<ObservationHistoryLookup, String>;
    fn persist_run(&self, run: &AnalysisRunResult) -> Result<AnalysisArtifactRecord, String>;
    fn load_run(&self, run_id: &str) -> Result<Option<AnalysisRunResult>, String>;
    fn load_run_metadata(&self, run_id: &str) -> Result<Option<AnalysisRunResult>, String>;
    /// A definition run id resolves to its latest committed attempt; any other id is itself.
    fn resolve_run_attempt(&self, run_id: &str) -> Result<String, String> {
        Ok(run_id.to_owned())
    }
    /// Committed attempts of a definition run id, oldest first.
    fn run_attempts(&self, _definition_run_id: &str) -> Result<Vec<String>, String> {
        Ok(Vec::new())
    }
    /// Authenticate scalar evidence without exceeding the caller's artifact byte budget.
    /// Every store implements this boundary; an unbounded metadata fallback is not permitted.
    fn load_assessment_metadata(
        &self,
        run_id: &str,
        maximum_bytes: u64,
    ) -> Result<Option<AssessmentRunEvidence>, AssessmentStoreError>;
    fn load_field_inventory(&self, run_id: &str) -> Result<Option<Vec<AnalysisField>>, String>;
    fn load_render_tile_index(&self, run_id: &str) -> Result<Option<RenderTileIndex>, String>;
    fn plan_render_view(
        &self,
        run_id: &str,
        view: Option<RenderViewRequest>,
    ) -> Result<Option<RenderViewSelection>, String>;
    fn load_render_tile(
        &self,
        run_id: &str,
        mesh_index: usize,
        tile_index: usize,
    ) -> Result<Option<RenderTile>, String>;
    fn load_render_lod_node(
        &self,
        run_id: &str,
        mesh_index: usize,
        node_index: usize,
    ) -> Result<Option<RenderTile>, String>;
    fn load_render_topology(
        &self,
        run_id: &str,
    ) -> Result<Option<crate::analysis::AnalysisRenderTopology>, String>;
    fn load_field_page(
        &self,
        run_id: &str,
        selection: &AnalysisFieldSelection,
        options: AnalysisFieldRequestOptions,
    ) -> Result<AnalysisFieldPageLookup, String>;
    fn list_run_metadata(&self) -> Result<Vec<AnalysisRunResult>, String>;
    fn list_runs(&self) -> Result<Vec<AnalysisRunResult>, String>;
    fn persist_structural_frequency(
        &self,
        step: &AnalysisStudyStepRunData,
        object_catalog: &runmat_execution::value::PublishedObjectCatalog,
        artifact_roots: &crate::analysis::StructuralFrequencyArtifactRootIndex,
        render_topology: Option<&crate::analysis::AnalysisRenderTopology>,
        source: &dyn runmat_execution::value::PublishedObjectSource,
    ) -> Result<AnalysisArtifactRecord, String>;
    fn load_structural_frequency(
        &self,
        run_id: &str,
    ) -> Result<Option<AnalysisStudyStepRunData>, String>;
    fn persist_acoustic_frequency(
        &self,
        step: &AnalysisStudyStepRunData,
        object_catalog: &runmat_execution::value::PublishedObjectCatalog,
        artifact_roots: &crate::analysis::FrequencySampleArtifactRootInventory,
        render_topology: Option<&crate::analysis::AnalysisRenderTopology>,
        source: &dyn runmat_execution::value::PublishedObjectSource,
    ) -> Result<AnalysisArtifactRecord, String>;
    fn load_acoustic_frequency(
        &self,
        run_id: &str,
    ) -> Result<Option<AnalysisStudyStepRunData>, String>;
}
