use super::*;
pub trait AnalysisArtifactStore: Send + Sync {
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
        source: &dyn runmat_execution::value::PublishedObjectSource,
    ) -> Result<AnalysisArtifactRecord, String>;
    fn load_structural_frequency(
        &self,
        run_id: &str,
    ) -> Result<Option<AnalysisStudyStepRunData>, String>;
}
