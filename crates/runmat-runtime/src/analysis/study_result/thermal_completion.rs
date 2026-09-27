//! Placement-neutral solved thermal evidence; the source port authenticates its complete closure.
use runmat_analysis_fea::execution::{
    CanonicalSemanticState, ThermalNumericalReceiptArtifact, ThermalSolverResultManifest,
    ThermalWorkload,
};
use runmat_execution::{
    value::{PublishedObjectCatalog, ValueRef},
    Digest,
};

#[derive(Clone, Debug, PartialEq)]
pub struct ThermalStudyCompletion {
    pub compiled: runmat_analysis_fea::study::CompiledAnalysisStudy,
    pub workload: ThermalWorkload,
    pub host_workload_digest: Digest,
    pub manifest: ThermalSolverResultManifest,
    pub numerical_receipt: ThermalNumericalReceiptArtifact,
    pub field_partitions: Vec<runmat_analysis_fea::results::field::RecoveredFieldPartition>,
    pub accuracy: runmat_analysis_fea::results::accuracy::AccuracyArtifact,
    pub states: Vec<CanonicalSemanticState>,
    pub baseline_usage: runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
    pub terminal_usage: runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
    pub publication: runmat_analysis_fea::results::publication::PublishedAnalysisResult,
    pub frozen_project_digest: Option<Digest>,
    pub root: ValueRef,
    pub catalog: PublishedObjectCatalog,
}
