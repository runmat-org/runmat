//! Placement-neutral solved thermal evidence; the source port authenticates its complete closure.
use runmat_analysis_fea::execution::{
    CanonicalSemanticState, ThermalNumericalReceiptArtifact, ThermalSolverResultManifest,
    ThermalWorkload,
};
use runmat_execution::{
    value::{PublishedObjectCatalog, ValueRef},
    Digest,
};

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct ThermalStudyCompletion {
    pub observations: Option<super::ThermalObservationPublication>,
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

impl ThermalStudyCompletion {
    /// Original completion metadata and its validated constitutive inventory.
    pub(in crate::analysis) fn retained_storage_bytes_estimate(
        &self,
        limits: runmat_canonical_codec::CanonicalLimits,
    ) -> Result<u64, String> {
        self.shared_clone_storage_bytes_estimate(limits)?
            .checked_add(
                self.compiled
                    .properties
                    .retained_storage_bytes_estimate(limits)?,
            )
            .ok_or_else(|| "thermal original storage overflow".into())
    }

    /// Additional completion metadata introduced by a clone; compiled property
    /// inventories remain shared. Original ownership and codec work are separate.
    pub(in crate::analysis) fn shared_clone_storage_bytes_estimate(
        &self,
        limits: runmat_canonical_codec::CanonicalLimits,
    ) -> Result<u64, String> {
        let Self {
            observations,
            compiled,
            workload,
            host_workload_digest,
            manifest,
            numerical_receipt,
            field_partitions,
            accuracy,
            states,
            baseline_usage,
            terminal_usage,
            publication,
            frozen_project_digest,
            root,
            catalog,
        } = self;
        // Typed owners (cloned, never materialized as a value tree).
        let metadata = runmat_canonical_codec::profile_typed_resident_bytes(
            &(
                observations,
                workload,
                host_workload_digest,
                manifest,
                numerical_receipt,
                field_partitions,
                accuracy,
                states,
                baseline_usage,
                terminal_usage,
                publication,
                frozen_project_digest,
                root,
                catalog,
            ),
            limits,
        )
        .map_err(|error| error.to_string())?;
        metadata
            .checked_add(compiled.shared_clone_storage_bytes_estimate(limits)?)
            .ok_or_else(|| "thermal completion clone storage overflow".into())
    }
}
