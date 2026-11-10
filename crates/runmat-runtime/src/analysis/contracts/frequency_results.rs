//! Typed whole-frequency result metadata; no representative sample aliases.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Published acoustic sweep record. Per-node values live in the frequency-sample field
/// partitions and in persisted native states; entries keep scalar closure and identities.
pub struct AcousticFrequencyResultsData {
    pub workload: runmat_analysis_fea::execution::AcousticFrequencyWorkload,
    pub publication: runmat_analysis_fea::execution::FrequencySamplePublicationManifest,
    pub entries: Vec<AcousticFrequencyEntryEvidence>,
    /// The catalog of the histories over the sweep's frequencies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub history_catalog:
        Option<runmat_analysis_fea::results::observation::ObservationPublicationCatalog>,
    /// The series reductions of those histories (a peak over frequency, for example).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub series_reductions: Vec<runmat_analysis_fea::results::observation::SeriesReductionResult>,
    pub resource_usage: runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
}

/// One exact authored frequency entry: scalar result summary, lifecycle evidence and state
/// references.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticFrequencyEntryEvidence {
    pub entry: runmat_analysis_fea::frequency::FrequencyResponsePlanEntry,
    pub host_workload_digest: runmat_execution::Digest,
    pub result: runmat_analysis_fea::execution::AcousticHarmonicResultSummary,
    pub numerical_receipt: runmat_analysis_fea::execution::AcousticNumericalReceiptArtifact,
    /// Authored outputs evaluated at this frequency (pressure, level, peaks).
    #[serde(default)]
    pub observations: Vec<runmat_analysis_fea::results::observation::ObservationResult>,
    /// The publication catalog of those outputs, which the entry's accuracy evaluation names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observation_catalog:
        Option<runmat_analysis_fea::results::observation::ObservationPublicationCatalog>,
    pub accuracy: runmat_analysis_fea::results::accuracy::AccuracyArtifact,
    pub states: Vec<crate::analysis::study_result_manifest::PersistedStateEvidence>,
    pub baseline_usage: runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
    pub terminal_usage: runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
    pub publication: runmat_analysis_fea::results::publication::PublishedAnalysisResult,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralFrequencyResultsData {
    pub publication: runmat_analysis_fea::execution::StructuralFrequencyPublicationManifest,
}
