//! Typed whole-frequency result metadata; no representative sample aliases.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticFrequencyResultsData {
    pub workload: runmat_analysis_fea::execution::AcousticFrequencyWorkload,
    pub result: runmat_analysis_fea::execution::AcousticFrequencyResult,
    pub publication: runmat_analysis_fea::execution::FrequencySamplePublicationManifest,
    pub entries: Vec<crate::analysis::AcousticStudyEntryCompletion>,
    pub resource_usage: runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralFrequencyResultsData {
    pub publication: runmat_analysis_fea::execution::StructuralFrequencyPublicationManifest,
}
