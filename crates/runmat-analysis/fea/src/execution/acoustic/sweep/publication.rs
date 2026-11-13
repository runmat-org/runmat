//! Bind the shared frequency inventory to authenticated original acoustic outputs.
use super::*;
use crate::execution::{
    FieldRecoverySource, FieldTopologyIdentity, FrequencySampleNumericalEntry,
    FrequencySamplePublicationAuthority, FrequencySamplePublicationManifest,
    FrequencySamplePublicationSample,
};

impl AcousticFrequencyResult {
    /// Scientific authority comes from the complete validated numerical result,
    /// not identities supplied by an untrusted publication catalog.
    pub fn publication_authority(
        &self,
    ) -> AnalysisExecutionResult<FrequencySamplePublicationAuthority> {
        self.validate()?;
        let entries = self
            .workload
            .plan()
            .entries()
            .map_err(invalid)?
            .into_iter()
            .zip(self.results())
            .map(|(entry, result)| {
                Ok(FrequencySampleNumericalEntry {
                    entry,
                    workload: self.workload.entry(entry)?.identity()?,
                    source: FieldRecoverySource::SolverResult(result.source_result_identity()?),
                })
            })
            .collect::<AnalysisExecutionResult<Vec<_>>>()?;
        let producer = self
            .results()
            .first()
            .ok_or_else(|| invalid("frequency result is empty"))?
            .producer();
        if self
            .results()
            .iter()
            .any(|result| result.producer() != producer)
        {
            return Err(invalid("frequency result entries have different producers"));
        }
        FrequencySamplePublicationAuthority::new(
            self.workload.identity()?,
            self.source_result_identity()?,
            producer,
            FieldTopologyIdentity::VolumeMesh {
                mesh_digest: self.mesh_digest,
            },
            self.workload.plan().clone(),
            entries,
        )
    }

    pub fn publication_manifest(
        &self,
        samples: Vec<FrequencySamplePublicationSample>,
    ) -> AnalysisExecutionResult<FrequencySamplePublicationManifest> {
        FrequencySamplePublicationManifest::new(self.publication_authority()?, samples)
    }
}
