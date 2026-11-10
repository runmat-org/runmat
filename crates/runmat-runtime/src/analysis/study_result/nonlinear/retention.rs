//! Runtime handoff owns its metadata and original carrier; shared state histories count once.
use super::*;
use runmat_canonical_codec::{profile_serialized, CanonicalLimits};
use serde::ser::SerializeStruct;

struct Metadata<'a>(&'a NonlinearSolidStudyCompletion);
impl serde::Serialize for Metadata<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let c = self.0;
        let mut value = serializer.serialize_struct("NonlinearCompletionMetadata", 14)?;
        value.serialize_field("compiled", &c.compiled)?;
        value.serialize_field("workload", &c.workload)?;
        value.serialize_field("terminal_block", &c.terminal_block)?;
        value.serialize_field("numerical_receipt", &c.numerical_receipt)?;
        value.serialize_field("host_workload_digest", &c.host_workload_digest)?;
        value.serialize_field("block_count", &c.block_count)?;
        value.serialize_field("observation_catalog", &c.observation_catalog)?;
        value.serialize_field("accuracy", &c.accuracy)?;
        value.serialize_field("baseline_usage", &c.baseline_usage)?;
        value.serialize_field("terminal_usage", &c.terminal_usage)?;
        value.serialize_field("publication", &c.publication)?;
        value.serialize_field("frozen_project_digest", &c.frozen_project_digest)?;
        value.serialize_field("root", &c.root)?;
        value.serialize_field("catalog", &c.catalog)?;
        value.end()
    }
}

impl NonlinearSolidStudyCompletion {
    /// Retained handoff inventory, excluding independently admitted decoder/wire construction.
    /// State aliases receive credit only for actual shared allocations, never equal digests.
    pub fn retained_storage_bytes(
        &self,
        maximum_bytes: u64,
    ) -> Result<u64, AnalysisPublicationError> {
        let limits = CanonicalLimits::content_bounded(
            usize::try_from(maximum_bytes)
                .map_err(|_| limited("nonlinear handoff exceeds address space"))?,
            65_536,
            64,
        );
        let metadata = profile_serialized(&Metadata(self), limits)?;
        let states = runmat_analysis_fea::execution::semantic_state_retained_storage_bytes(
            self.states.iter(),
            limits,
        )
        .map_err(|error| match error {
            runmat_analysis_fea::execution::AnalysisExecutionError::Codec(error) => error.into(),
            rejected => AnalysisPublicationError::from(rejected.to_string()),
        })?;
        let bytes = metadata
            .materialization_resident_bytes_estimate
            .checked_add(states)
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>() as u64))
            .ok_or_else(|| limited("nonlinear handoff storage overflow"))?;
        if bytes > maximum_bytes {
            return Err(limited("nonlinear handoff exceeds retained workspace"));
        }
        Ok(bytes)
    }
}
fn limited(message: &str) -> AnalysisPublicationError {
    AnalysisPublicationError::ResourceLimit {
        message: message.into(),
    }
}
