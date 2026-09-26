use super::*;
use crate::execution::{
    SemanticStatePublicationRequest, SemanticStatePublicationSource, SequenceIdentity,
    SolverResultIdentity, StatePublicationProfiles, StepIdentity,
};
use runmat_analysis_core::{StateCompleteness, StatePersistence};
use runmat_canonical_codec::SerializedCanonicalProfile;

impl ThermalWorkload {
    /// Exact authored persisted states supported by the thermal producer. Used both for
    /// pre-solve metadata admission and post-solve source-bound publication metadata.
    pub fn state_publication_requests(
        &self,
        topology_digest: Digest,
        source_result: SolverResultIdentity,
    ) -> AnalysisExecutionResult<Vec<SemanticStatePublicationRequest>> {
        let step = self.validated_step()?;
        let producer = StepIdentity::from_solver_contract(
            SequenceIdentity::from_solver_contract(&self.model.sequence)?,
            step,
        )?;
        step.state_outputs
            .iter()
            .filter(|declaration| {
                declaration.completeness == StateCompleteness::Complete
                    && declaration.persistence != StatePersistence::CheckpointOnly
            })
            .map(|declaration| {
                SemanticStatePublicationRequest::new(
                    SemanticStatePublicationSource::ThermalPrimaryField,
                    producer,
                    declaration.clone(),
                    topology_digest,
                    source_result,
                )
            })
            .collect()
    }

    /// Numerical identities have fixed wire width. A placeholder is used only for sizing
    /// the metadata wrapper; it is never emitted as result or publication evidence.
    pub fn state_publication_profiles_encoding_profile(
        &self,
        topology_digest: Digest,
    ) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
        let producer = StepIdentity::from_solver_contract(
            SequenceIdentity::from_solver_contract(&self.model.sequence)?,
            self.validated_step()?,
        )?;
        let requests = self.state_publication_requests(
            topology_digest,
            SolverResultIdentity::from_step_result(producer, Digest::sha256([]))?,
        )?;
        StatePublicationProfiles::encoding_profile_for_requests(&requests)
    }
}
