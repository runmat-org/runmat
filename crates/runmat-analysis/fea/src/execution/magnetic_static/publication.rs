//! Publication is bound to original retained declarations, not invented at import.
use super::*;
use crate::execution::{
    SemanticStatePublicationProfileSource, SemanticStatePublicationRequest,
    SemanticStatePublicationSource, SolverResultIdentity, StatePublicationProfileEntry,
    StatePublicationProfiles,
};
use runmat_analysis_core::{StateCompleteness, StateKind, StateOutput, StatePersistence};
use runmat_canonical_codec::SerializedCanonicalProfile;

impl MagnetostaticWorkload {
    pub(super) fn publication_declarations(&self) -> AnalysisExecutionResult<Vec<&StateOutput>> {
        self.validate()?;
        Ok(self.model.sequence.steps[0]
            .state_outputs
            .iter()
            .filter(|declaration| {
                declaration.kind == StateKind::ElectromagneticStatic
                    && declaration.completeness == StateCompleteness::Complete
                    && declaration.persistence != StatePersistence::CheckpointOnly
            })
            .collect())
    }
    pub fn state_publication_requests(
        &self,
        topology_digest: Digest,
        source_result: SolverResultIdentity,
    ) -> AnalysisExecutionResult<Vec<SemanticStatePublicationRequest>> {
        self.publication_declarations()?
            .into_iter()
            .map(|declaration| {
                SemanticStatePublicationRequest::new(
                    SemanticStatePublicationSource::MagnetostaticNativeFields,
                    self.producer_identity()?,
                    declaration.clone(),
                    topology_digest,
                    source_result,
                )
            })
            .collect()
    }
    pub fn state_publication_profiles_encoding_profile(
        &self,
        topology_digest: Digest,
    ) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
        StatePublicationProfiles::encoding_profile_for_requests(&self.state_publication_requests(
            topology_digest,
            SolverResultIdentity::from_step_result(self.producer_identity()?, Digest::sha256([]))?,
        )?)
    }
}
impl MagnetostaticSolverResult {
    pub fn state_publication_requests(
        &self,
    ) -> AnalysisExecutionResult<Vec<SemanticStatePublicationRequest>> {
        self.validate()?;
        self.publication_declarations()
            .iter()
            .map(|declaration| {
                SemanticStatePublicationRequest::new(
                    SemanticStatePublicationSource::MagnetostaticNativeFields,
                    self.producer(),
                    declaration.clone(),
                    self.sampling_topology()
                        .identity()
                        .map_err(invalid)?
                        .discretization_digest(),
                    self.source_result_identity()?,
                )
            })
            .collect()
    }
    pub fn validate_publication_request(
        &self,
        request: &SemanticStatePublicationRequest,
    ) -> AnalysisExecutionResult<()> {
        request.validate()?;
        if !self.state_publication_requests()?.contains(request) {
            return Err(invalid("magnetic publication request differs from retained authored declaration/source/topology"));
        }
        Ok(())
    }
    pub fn state_publication_profiles(
        &self,
        source_payload_digest: Digest,
    ) -> AnalysisExecutionResult<StatePublicationProfiles> {
        if self.payload_digest()? != source_payload_digest {
            return Err(invalid(
                "magnetic publication profiles name another source payload",
            ));
        }
        let entries = self
            .state_publication_requests()?
            .into_iter()
            .map(|request| {
                let profile = request.encoding_profile(
                    SemanticStatePublicationProfileSource::MagnetostaticNativeFields(self),
                )?;
                StatePublicationProfileEntry::new(request, profile)
            })
            .collect::<AnalysisExecutionResult<Vec<_>>>()?;
        StatePublicationProfiles::new(
            self.producer(),
            self.source_result_identity()?,
            self.sampling_topology()
                .identity()
                .map_err(invalid)?
                .discretization_digest(),
            source_payload_digest,
            entries,
        )
    }
}
