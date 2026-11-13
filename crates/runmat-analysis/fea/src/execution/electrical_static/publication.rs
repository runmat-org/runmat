//! Exact retained authored authority; detached publication never invents a declaration.
use super::*;
use crate::execution::{
    SemanticStatePublicationProfileSource, SemanticStatePublicationRequest,
    SemanticStatePublicationSource, SolverResultIdentity, StatePublicationProfileEntry,
    StatePublicationProfiles,
};
use runmat_analysis_core::{StateCompleteness, StateKind, StateOutput, StatePersistence};
use runmat_canonical_codec::SerializedCanonicalProfile;

impl ElectricalStaticWorkload {
    pub(super) fn publication_declaration(&self) -> AnalysisExecutionResult<Option<&StateOutput>> {
        self.validate()?;
        match self.model.sequence.steps[0].state_outputs.as_slice() {
            [] => Ok(None),
            [declaration]
                if declaration.kind == StateKind::ElectromagneticStatic
                    && declaration.completeness == StateCompleteness::Complete
                    && declaration.persistence != StatePersistence::CheckpointOnly =>
            {
                Ok(Some(declaration))
            }
            _ => Err(invalid(
                "electrical publication requires at most one complete persisted static declaration",
            )),
        }
    }
    pub fn state_publication_requests(
        &self,
        topology_digest: Digest,
        source_result: SolverResultIdentity,
    ) -> AnalysisExecutionResult<Vec<SemanticStatePublicationRequest>> {
        self.publication_declaration()?
            .map(|declaration| {
                SemanticStatePublicationRequest::new(
                    SemanticStatePublicationSource::StaticElectricalNativeFields,
                    self.producer_identity()?,
                    declaration.clone(),
                    topology_digest,
                    source_result,
                )
            })
            .transpose()
            .map(|request| request.into_iter().collect())
    }
    /// Fixed-width source identity placeholders are encoded sizing, not publication evidence.
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
impl ElectricalStaticSolverResult {
    pub fn state_publication_requests(
        &self,
    ) -> AnalysisExecutionResult<Vec<SemanticStatePublicationRequest>> {
        self.validate()?;
        self.publication_declaration()
            .map(|declaration| {
                SemanticStatePublicationRequest::new(
                    SemanticStatePublicationSource::StaticElectricalNativeFields,
                    self.producer(),
                    declaration.clone(),
                    Digest::from_bytes(*self.mesh_digest().bytes()),
                    self.source_result_identity()?,
                )
            })
            .transpose()
            .map(|request| request.into_iter().collect())
    }
    pub fn validate_publication_request(
        &self,
        request: &SemanticStatePublicationRequest,
    ) -> AnalysisExecutionResult<()> {
        request.validate()?;
        if self.state_publication_requests()?.as_slice() != [request.clone()] {
            return Err(invalid(
                "electrical publication request differs from retained authored result authority",
            ));
        }
        Ok(())
    }
    pub fn state_publication_profiles(
        &self,
        source_payload_digest: Digest,
    ) -> AnalysisExecutionResult<StatePublicationProfiles> {
        if Digest::sha256(self.canonical_bytes()?) != source_payload_digest {
            return Err(invalid(
                "electrical publication profiles name another source payload",
            ));
        }
        let entries = self
            .state_publication_requests()?
            .into_iter()
            .map(|request| {
                let profile = request.encoding_profile(
                    SemanticStatePublicationProfileSource::StaticElectricalNativeFields(self),
                )?;
                StatePublicationProfileEntry::new(request, profile)
            })
            .collect::<AnalysisExecutionResult<Vec<_>>>()?;
        StatePublicationProfiles::new(
            self.producer(),
            self.source_result_identity()?,
            Digest::from_bytes(*self.mesh_digest().bytes()),
            source_payload_digest,
            entries,
        )
    }
}
