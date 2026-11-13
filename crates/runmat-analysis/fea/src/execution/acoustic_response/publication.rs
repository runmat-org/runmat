//! Exact retained authored authority; detached publication never invents a declaration.
use super::*;
use crate::execution::{
    CanonicalSemanticState, CanonicalSemanticStateInput, CanonicalSemanticStatePayload,
    SemanticStatePublicationProfileSource, SemanticStatePublicationRequest,
    SemanticStatePublicationSource, SolverResultIdentity, StatePublicationProfileEntry,
    StatePublicationProfiles,
};
use runmat_canonical_codec::SerializedCanonicalProfile;

impl AcousticResponseWorkload {
    pub fn state_publication_requests(
        &self,
        topology_digest: Digest,
        source_result: SolverResultIdentity,
    ) -> AnalysisExecutionResult<Vec<SemanticStatePublicationRequest>> {
        let declaration = self.step()?.state_outputs[0].clone();
        Ok(vec![SemanticStatePublicationRequest::new(
            SemanticStatePublicationSource::AcousticResponseFields,
            self.producer_identity()?,
            declaration,
            topology_digest,
            source_result,
        )?])
    }
    /// Fixed-width identity placeholders are encoded sizing, not publication evidence.
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

impl AcousticResponseSolverResult {
    pub fn state_publication_requests(
        &self,
    ) -> AnalysisExecutionResult<Vec<SemanticStatePublicationRequest>> {
        self.validate()?;
        Ok(vec![SemanticStatePublicationRequest::new(
            SemanticStatePublicationSource::AcousticResponseFields,
            self.producer(),
            self.publication_declaration().clone(),
            Digest::from_bytes(*self.mesh_digest().bytes()),
            self.source_result_identity()?,
        )?])
    }
    pub fn validate_publication_request(
        &self,
        request: &SemanticStatePublicationRequest,
    ) -> AnalysisExecutionResult<()> {
        request.validate()?;
        if self.state_publication_requests()?.as_slice() != [request.clone()] {
            return Err(invalid(
                "acoustic response publication request differs from retained authored authority",
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
                "acoustic response profiles name another source payload",
            ));
        }
        let fields = self.state_fields();
        let entries = self
            .state_publication_requests()?
            .into_iter()
            .map(|request| {
                let profile = request.encoding_profile(
                    SemanticStatePublicationProfileSource::AcousticResponseFields(&fields),
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

impl CanonicalSemanticState {
    /// Mode shapes with frequencies, or the final transient pressure, as portable fields.
    pub fn from_acoustic_response_publication(
        request: &SemanticStatePublicationRequest,
        result: &AcousticResponseSolverResult,
    ) -> AnalysisExecutionResult<Self> {
        result.validate_publication_request(request)?;
        Self::new(CanonicalSemanticStateInput {
            producer: request.producer,
            declaration: request.declaration.clone(),
            topology_digest: request.topology_digest,
            source_result: request.source_result,
            payload: CanonicalSemanticStatePayload::Fields {
                fields: result.state_fields(),
            },
        })
    }
}
