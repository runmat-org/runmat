//! Authored publication authority retained by the numerical result for detached workers.
use super::*;
use crate::execution::{
    SemanticStatePublicationProfileSource, SemanticStatePublicationRequest,
    SemanticStatePublicationSource, SequenceIdentity, SolverResultIdentity,
    StatePublicationProfileEntry, StatePublicationProfiles, StepIdentity,
};
use runmat_analysis_core::{IndependentDomain, StateOutput, StatePersistence};
use runmat_canonical_codec::SerializedCanonicalProfile;

impl AcousticHarmonicWorkload {
    pub fn producer_identity(&self) -> AnalysisExecutionResult<StepIdentity> {
        StepIdentity::from_solver_contract(
            SequenceIdentity::from_solver_contract(&self.model.sequence)?,
            self.step()?,
        )
    }

    pub(super) fn publication_declaration(&self) -> AnalysisExecutionResult<Option<&StateOutput>> {
        let step = self.step()?;
        let IndependentDomain::FrequencyHertz(frequencies) = &step.independent_domain else {
            return Ok(None);
        };
        let [declaration] = step.state_outputs.as_slice() else {
            return Ok(None);
        };
        Ok((frequencies.contains(&self.frequency.canonical_value())
            && declaration.kind == StateKind::AcousticComplex
            && declaration.completeness == StateCompleteness::Complete
            && declaration.persistence != StatePersistence::CheckpointOnly)
            .then_some(declaration))
    }

    /// The request is scoped to this entry's original result, never to an aggregate phasor.
    pub fn state_publication_requests(
        &self,
        topology_digest: Digest,
        source_result: SolverResultIdentity,
    ) -> AnalysisExecutionResult<Vec<SemanticStatePublicationRequest>> {
        self.publication_declaration()?
            .map(|declaration| {
                SemanticStatePublicationRequest::new(
                    SemanticStatePublicationSource::AcousticNativeFields,
                    self.producer_identity()?,
                    declaration.clone(),
                    topology_digest,
                    source_result,
                )
            })
            .transpose()
            .map(|request| request.into_iter().collect())
    }

    /// Fixed-width result identities here are sizing placeholders, never publication evidence.
    pub fn state_publication_profiles_encoding_profile(
        &self,
        topology_digest: Digest,
    ) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
        let requests = self.state_publication_requests(
            topology_digest,
            SolverResultIdentity::from_step_result(self.producer_identity()?, Digest::sha256([]))?,
        )?;
        StatePublicationProfiles::encoding_profile_for_requests(&requests)
    }
}

impl AcousticHarmonicSolverResult {
    pub fn state_publication_requests(
        &self,
    ) -> AnalysisExecutionResult<Vec<SemanticStatePublicationRequest>> {
        self.validate()?;
        self.publication_declaration()
            .map(|declaration| {
                SemanticStatePublicationRequest::new(
                    SemanticStatePublicationSource::AcousticNativeFields,
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
                "acoustic state publication differs from exact authored result authority",
            ));
        }
        Ok(())
    }

    /// Producer-side profiles borrow numerical arrays and bind their exact encoded source digest.
    pub fn state_publication_profiles(
        &self,
        source_payload_digest: Digest,
    ) -> AnalysisExecutionResult<StatePublicationProfiles> {
        if Digest::sha256(self.canonical_bytes()?) != source_payload_digest {
            return Err(invalid(
                "acoustic state profiles name another source payload",
            ));
        }
        let entries = self
            .state_publication_requests()?
            .into_iter()
            .map(|request| {
                let profile = request.encoding_profile(
                    SemanticStatePublicationProfileSource::AcousticNativeFields(self),
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
