//! Complex publication binds retained declarations and original result authority.
use super::*;
use crate::execution::{
    SemanticStatePublicationProfileSource, SemanticStatePublicationRequest,
    SemanticStatePublicationSource, SolverResultIdentity, StatePublicationProfileEntry,
    StatePublicationProfiles,
};
use runmat_analysis_core::{StateCompleteness, StateKind, StateOutput, StatePersistence};
use runmat_canonical_codec::SerializedCanonicalProfile;

fn supported(declaration: &StateOutput) -> bool {
    declaration.kind == StateKind::ElectromagneticComplex
        && declaration.completeness == StateCompleteness::Complete
        && declaration.persistence != StatePersistence::CheckpointOnly
}

impl MaxwellHarmonicWorkload {
    pub fn state_publication_requests(
        &self,
        topology_digest: Digest,
        source_result: SolverResultIdentity,
    ) -> AnalysisExecutionResult<Vec<SemanticStatePublicationRequest>> {
        self.validate()?;
        self.model.sequence.steps[0]
            .state_outputs
            .iter()
            .filter(|declaration| declaration.persistence != StatePersistence::CheckpointOnly)
            .map(|declaration| {
                if !supported(declaration) {
                    return Err(invalid(
                        "Maxwell publication requires complete electromagnetic-complex state",
                    ));
                }
                SemanticStatePublicationRequest::new(
                    SemanticStatePublicationSource::MaxwellNativeFields,
                    self.producer_identity()?,
                    declaration.clone(),
                    topology_digest,
                    source_result,
                )
            })
            .collect()
    }

    /// Metadata-only profile of declaration inventory, not numerical state bytes.
    /// A mesh digest placeholder has the same fixed-width encoding as sampling.
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

impl MaxwellHarmonicSolverResult {
    pub fn state_publication_requests(
        &self,
    ) -> AnalysisExecutionResult<Vec<SemanticStatePublicationRequest>> {
        self.validate()?;
        let topology = self
            .sampling_topology()
            .identity()
            .map_err(invalid)?
            .discretization_digest();
        let source_result = self.identity()?;
        self.publication_declarations()
            .iter()
            .map(|declaration| {
                if !supported(declaration) {
                    return Err(invalid(
                        "retained Maxwell declaration is not a complete complex state",
                    ));
                }
                SemanticStatePublicationRequest::new(
                    SemanticStatePublicationSource::MaxwellNativeFields,
                    self.producer_identity(),
                    declaration.clone(),
                    topology,
                    source_result,
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
            return Err(invalid(
                "Maxwell publication differs from retained authored declaration/source/topology",
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
                "Maxwell publication profiles name another original source payload",
            ));
        }
        let entries = self
            .state_publication_requests()?
            .into_iter()
            .map(|request| {
                let profile = request.encoding_profile(
                    SemanticStatePublicationProfileSource::MaxwellNativeFields(self),
                )?;
                StatePublicationProfileEntry::new(request, profile)
            })
            .collect::<AnalysisExecutionResult<Vec<_>>>()?;
        StatePublicationProfiles::new(
            self.producer_identity(),
            self.identity()?,
            self.sampling_topology()
                .identity()
                .map_err(invalid)?
                .discretization_digest(),
            source_payload_digest,
            entries,
        )
    }
}
