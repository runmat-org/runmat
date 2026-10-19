//! Exact retained peak electric phasors, exp(+iωt), and their original authority.
//! The result owns its input, canonical sampling, E circulations and E/B/H/D/J
//! samples. Publication neither re-solves nor fabricates vector potential A.
use super::*;
use crate::execution::MaxwellHarmonicSolverResult;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalMaxwellStatePayload {
    pub result: Box<MaxwellHarmonicSolverResult>,
}

impl CanonicalMaxwellStatePayload {
    pub fn validate_for(
        &self,
        topology_digest: Digest,
        source_result: SolverResultIdentity,
        producer: StepIdentity,
    ) -> AnalysisExecutionResult<()> {
        self.result.validate()?;
        let expected_topology = self
            .result
            .sampling_topology()
            .identity()
            .map_err(invalid)?
            .discretization_digest();
        if self.result.producer_identity() != producer
            || self.result.identity()? != source_result
            || expected_topology != topology_digest
        {
            return Err(invalid("Maxwell complex state differs from original retained producer/result/sample topology"));
        }
        Ok(())
    }
}

impl CanonicalSemanticState {
    pub fn from_maxwell_publication(
        request: &SemanticStatePublicationRequest,
        result: &MaxwellHarmonicSolverResult,
    ) -> AnalysisExecutionResult<Self> {
        result.validate_publication_request(request)?;
        let payload = CanonicalMaxwellStatePayload {
            result: Box::new(result.clone()),
        };
        payload.validate_for(
            request.topology_digest,
            request.source_result,
            request.producer,
        )?;
        Self::new(CanonicalSemanticStateInput {
            producer: request.producer,
            declaration: request.declaration.clone(),
            topology_digest: request.topology_digest,
            source_result: request.source_result,
            payload: CanonicalSemanticStatePayload::MaxwellHarmonic {
                state: Box::new(payload),
            },
        })
    }
}

fn invalid(reason: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(reason.to_string())
}

#[cfg(test)]
mod tests;
