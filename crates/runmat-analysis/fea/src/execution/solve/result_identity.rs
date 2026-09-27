//! Numerical identity is independent of transport encoding for solved thermal preload.
//! The receipt separately commits to the complete transport payload and workload.

use super::*;
use runmat_execution::Digest;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum ResultIdentitySource {
    Payload,
    ThermalPhysical {
        producer: super::super::StepIdentity,
        content: Digest,
    },
}

impl ResultIdentitySource {
    pub(super) fn from_result(result: &StructuralLinearWorkloadResult) -> Self {
        match result
            .linearization
            .as_ref()
            .and_then(|value| value.thermal_source())
        {
            Some(source) => Self::ThermalPhysical {
                producer: source.transition().consumer,
                content: source.structural_result_content(),
            },
            None => Self::Payload,
        }
    }

    pub(super) fn identity(
        &self,
        workload: AnalysisWorkloadIdentity,
        payload: Digest,
    ) -> AnalysisExecutionResult<SolverResultIdentity> {
        match self {
            Self::Payload => SolverResultIdentity::new(workload, payload),
            Self::ThermalPhysical { producer, content } => {
                SolverResultIdentity::from_step_result(*producer, *content)
            }
        }
    }
}

impl StructuralLinearWorkloadResult {
    pub fn result_identity(
        &self,
        workload: AnalysisWorkloadIdentity,
    ) -> AnalysisExecutionResult<SolverResultIdentity> {
        self.validate()?;
        ResultIdentitySource::from_result(self)
            .identity(workload, Digest::sha256(self.canonical_bytes()?))
    }
}

impl StructuralLinearNumericalReceiptArtifact {
    pub(super) fn expected_identity(&self) -> AnalysisExecutionResult<NumericalReceiptIdentity> {
        let commitment = Digest::sha256(encode_contract(
            SOLVE_PREFIX,
            "structural-linear-receipt-source",
            &(
                self.source_workload,
                self.source_payload_digest,
                &self.source_identity,
                self.receipt,
            ),
            SOLVE_LIMITS,
        )?);
        NumericalReceiptIdentity::new(self.source_result, commitment)
    }
}
