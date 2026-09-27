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
    /// Validate retained thermal physics against the complete authored worker input.
    pub fn validate_thermal_workload(
        &self,
        workload: &StructuralLinearWorkload,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        workload.validate()?;
        let source = self
            .linearization
            .as_ref()
            .and_then(|value| value.thermal_source());
        match (workload.state_input.as_ref(), source) {
            (Some(StructuralLinearStateInput::ThermalPrestress(transition)), Some(source)) => {
                source
                    .validate_model(&workload.model)
                    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
                if source.transition() != transition {
                    return Err(AnalysisExecutionError::Identity(
                        "thermal structural result retained a different source transition",
                    ));
                }
            }
            (Some(StructuralLinearStateInput::ThermalPrestress(_)), None) | (_, Some(_)) => {
                return Err(AnalysisExecutionError::Identity(
                    "thermal structural result source differs from the prepared workload",
                ));
            }
            _ => {}
        }
        Ok(())
    }

    /// Validate physical source identity at downstream publication/recovery boundaries which
    /// possess the result and producer identity but not the full authored model.
    pub fn validate_thermal_source(
        &self,
        producer: super::super::StepIdentity,
        result: SolverResultIdentity,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        if let Some(source) = self
            .linearization
            .as_ref()
            .and_then(|value| value.thermal_source())
        {
            if source.transition().consumer != producer || source.structural_result() != result {
                return Err(AnalysisExecutionError::Identity("thermal structural result differs from its publication producer or physical source"));
            }
        }
        Ok(())
    }

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
