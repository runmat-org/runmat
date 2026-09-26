//! Source-bound accepted thermal solve evidence; not an independent accuracy assessment.
use super::{
    thermal_result::ThermalResultMetadata, AnalysisExecutionError, AnalysisExecutionResult,
    AnalysisWorkloadIdentity, NumericalReceiptIdentity, SequenceIdentity, SolverResultIdentity,
    StepIdentity, ThermalSolverResultManifest, ThermalWorkload,
};
use crate::FeaThermalRunResult;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

const PREFIX: &[u8] = b"runmat-thermal-numerical-receipt/v1\0";
const LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024, 4096, 4096, 32);

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalNumericalReceipt {
    pub producer: StepIdentity,
    pub topology_digest: Digest,
    pub accepted_sample_count: u64,
    pub first_sample_time_s: f64,
    pub terminal_sample_time_s: f64,
    pub maximum_relative_linear_residual: f64,
    pub residual_target: f64,
    pub reference_temperature_k: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ReceiptWire")]
pub struct ThermalNumericalReceiptArtifact {
    pub schema_version: u16,
    identity: NumericalReceiptIdentity,
    pub source_workload: AnalysisWorkloadIdentity,
    pub source_result: SolverResultIdentity,
    receipt: ThermalNumericalReceipt,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptWire {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_result: SolverResultIdentity,
    receipt: ThermalNumericalReceipt,
}
impl TryFrom<ReceiptWire> for ThermalNumericalReceiptArtifact {
    type Error = AnalysisExecutionError;
    fn try_from(w: ReceiptWire) -> Result<Self, Self::Error> {
        let receipt = Self {
            schema_version: w.schema_version,
            identity: w.identity,
            source_workload: w.source_workload,
            source_result: w.source_result,
            receipt: w.receipt,
        };
        receipt.validate()?;
        Ok(receipt)
    }
}

fn invalid(message: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(format!("thermal numerical receipt: {message}"))
}

impl ThermalNumericalReceiptArtifact {
    pub fn new(
        workload: &ThermalWorkload,
        result: &FeaThermalRunResult,
    ) -> AnalysisExecutionResult<Self> {
        result.validate_complete()?;
        let source_workload = workload.identity()?;
        let source_result = result.source_result_identity()?;
        let receipt = evidence(workload, result.metadata())?;
        let artifact = Self {
            schema_version: 1,
            identity: receipt_identity(source_workload, source_result, &receipt)?,
            source_workload,
            source_result,
            receipt,
        };
        artifact.validate()?;
        Ok(artifact)
    }
    pub const fn identity(&self) -> NumericalReceiptIdentity {
        self.identity
    }
    pub const fn source_result(&self) -> SolverResultIdentity {
        self.source_result
    }
    pub const fn receipt(&self) -> ThermalNumericalReceipt {
        self.receipt
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let r = self.receipt;
        if self.schema_version != 1
            || r.accepted_sample_count == 0
            || !r.first_sample_time_s.is_finite()
            || !r.terminal_sample_time_s.is_finite()
            || r.first_sample_time_s.to_bits() == (-0.0_f64).to_bits()
            || r.terminal_sample_time_s.to_bits() == (-0.0_f64).to_bits()
            || (r.accepted_sample_count == 1 && r.first_sample_time_s != r.terminal_sample_time_s)
            || (r.accepted_sample_count > 1 && r.first_sample_time_s >= r.terminal_sample_time_s)
            || !r.maximum_relative_linear_residual.is_finite()
            || r.maximum_relative_linear_residual < 0.0
            || !r.residual_target.is_finite()
            || r.residual_target <= 0.0
            || r.maximum_relative_linear_residual > r.residual_target
            || !r.reference_temperature_k.is_finite()
            || r.reference_temperature_k <= 0.0
            || self.identity != receipt_identity(self.source_workload, self.source_result, &r)?
        {
            return Err(invalid(
                "invalid accepted evidence or nonconvergent identity",
            ));
        }
        Ok(())
    }

    pub fn validate_against(
        &self,
        workload: &ThermalWorkload,
        result: &ThermalSolverResultManifest,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        result.validate()?;
        let metadata = ThermalResultMetadata {
            source_workload: result.source_workload,
            numerical_intent: result.numerical_intent,
            run: &result.run,
            time_points_s: &result.time_points_s,
            residual_norms: &result.residual_norms,
            reference_temperature_k: result.reference_temperature_k,
            topology_digest: result.topology_digest,
            producer: result.producer,
            ordered_node_ids: &result.ordered_node_ids,
            ordered_element_ids: &result.ordered_element_ids,
            initial_state: &result.initial_state,
        };
        if self.source_workload != workload.identity()?
            || self.source_result != result.identity()
            || self.receipt != evidence(workload, metadata)?
        {
            return Err(invalid("receipt differs from workload or retained result"));
        }
        Ok(())
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "thermal-numerical-receipt",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            PREFIX,
            "thermal-numerical-receipt",
            bytes,
            LIMITS,
        )?)
    }
}

fn receipt_identity(
    workload: AnalysisWorkloadIdentity,
    result: SolverResultIdentity,
    receipt: &ThermalNumericalReceipt,
) -> AnalysisExecutionResult<NumericalReceiptIdentity> {
    let content = encode_contract(
        PREFIX,
        "thermal-numerical-receipt-content",
        &(workload, receipt),
        LIMITS,
    )?;
    NumericalReceiptIdentity::new(result, Digest::sha256(content))
}

fn evidence(
    workload: &ThermalWorkload,
    metadata: ThermalResultMetadata<'_>,
) -> AnalysisExecutionResult<ThermalNumericalReceipt> {
    workload.validate()?;
    metadata.validate()?;
    if metadata.source_workload != workload.identity()?
        || metadata.numerical_intent != workload.numerical_intent_identity()?
    {
        return Err(invalid(
            "result was produced by a different numerical workload",
        ));
    }
    let identity = &metadata.run.identity;
    crate::admission::admit_formulation(crate::admission::FormulationAdmissionRequest {
        family: identity.family(),
        procedure: identity.procedure(),
        solution_space: identity.solution_space(),
        requirement: workload.formulation_requirement,
        realization: identity.realization().clone(),
    })
    .map_err(invalid)?;
    let step = workload
        .model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id == workload.step_id)
        .ok_or_else(|| invalid("selected step is absent"))?;
    let producer = StepIdentity::from_solver_contract(
        SequenceIdentity::from_solver_contract(&workload.model.sequence)?,
        step,
    )?;
    let schedule = workload.solve_schedule().map_err(invalid)?;
    let maximum = metadata
        .residual_norms
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    if metadata.producer != producer
        || metadata.initial_state.procedure_step_id != workload.step_id.0
        || metadata.run.identity.procedure() != step.procedure
        || metadata.run.requested_backend != workload.backend
        || schedule.points_s() != metadata.time_points_s
        || workload
            .model
            .thermal
            .as_ref()
            .map(|domain| domain.reference_temperature.kelvin)
            != Some(metadata.reference_temperature_k)
        || maximum > workload.residual_target
    {
        return Err(invalid(
            "result producer, schedule, backend, reference, or residual differs from workload",
        ));
    }
    Ok(ThermalNumericalReceipt {
        producer,
        topology_digest: metadata.topology_digest,
        accepted_sample_count: u64::try_from(schedule.sample_count()).map_err(invalid)?,
        first_sample_time_s: schedule.points_s()[0],
        terminal_sample_time_s: *schedule.points_s().last().unwrap(),
        maximum_relative_linear_residual: maximum,
        residual_target: workload.residual_target,
        reference_temperature_k: metadata.reference_temperature_k,
    })
}

#[cfg(test)]
mod tests;
