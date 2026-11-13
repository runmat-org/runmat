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

const PREFIX: &[u8] = b"runmat-thermal-numerical-receipt/v2\0";
pub const THERMAL_RECEIPT_MAX_ENCODED_BYTES: u64 = 64 * 1024;
const LIMITS: CanonicalLimits =
    CanonicalLimits::new(THERMAL_RECEIPT_MAX_ENCODED_BYTES as usize, 4096, 4096, 32);

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalNumericalReceipt {
    pub producer: StepIdentity,
    pub topology_digest: Digest,
    pub accepted_sample_count: u64,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub first_sample_time_s: runmat_types::physical::Time,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub terminal_sample_time_s: runmat_types::physical::Time,
    pub maximum_relative_linear_residual: f64,
    pub nonlinear_sample_count: usize,
    pub maximum_relative_nonlinear_residual: Option<f64>,
    pub maximum_relative_nonlinear_increment: Option<f64>,
    pub maximum_relative_energy_residual: f64,
    pub residual_target: f64,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub reference_temperature_k: runmat_types::physical::Temperature,
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
        Self::from_metadata(
            workload,
            result.source_result_identity()?,
            result.metadata(),
        )
    }

    /// Bind accepted evidence from the validated header without reconstructing field arrays.
    /// This attests header evidence; transport closure validation remains the caller's duty.
    pub fn from_manifest(
        workload: &ThermalWorkload,
        result: &ThermalSolverResultManifest,
    ) -> AnalysisExecutionResult<Self> {
        result.validate()?;
        Self::from_metadata(workload, result.identity(), result.metadata())
    }

    fn from_metadata(
        workload: &ThermalWorkload,
        source_result: SolverResultIdentity,
        metadata: ThermalResultMetadata<'_>,
    ) -> AnalysisExecutionResult<Self> {
        let source_workload = workload.identity()?;
        let receipt = evidence(workload, metadata)?;
        let artifact = Self {
            schema_version: 2,
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
        if self.schema_version != 2
            || r.accepted_sample_count == 0
            || !r.first_sample_time_s.canonical_value().is_finite()
            || !r.terminal_sample_time_s.canonical_value().is_finite()
            || r.first_sample_time_s.canonical_value().to_bits() == (-0.0_f64).to_bits()
            || r.terminal_sample_time_s.canonical_value().to_bits() == (-0.0_f64).to_bits()
            || (r.accepted_sample_count == 1
                && r.first_sample_time_s.canonical_value()
                    != r.terminal_sample_time_s.canonical_value())
            || (r.accepted_sample_count > 1
                && r.first_sample_time_s.canonical_value()
                    >= r.terminal_sample_time_s.canonical_value())
            || !r.maximum_relative_linear_residual.is_finite()
            || r.maximum_relative_linear_residual < 0.0
            || !r.residual_target.is_finite()
            || r.residual_target <= 0.0
            || r.maximum_relative_linear_residual > r.residual_target
            || !match (
                r.maximum_relative_nonlinear_residual,
                r.maximum_relative_nonlinear_increment,
            ) {
                (None, None) => r.nonlinear_sample_count == 0,
                (Some(residual), Some(increment)) => {
                    let count = if r.accepted_sample_count == 1 {
                        1
                    } else {
                        r.accepted_sample_count - 1
                    };
                    r.nonlinear_sample_count as u64 == count
                        && [residual, increment].iter().all(|value| {
                            value.is_finite() && *value >= 0.0 && *value <= r.residual_target
                        })
                }
                _ => false,
            }
            || !r.maximum_relative_energy_residual.is_finite()
            || r.maximum_relative_energy_residual < 0.0
            || r.maximum_relative_energy_residual > r.residual_target
            || !r.reference_temperature_k.canonical_value().is_finite()
            || r.reference_temperature_k.canonical_value() <= 0.0
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
        if self.source_workload != workload.identity()?
            || self.source_result != result.identity()
            || self.receipt != evidence(workload, result.metadata())?
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
    if metadata.residual_target != workload.residual_target
        || metadata.source_workload != workload.identity()?
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
    let nonlinear = workload.model.boundaries.iter().any(|boundary| {
        matches!(
            boundary.operator,
            runmat_analysis_core::BoundaryOperator::ThermalRadiation { .. }
        )
    });
    if (nonlinear && metadata.nonlinear_evidence.len() != schedule.solve_count())
        || (!nonlinear && !metadata.nonlinear_evidence.is_empty())
    {
        return Err(invalid(
            "nonlinear accepted evidence differs from original boundary law",
        ));
    }
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
            .map(|domain| domain.reference_temperature.canonical_value())
            != Some(metadata.reference_temperature_k.canonical_value())
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
        first_sample_time_s: runmat_types::physical::Time::new(schedule.points_s()[0]),
        terminal_sample_time_s: runmat_types::physical::Time::new(
            *schedule.points_s().last().unwrap(),
        ),
        maximum_relative_linear_residual: maximum,
        nonlinear_sample_count: metadata.nonlinear_evidence.len(),
        maximum_relative_nonlinear_residual: nonlinear.then(|| {
            metadata
                .nonlinear_evidence
                .iter()
                .map(|evidence| evidence.relative_equilibrium_residual)
                .fold(0.0_f64, f64::max)
        }),
        maximum_relative_nonlinear_increment: nonlinear.then(|| {
            metadata
                .nonlinear_evidence
                .iter()
                .map(|evidence| evidence.relative_increment)
                .fold(0.0_f64, f64::max)
        }),
        maximum_relative_energy_residual: metadata
            .energy_balances
            .iter()
            .map(|balance| balance.relative_residual)
            .fold(0.0_f64, f64::max),
        residual_target: workload.residual_target,
        reference_temperature_k: runmat_types::physical::Temperature::new(
            metadata.reference_temperature_k.canonical_value(),
        ),
    })
}

#[cfg(test)]
mod tests;
