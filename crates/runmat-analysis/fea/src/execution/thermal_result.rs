//! Portable thermal result validation, independent of runtime storage and caller context.
use super::state::field_validation::{validate_portable_field, validate_portable_fields};
use super::{AnalysisExecutionError, AnalysisExecutionResult, SolverResultIdentity};
use crate::contracts::{
    fea_thermal_energy_storage_rate_field_id, fea_thermal_heat_flux_field_id,
    fea_thermal_temperature_field_id, fea_thermal_temperature_gradient_field_id,
    FeaThermalRunResult,
};
use runmat_analysis_core::{AnalysisField, AnalysisProcedure, PhysicsFamily};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;

const PREFIX: &[u8] = b"runmat-thermal-solver-result/v1\0";
const LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[path = "thermal_result_fields.rs"]
mod fields;
pub(in crate::execution) mod identity;

#[cfg(test)]
#[path = "thermal_result_validation_tests.rs"]
mod tests;

impl FeaThermalRunResult {
    pub(super) fn metadata(&self) -> ThermalResultMetadata<'_> {
        ThermalResultMetadata {
            source_workload: self.source_workload,
            numerical_intent: self.numerical_intent,
            run: &self.run,
            time_points_s: &self.time_points_s,
            residual_norms: &self.residual_norms,
            nonlinear_evidence: &self.nonlinear_evidence,
            residual_target: self.residual_target,
            energy_balances: &self.energy_balances,
            reference_temperature_k: runmat_types::physical::Temperature::new(
                self.reference_temperature_k.canonical_value(),
            ),
            topology_digest: self.topology_digest,
            producer: self.producer,
            ordered_node_ids: &self.ordered_node_ids,
            ordered_element_ids: &self.ordered_element_ids,
            initial_state: &self.initial_state,
        }
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate_complete()?;
        Ok(encode_contract(
            PREFIX,
            "thermal-solver-result",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let result: Self = decode_contract(PREFIX, "thermal-solver-result", bytes, LIMITS)?;
        result.validate_complete()?;
        Ok(result)
    }
    pub fn source_result_identity(&self) -> AnalysisExecutionResult<SolverResultIdentity> {
        self.validate_complete()?;
        let fields = fields::retained_field_identities(self)?;
        self.metadata()
            .identity(super::thermal_result::identity::FieldIdentities {
                samples: &fields,
                interface_node_ids: &self.interface_node_ids,
                identities: super::thermal_result::identity::retained_identities,
            })
    }
    /// Checks structural consistency and accepted solver evidence, not independent accuracy.
    pub fn validate_complete(&self) -> AnalysisExecutionResult<()> {
        self.metadata().validate()?;
        let samples = self.time_points_s.len();
        let invalid =
            || AnalysisExecutionError::Invalid("thermal result has invalid temperatures".into());
        validate_history(
            &self.temperature_snapshots,
            samples,
            &[self.ordered_node_ids.len()],
            fea_thermal_temperature_field_id,
        )?;
        if self.interface_node_ids.is_empty() {
            if !self.interface_temperature_snapshots.is_empty() {
                return Err(invalid());
            }
        } else {
            validate_history(
                &self.interface_temperature_snapshots,
                samples,
                &[self.interface_node_ids.len()],
                crate::contracts::fea_thermal_interface_temperature_field_id,
            )?;
        }
        validate_history(
            &self.temperature_gradient_snapshots,
            samples,
            &[self.ordered_element_ids.len(), 3],
            fea_thermal_temperature_gradient_field_id,
        )?;
        validate_history(
            &self.heat_flux_snapshots,
            samples,
            &[self.ordered_element_ids.len(), 3],
            fea_thermal_heat_flux_field_id,
        )?;
        validate_history(
            &self.energy_storage_rate_snapshots,
            samples,
            &[self.ordered_element_ids.len()],
            fea_thermal_energy_storage_rate_field_id,
        )?;
        if self
            .temperature_snapshots
            .iter()
            .chain(&self.interface_temperature_snapshots)
            .any(|field| field.as_host_f64().unwrap().iter().any(|x| *x <= 0.0))
        {
            return Err(invalid());
        }
        Ok(())
    }
}
fn unique_nonempty(ids: &[u64]) -> AnalysisExecutionResult<bool> {
    if ids.is_empty() {
        return Ok(false);
    }
    // Canonical producer order is usually ascending. Arbitrary unique declared order remains
    // supported without mutating that order or imposing quadratic duplicate detection.
    if ids.windows(2).all(|pair| pair[0] < pair[1]) {
        return Ok(true);
    }
    let mut sorted = Vec::new();
    sorted.try_reserve_exact(ids.len()).map_err(|error| {
        AnalysisExecutionError::Invalid(format!(
            "cannot reserve thermal entity validation: {error}"
        ))
    })?;
    sorted.extend_from_slice(ids);
    sorted.sort_unstable();
    Ok(sorted.windows(2).all(|pair| pair[0] != pair[1]))
}
fn validate_history(
    fields: &[AnalysisField],
    samples: usize,
    shape: &[usize],
    field_id: fn(usize) -> String,
) -> AnalysisExecutionResult<()> {
    for field in fields {
        validate_portable_field(field)?;
    }
    // The exact ordinal-derived identities below imply uniqueness without a history-sized set.
    if fields.len() != samples
        || fields
            .iter()
            .enumerate()
            .any(|(index, field)| field.shape != shape || field.field_id != field_id(index))
    {
        return Err(AnalysisExecutionError::Invalid(
            "thermal result history has incorrect field identities, shapes, or sample count".into(),
        ));
    }
    Ok(())
}

pub(super) struct ThermalResultMetadata<'a> {
    pub source_workload: super::AnalysisWorkloadIdentity,
    pub numerical_intent: super::ThermalNumericalIntentIdentity,
    pub run: &'a crate::FeaRunResult,
    pub time_points_s: &'a [f64],
    pub residual_norms: &'a [f64],
    pub nonlinear_evidence: &'a [crate::contracts::ThermalNonlinearSampleEvidence],
    pub residual_target: f64,
    pub energy_balances: &'a [crate::contracts::ThermalEnergyBalance],
    pub reference_temperature_k: runmat_types::physical::Temperature,
    pub topology_digest: Digest,
    pub producer: super::StepIdentity,
    pub ordered_node_ids: &'a [u64],
    pub ordered_element_ids: &'a [u64],
    pub initial_state: &'a crate::ThermalInitialStateEvidence,
}
impl ThermalResultMetadata<'_> {
    pub(super) fn validate(&self) -> AnalysisExecutionResult<()> {
        let invalid = || {
            AnalysisExecutionError::Invalid(
                "thermal result has incomplete or inconsistent history/provenance".into(),
            )
        };
        self.run
            .validate_solver_formulation()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let samples = self.time_points_s.len();
        let residuals = match self.run.identity.procedure() {
            AnalysisProcedure::Static if samples == 1 => 1,
            AnalysisProcedure::TransientResponse if samples >= 2 => samples - 1,
            _ => return Err(invalid()),
        };
        let initial = &self.initial_state;
        let initial_nodes = initial
            .authored_node_count
            .checked_add(initial.prescribed_node_count)
            .and_then(|count| count.checked_sub(initial.authored_and_prescribed_node_count))
            .and_then(|count| count.checked_add(initial.reference_node_count));
        if self.run.identity.family() != PhysicsFamily::Thermal
            || !unique_nonempty(self.ordered_node_ids)?
            || !unique_nonempty(self.ordered_element_ids)?
            || !self.reference_temperature_k.canonical_value().is_finite()
            || self.reference_temperature_k.canonical_value() <= 0.0
            || self.initial_state.reference_temperature_k.canonical_value()
                != self.reference_temperature_k.canonical_value()
            || self.initial_state.procedure_step_id.trim().is_empty()
            || initial_nodes != Some(self.ordered_node_ids.len())
            || initial.authored_and_prescribed_node_count
                > initial
                    .authored_node_count
                    .min(initial.prescribed_node_count)
            || !initial.minimum_temperature_k.canonical_value().is_finite()
            || initial.minimum_temperature_k.canonical_value() <= 0.0
            || !initial.maximum_temperature_k.canonical_value().is_finite()
            || initial.maximum_temperature_k.canonical_value()
                < initial.minimum_temperature_k.canonical_value()
            || self
                .time_points_s
                .iter()
                .any(|x| !x.is_finite() || x.to_bits() == (-0.0_f64).to_bits())
            || self.time_points_s.windows(2).any(|pair| pair[1] <= pair[0])
            || self.residual_norms.len() != residuals
            || self.energy_balances.len() != residuals
            || !self.residual_target.is_finite()
            || self.residual_target <= 0.0
            || self
                .residual_norms
                .iter()
                .any(|x| !x.is_finite() || *x < 0.0 || *x > self.residual_target)
        {
            return Err(invalid());
        }
        validate_portable_fields(&self.run.fields)?;
        let nonlinear = self
            .run
            .identity
            .realization()
            .temporals
            .iter()
            .any(|temporal| {
                matches!(temporal,
            runmat_analysis_core::TemporalImplementation::ThermalSteadyDampedNewton
                | runmat_analysis_core::TemporalImplementation::ThermalBackwardEulerDampedNewton)
            });
        if nonlinear == self.nonlinear_evidence.is_empty()
            || (nonlinear
                && (self.run.execution_backend != crate::ExecutionBackend::Cpu
                    || (self.run.requested_backend != crate::ComputeBackend::Cpu
                        && !self.run.diagnostics.iter().any(|diagnostic| {
                            diagnostic.code
                                == crate::contracts::accelerator_admission::ACCELERATOR_ADMISSION_FALLBACK_CODE
                        }))))
        {
            return Err(invalid());
        }
        if !self.nonlinear_evidence.is_empty() {
            if self.nonlinear_evidence.len() != residuals {
                return Err(invalid());
            }
            let first =
                usize::from(self.run.identity.procedure() == AnalysisProcedure::TransientResponse);
            for (evidence, time) in self
                .nonlinear_evidence
                .iter()
                .zip(&self.time_points_s[first..])
            {
                if evidence.time_s.canonical_value() != *time
                    || !evidence.satisfies(self.residual_target)
                {
                    return Err(invalid());
                }
            }
        }
        for balance in self.energy_balances {
            balance
                .validate()
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
            if balance.relative_residual > self.residual_target {
                return Err(AnalysisExecutionError::Invalid(format!(
                    "thermal energy residual {} exceeds retained acceptance target {}; residual={} W, storage={} W, boundary loss={} W, applied={} W, prescribed={} W, physical scale={} W",
                    balance.relative_residual,
                    self.residual_target,
                    balance.residual_w.canonical_value(),
                    balance.storage_rate_w.canonical_value(),
                    balance.boundary_loss_w.canonical_value(),
                    balance.applied_power_w.canonical_value(),
                    balance.prescribed_boundary_power_w.canonical_value(),
                    balance.scale.maximum().canonical_value(),
                )));
            }
        }
        Ok(())
    }
}
