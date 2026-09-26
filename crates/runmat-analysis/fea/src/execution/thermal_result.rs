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
const LIMITS: CanonicalLimits = CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64);

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
            residual_target: self.residual_target,
            energy_balances: &self.energy_balances,
            reference_temperature_k: self.reference_temperature_k,
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
        self.metadata().identity(&fields)
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
    pub residual_target: f64,
    pub energy_balances: &'a [crate::contracts::ThermalEnergyBalance],
    pub reference_temperature_k: f64,
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
            || !self.reference_temperature_k.is_finite()
            || self.reference_temperature_k <= 0.0
            || self.initial_state.reference_temperature_k != self.reference_temperature_k
            || self.initial_state.procedure_step_id.trim().is_empty()
            || initial_nodes != Some(self.ordered_node_ids.len())
            || initial.authored_and_prescribed_node_count
                > initial
                    .authored_node_count
                    .min(initial.prescribed_node_count)
            || !initial.minimum_temperature_k.is_finite()
            || initial.minimum_temperature_k <= 0.0
            || !initial.maximum_temperature_k.is_finite()
            || initial.maximum_temperature_k < initial.minimum_temperature_k
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
        for balance in self.energy_balances {
            balance
                .validate()
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
            if balance.relative_residual > self.residual_target {
                return Err(AnalysisExecutionError::Invalid(
                    "thermal energy residual exceeds retained acceptance target".into(),
                ));
            }
        }
        Ok(())
    }
}
