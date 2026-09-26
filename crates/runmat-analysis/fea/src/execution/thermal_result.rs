//! Portable thermal result validation, independent of runtime storage and caller context.
use super::state::field_validation::validate_portable_fields;
use super::{AnalysisExecutionError, AnalysisExecutionResult, SolverResultIdentity};
use crate::contracts::{
    fea_thermal_energy_storage_rate_field_id, fea_thermal_heat_flux_field_id,
    fea_thermal_temperature_field_id, fea_thermal_temperature_gradient_field_id,
    FeaThermalRunResult,
};
use crate::diagnostics::FeaDiagnosticSeverity;
use runmat_analysis_core::{AnalysisField, AnalysisProcedure, PhysicsFamily};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use std::collections::BTreeSet;

const PREFIX: &[u8] = b"runmat-thermal-solver-result/v1\0";
const LIMITS: CanonicalLimits = CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64);

impl FeaThermalRunResult {
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
        // Numerical identity excludes presentation diagnostics and execution telemetry.
        let content = encode_contract(
            PREFIX,
            "thermal-solver-result-content",
            &(
                &self.run.identity,
                &self.time_points_s,
                &self.temperature_snapshots,
                &self.temperature_gradient_snapshots,
                &self.heat_flux_snapshots,
                &self.energy_storage_rate_snapshots,
                &self.residual_norms,
                self.reference_temperature_k,
                self.topology_digest,
                self.producer,
                &self.ordered_node_ids,
                &self.ordered_element_ids,
                &self.initial_state,
            ),
            LIMITS,
        )?;
        SolverResultIdentity::from_step_result(self.producer, Digest::sha256(content))
    }
    /// Checks structural consistency and accepted solver evidence, not independent accuracy.
    pub fn validate_complete(&self) -> AnalysisExecutionResult<()> {
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
            || !unique_nonempty(&self.ordered_node_ids)
            || !unique_nonempty(&self.ordered_element_ids)
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
            || self.time_points_s.iter().any(|x| !x.is_finite())
            || self.time_points_s.windows(2).any(|pair| pair[1] <= pair[0])
            || self.residual_norms.len() != residuals
            || self
                .residual_norms
                .iter()
                .any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err(invalid());
        }
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
        validate_portable_fields(&self.run.fields)?;
        for code in ["FEA_THERMAL_STABILITY", "FEA_THERMAL_ENERGY_BALANCE"] {
            let mut matching = self.run.diagnostics.iter().filter(|item| item.code == code);
            if !matching
                .next()
                .is_some_and(|item| item.severity == FeaDiagnosticSeverity::Info)
                || matching.next().is_some()
            {
                return Err(AnalysisExecutionError::Invalid(format!(
                    "thermal result lacks unique accepted {code} evidence"
                )));
            }
        }
        Ok(())
    }
}
fn unique_nonempty(ids: &[u64]) -> bool {
    !ids.is_empty() && ids.iter().collect::<BTreeSet<_>>().len() == ids.len()
}
fn validate_history(
    fields: &[AnalysisField],
    samples: usize,
    shape: &[usize],
    field_id: fn(usize) -> String,
) -> AnalysisExecutionResult<()> {
    validate_portable_fields(fields)?;
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
