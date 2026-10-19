//! Total stored elastic energy after thermal eigenstrain has been removed from solid strain.

use runmat_analysis_core::AnalysisField;

use crate::{assembly::AssemblySummary, contracts::FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY};

use super::StructuralFieldRecoveryError;

pub(super) fn total_elastic_energy(
    summary: &AssemblySummary,
    fields: &[AnalysisField],
) -> Result<f64, StructuralFieldRecoveryError> {
    let density = fields
        .iter()
        .find(|field| field.field_id == FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY)
        .and_then(AnalysisField::as_host_f64)
        .ok_or(StructuralFieldRecoveryError::MissingThermoelasticEnergyDensity)?;
    if density.len() != summary.structural_solid_recovery.len() {
        return Err(StructuralFieldRecoveryError::ElementValueCountMismatch {
            expected: summary.structural_solid_recovery.len(),
            actual: density.len(),
        });
    }
    summary
        .structural_solid_recovery
        .iter()
        .zip(density)
        .try_fold(0.0, |total, (element, energy_density)| {
            let volume = element.geometry.volume_m3().map_err(|_| {
                StructuralFieldRecoveryError::NonFiniteResult {
                    element_id: element.element_id.clone(),
                }
            })?;
            let energy = total + energy_density * volume;
            if energy.is_finite() {
                Ok(energy)
            } else {
                Err(StructuralFieldRecoveryError::NonFiniteResult {
                    element_id: element.element_id.clone(),
                })
            }
        })
}
