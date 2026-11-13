//! Total stored elastic energy after thermal eigenstrain has been removed from solid strain.

use runmat_analysis_core::AnalysisField;

use crate::{assembly::AssemblySummary, contracts::FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY};

use super::StructuralFieldRecoveryError;
use crate::algebra::{capacity::measured, coercivity::NumericalCapacityError};
use runmat_simulation_usage::{AsyncCapacityOwner, OperationCount, UsageCategory};

pub(super) async fn total_elastic_energy_with_capacity<A>(
    summary: &AssemblySummary,
    fields: &[AnalysisField],
    owner: Option<&dyn AsyncCapacityOwner<A>>,
) -> Result<f64, NumericalCapacityError<StructuralFieldRecoveryError, A>> {
    let density = fields
        .iter()
        .find(|field| field.field_id == FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY)
        .and_then(AnalysisField::as_host_f64)
        .ok_or(StructuralFieldRecoveryError::MissingThermoelasticEnergyDensity)?;
    if density.len() != summary.structural_solid_recovery.len() {
        return Err(StructuralFieldRecoveryError::ElementValueCountMismatch {
            expected: summary.structural_solid_recovery.len(),
            actual: density.len(),
        }
        .into());
    }
    let mut total = 0.0;
    for (element, energy_density) in summary.structural_solid_recovery.iter().zip(density) {
        let volume = element
            .geometry
            .volume_m3_with_capacity(owner)
            .await
            .map_err(|error| {
                error.map_numerical(|error| {
                    StructuralFieldRecoveryError::geometry(&element.element_id, error)
                })
            })?;
        total = measured(
            owner,
            UsageCategory::Solve,
            OperationCount::from_operations(2),
            || {
                let energy = total + energy_density * volume;
                if energy.is_finite() {
                    Ok(energy)
                } else {
                    Err(StructuralFieldRecoveryError::NonFiniteResult {
                        element_id: element.element_id.clone(),
                    })
                }
            },
        )
        .await?;
    }
    Ok(total)
}
