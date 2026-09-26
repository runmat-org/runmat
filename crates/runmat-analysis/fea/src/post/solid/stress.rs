use crate::assembly::SolidRecoveryElementSummary;

use super::{StructuralFieldRecoveryError, COMPONENT_COUNT};

pub(super) fn recover_element_stress(
    element: &SolidRecoveryElementSummary,
    strain: [f64; COMPONENT_COUNT],
    eigenstrain: f64,
) -> Result<[f64; COMPONENT_COUNT], StructuralFieldRecoveryError> {
    let lambda = element.material.lame_lambda_pa().map_err(|_| {
        StructuralFieldRecoveryError::InvalidMaterial {
            element_id: element.element_id.clone(),
        }
    })?;
    let mu = element.material.shear_modulus_pa().map_err(|_| {
        StructuralFieldRecoveryError::InvalidMaterial {
            element_id: element.element_id.clone(),
        }
    })?;
    let mechanical_normal_strain = [
        strain[0] - eigenstrain,
        strain[1] - eigenstrain,
        strain[2] - eigenstrain,
    ];
    let trace = mechanical_normal_strain.iter().sum::<f64>();
    let stress = [
        lambda * trace + 2.0 * mu * mechanical_normal_strain[0],
        lambda * trace + 2.0 * mu * mechanical_normal_strain[1],
        lambda * trace + 2.0 * mu * mechanical_normal_strain[2],
        mu * strain[3],
        mu * strain[4],
        mu * strain[5],
    ];
    if stress.iter().all(|value| value.is_finite()) {
        Ok(stress)
    } else {
        Err(StructuralFieldRecoveryError::NonFiniteResult {
            element_id: element.element_id.clone(),
        })
    }
}

pub(super) fn thermal_eigenstrain_at(element: &SolidRecoveryElementSummary, point: usize) -> f64 {
    element.solved_thermal_quadrature.as_ref().map_or_else(
        || element.thermal_eigenstrain.unwrap_or(0.0),
        |field| field.eigenstrain[point],
    )
}
