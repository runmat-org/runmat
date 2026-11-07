use crate::assembly::{
    elements::solid::response::{response_from_strain, SolidResponseError},
    SolidRecoveryElementSummary,
};

use super::{StructuralFieldRecoveryError, COMPONENT_COUNT};

pub(super) fn recover_element_stress(
    element: &SolidRecoveryElementSummary,
    strain: [f64; COMPONENT_COUNT],
    eigenstrain: f64,
) -> Result<[f64; COMPONENT_COUNT], StructuralFieldRecoveryError> {
    response_from_strain(element.material, strain, eigenstrain)
        .map(|response| response.stress)
        .map_err(|error| response_error(element, error))
}

pub(super) fn response_error(
    element: &SolidRecoveryElementSummary,
    error: SolidResponseError,
) -> StructuralFieldRecoveryError {
    match error {
        SolidResponseError::Accounting(error) => StructuralFieldRecoveryError::Accounting(error),
        SolidResponseError::InvalidMaterial => StructuralFieldRecoveryError::InvalidMaterial {
            element_id: element.element_id.clone(),
        },
        SolidResponseError::InvalidInput => StructuralFieldRecoveryError::NonFiniteResult {
            element_id: element.element_id.clone(),
        },
    }
}

pub(super) fn thermal_eigenstrain_at(element: &SolidRecoveryElementSummary, point: usize) -> f64 {
    element.solved_thermal_quadrature.as_ref().map_or_else(
        || element.thermal_eigenstrain.unwrap_or(0.0),
        |field| field.eigenstrain[point],
    )
}
