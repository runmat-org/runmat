//! One definition inventory for thermal history planning and evaluation.
use crate::results::field::{
    RecoveredFieldDefinition, RecoveredFieldEntityDomain, RecoveredFieldError,
    RecoveredFieldNumericRepresentation,
};
use runmat_analysis_core::{FieldLocation, ObservationQuantity, ObservationValueShape};

pub(crate) fn definitions() -> Result<Vec<RecoveredFieldDefinition>, RecoveredFieldError> {
    [
        (
            "thermal.temperature",
            ObservationQuantity::Temperature,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::StructuralNodes,
            ObservationValueShape::Scalar,
        ),
        (
            "thermal.temperature_gradient",
            ObservationQuantity::TemperatureGradient,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::SolidElements,
            ObservationValueShape::Vector3,
        ),
        (
            "thermal.heat_flux",
            ObservationQuantity::HeatFlux,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::SolidElements,
            ObservationValueShape::Vector3,
        ),
        (
            "thermal.energy_storage_rate",
            ObservationQuantity::ThermalEnergyStorageRateDensity,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::SolidElements,
            ObservationValueShape::Scalar,
        ),
    ]
    .into_iter()
    .map(|(id, quantity, location, domain, shape)| {
        RecoveredFieldDefinition::new(
            id,
            quantity,
            location,
            domain,
            shape,
            RecoveredFieldNumericRepresentation::Real,
        )
    })
    .collect()
}
