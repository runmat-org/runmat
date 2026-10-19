//! One definition inventory for thermal history planning and evaluation.
use crate::results::field::{
    RecoveredFieldDefinition, RecoveredFieldEntityDomain, RecoveredFieldError,
    RecoveredFieldNumericRepresentation,
};
use runmat_analysis_core::{FieldLocation, ObservationQuantity, ObservationValueShape};

/// A validated steady solve has one snapshot; transient schedules have at least two.
/// Preserve the shared final-field coordinate instead of inventing a time domain for steady heat.
pub(crate) fn sample(
    sample_count: usize,
    time: f64,
) -> crate::results::field::RecoveredFieldSample {
    if sample_count == 1 {
        crate::results::field::RecoveredFieldSample::Final
    } else {
        crate::results::field::RecoveredFieldSample::TimeSeconds(time)
    }
}

/// Same ordinal-derived field definition used by publication and preassembly profiling.
pub(crate) fn snapshot_definition(
    index: usize,
    ordinal: usize,
) -> Result<RecoveredFieldDefinition, RecoveredFieldError> {
    let template = TEMPLATES
        .get(index)
        .ok_or(RecoveredFieldError::StorageShapeMismatch)?;
    let ids = [
        crate::contracts::fea_thermal_temperature_field_id,
        crate::contracts::fea_thermal_temperature_gradient_field_id,
        crate::contracts::fea_thermal_heat_flux_field_id,
        crate::contracts::fea_thermal_energy_storage_rate_field_id,
    ];
    RecoveredFieldDefinition::new_derived(
        ids[index](ordinal),
        template.1,
        template.2,
        template.3,
        template.4.clone(),
        RecoveredFieldNumericRepresentation::Real,
        crate::results::field::RecoveredFieldDerivation::Native,
    )
}

static TEMPLATES: [(
    &str,
    ObservationQuantity,
    FieldLocation,
    RecoveredFieldEntityDomain,
    ObservationValueShape,
); 4] = [
    (
        "thermal.temperature",
        ObservationQuantity::Temperature,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Scalar,
    ),
    (
        "thermal.temperature_gradient",
        ObservationQuantity::TemperatureGradient,
        FieldLocation::Element,
        RecoveredFieldEntityDomain::VolumeElements,
        ObservationValueShape::Vector3,
    ),
    (
        "thermal.heat_flux",
        ObservationQuantity::HeatFlux,
        FieldLocation::Element,
        RecoveredFieldEntityDomain::VolumeElements,
        ObservationValueShape::Vector3,
    ),
    (
        "thermal.energy_storage_rate",
        ObservationQuantity::ThermalEnergyStorageRateDensity,
        FieldLocation::Element,
        RecoveredFieldEntityDomain::VolumeElements,
        ObservationValueShape::Scalar,
    ),
];

pub(crate) fn definitions() -> Result<Vec<RecoveredFieldDefinition>, RecoveredFieldError> {
    TEMPLATES
        .iter()
        .map(|(id, quantity, location, domain, shape)| {
            RecoveredFieldDefinition::new(
                *id,
                *quantity,
                *location,
                *domain,
                shape.clone(),
                RecoveredFieldNumericRepresentation::Real,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn snapshot_matches_original_catalog_template_with_less_requested_heap() {
        let ids = [
            crate::contracts::fea_thermal_temperature_field_id,
            crate::contracts::fea_thermal_temperature_gradient_field_id,
            crate::contracts::fea_thermal_heat_flux_field_id,
            crate::contracts::fea_thermal_energy_storage_rate_field_id,
        ];
        for ordinal in [0, 16, usize::MAX] {
            for (index, id) in ids.iter().enumerate() {
                let (actual, _, peak) = crate::topology::measure_test_allocations(|| {
                    snapshot_definition(index, ordinal).unwrap()
                });
                let (expected, _, old_peak) = crate::topology::measure_test_allocations(|| {
                    let catalog = definitions().unwrap();
                    let template = &catalog[index];
                    RecoveredFieldDefinition::new_derived(
                        id(ordinal),
                        template.quantity(),
                        template.location(),
                        template.entity_domain(),
                        template.value_shape().clone(),
                        template.numeric_representation(),
                        template.derivation(),
                    )
                    .unwrap()
                });
                assert_eq!(actual, expected);
                assert!(peak < old_peak);
            }
        }
        assert!(snapshot_definition(4, 0).is_err());
    }
}
