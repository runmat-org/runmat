use runmat_geometry_core::{
    BodyMassProperties, ExactMassPropertiesEvaluator, GeometryEvaluationControl,
    GeometryEvaluationError, GeometryEvaluationErrorKind, MassPropertiesEvaluatorId,
};

use super::{evaluator::OcctExactEvaluator, evaluator_bindings::MassPropertiesBinding, ffi};

impl ExactMassPropertiesEvaluator for OcctExactEvaluator {
    fn mass_properties(
        &self,
        id: &MassPropertiesEvaluatorId,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<BodyMassProperties, GeometryEvaluationError> {
        let binding = self
            .bindings
            .mass_properties
            .get(id)
            .cloned()
            .ok_or_else(|| {
                GeometryEvaluationError::new(
                    GeometryEvaluationErrorKind::UnknownEvaluator,
                    format!("unknown OCCT mass-properties evaluator {}", id.as_str()),
                )
            })?;
        control.checkpoint()?;
        control.consume_iterations(1)?;
        let properties = match binding {
            MassPropertiesBinding::Validated(properties) => properties,
            MassPropertiesBinding::Kernel {
                shape_keys,
                is_sheet_body,
            } => {
                let value =
                    ffi::bridge::exact_mass_properties(self.session_id, &shape_keys, is_sheet_body)
                        .map_err(|error| kernel(error.to_string()))?;
                BodyMassProperties {
                    volume_m3: value.volume,
                    surface_area_m2: value.surface_area,
                    centroid_m: [value.centroid_x, value.centroid_y, value.centroid_z],
                    inertia_about_centroid_m5: [
                        value.inertia_xx,
                        value.inertia_yy,
                        value.inertia_zz,
                        value.inertia_xy,
                        value.inertia_xz,
                        value.inertia_yz,
                    ],
                }
            }
        };
        control.checkpoint()?;
        validate(properties)
    }
}

fn validate(properties: BodyMassProperties) -> Result<BodyMassProperties, GeometryEvaluationError> {
    properties.validate().map_err(|error| {
        GeometryEvaluationError::new(
            GeometryEvaluationErrorKind::InvalidResult,
            error.to_string(),
        )
    })?;
    Ok(properties)
}

fn kernel(reason: impl Into<String>) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::KernelFailure, reason)
}
