use super::super::definition_validation::pcurve_domain;
use super::super::{
    ExactPcurveDefinition, ExactPcurveEvaluator, ExactPcurveImplementation,
    GeometryEvaluationControl, GeometryEvaluationError, ParameterRange, PcurveDerivatives,
    PcurveEvaluatorId,
};
use super::curve::trigonometric_curve;
use super::spline::rational_derivatives;
use super::vector::add_scaled;
use super::{invalid_result, kernel_owned, outside_domain, PortableExactEvaluator};

impl ExactPcurveEvaluator for PortableExactEvaluator<'_> {
    fn parameter_range(
        &self,
        id: &PcurveEvaluatorId,
    ) -> Result<ParameterRange, GeometryEvaluationError> {
        let record = self.pcurve_record(id)?;
        pcurve_domain(&record.implementation).ok_or_else(|| kernel_owned("pcurve"))
    }

    fn point(
        &self,
        id: &PcurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 2], GeometryEvaluationError> {
        Ok(ExactPcurveEvaluator::derivatives(self, id, parameter, control)?.point_uv)
    }

    fn derivatives(
        &self,
        id: &PcurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<PcurveDerivatives, GeometryEvaluationError> {
        let record = self.pcurve_record(id)?;
        let ExactPcurveImplementation::Portable { definition } = &record.implementation else {
            return Err(kernel_owned("pcurve"));
        };
        evaluate_pcurve(definition, parameter, control)
    }
}

pub(super) fn evaluate_pcurve(
    definition: &ExactPcurveDefinition,
    parameter: f64,
    control: &dyn GeometryEvaluationControl,
) -> Result<PcurveDerivatives, GeometryEvaluationError> {
    require_parameter(parameter, definition_range(definition))?;
    control.checkpoint()?;
    control.consume_iterations(1)?;
    let result = match definition {
        ExactPcurveDefinition::Line {
            origin_uv,
            direction_uv_per_parameter,
            ..
        } => PcurveDerivatives {
            point_uv: add_scaled(
                origin_uv,
                direction_uv_per_parameter,
                parameter,
                &[0.0; 2],
                0.0,
            ),
            first_uv: *direction_uv_per_parameter,
            second_uv: [0.0; 2],
        },
        ExactPcurveDefinition::Circle {
            center_uv,
            x_axis_uv,
            y_axis_uv,
            radius_uv,
            ..
        } => {
            let value = trigonometric_curve(
                center_uv, x_axis_uv, y_axis_uv, *radius_uv, *radius_uv, parameter,
            );
            PcurveDerivatives {
                point_uv: value.0,
                first_uv: value.1,
                second_uv: value.2,
            }
        }
        ExactPcurveDefinition::Nurbs { definition } => {
            let value = rational_derivatives(
                definition.degree,
                &definition.knots,
                &definition.control_points_uv,
                &definition.weights,
                parameter,
                control,
            )?;
            PcurveDerivatives {
                point_uv: value.point,
                first_uv: value.first,
                second_uv: value.second,
            }
        }
    };
    if result
        .point_uv
        .iter()
        .chain(&result.first_uv)
        .chain(&result.second_uv)
        .any(|value| !value.is_finite())
    {
        return Err(invalid_result(
            "pcurve evaluation produced a non-finite result",
        ));
    }
    Ok(result)
}

fn definition_range(definition: &ExactPcurveDefinition) -> ParameterRange {
    match definition {
        ExactPcurveDefinition::Line { domain, .. }
        | ExactPcurveDefinition::Circle { domain, .. } => *domain,
        ExactPcurveDefinition::Nurbs { definition } => definition.domain,
    }
}

fn require_parameter(
    parameter: f64,
    domain: ParameterRange,
) -> Result<(), GeometryEvaluationError> {
    if !parameter.is_finite() || parameter < domain.start || parameter > domain.end {
        return Err(outside_domain(
            "pcurve parameter lies outside the admitted domain",
        ));
    }
    Ok(())
}
