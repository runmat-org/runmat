use super::super::definition_validation::pcurve_domain;
use super::super::{
    ExactPcurveDefinitionV2, ExactPcurveEvaluatorV2, ExactPcurveImplementationV2,
    GeometryEvaluationControl, GeometryEvaluationError, ParameterRangeV2, PcurveDerivativesV2,
    PcurveEvaluatorIdV2,
};
use super::curve::trigonometric_curve;
use super::spline::rational_derivatives;
use super::vector::add_scaled;
use super::{invalid_result, kernel_owned, outside_domain, PortableExactEvaluatorV2};

impl ExactPcurveEvaluatorV2 for PortableExactEvaluatorV2<'_> {
    fn parameter_range(
        &self,
        id: &PcurveEvaluatorIdV2,
    ) -> Result<ParameterRangeV2, GeometryEvaluationError> {
        let record = self.pcurve_record(id)?;
        pcurve_domain(&record.implementation).ok_or_else(|| kernel_owned("pcurve"))
    }

    fn point(
        &self,
        id: &PcurveEvaluatorIdV2,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 2], GeometryEvaluationError> {
        Ok(ExactPcurveEvaluatorV2::derivatives(self, id, parameter, control)?.point_uv)
    }

    fn derivatives(
        &self,
        id: &PcurveEvaluatorIdV2,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<PcurveDerivativesV2, GeometryEvaluationError> {
        let record = self.pcurve_record(id)?;
        let ExactPcurveImplementationV2::Portable { definition } = &record.implementation else {
            return Err(kernel_owned("pcurve"));
        };
        evaluate_pcurve(definition, parameter, control)
    }
}

fn evaluate_pcurve(
    definition: &ExactPcurveDefinitionV2,
    parameter: f64,
    control: &dyn GeometryEvaluationControl,
) -> Result<PcurveDerivativesV2, GeometryEvaluationError> {
    require_parameter(parameter, definition_range(definition))?;
    control.checkpoint()?;
    control.consume_iterations(1)?;
    let result = match definition {
        ExactPcurveDefinitionV2::Line {
            origin_uv,
            direction_uv_per_parameter,
            ..
        } => PcurveDerivativesV2 {
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
        ExactPcurveDefinitionV2::Circle {
            center_uv,
            x_axis_uv,
            y_axis_uv,
            radius_uv,
            ..
        } => {
            let value = trigonometric_curve(
                center_uv, x_axis_uv, y_axis_uv, *radius_uv, *radius_uv, parameter,
            );
            PcurveDerivativesV2 {
                point_uv: value.0,
                first_uv: value.1,
                second_uv: value.2,
            }
        }
        ExactPcurveDefinitionV2::Nurbs { definition } => {
            let value = rational_derivatives(
                definition.degree,
                &definition.knots,
                &definition.control_points_uv,
                &definition.weights,
                parameter,
                control,
            )?;
            PcurveDerivativesV2 {
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

fn definition_range(definition: &ExactPcurveDefinitionV2) -> ParameterRangeV2 {
    match definition {
        ExactPcurveDefinitionV2::Line { domain, .. }
        | ExactPcurveDefinitionV2::Circle { domain, .. } => *domain,
        ExactPcurveDefinitionV2::Nurbs { definition } => definition.domain,
    }
}

fn require_parameter(
    parameter: f64,
    domain: ParameterRangeV2,
) -> Result<(), GeometryEvaluationError> {
    if !parameter.is_finite() || parameter < domain.start || parameter > domain.end {
        return Err(outside_domain(
            "pcurve parameter lies outside the admitted domain",
        ));
    }
    Ok(())
}
