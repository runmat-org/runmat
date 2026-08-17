use super::super::definition_validation::curve_domain;
use super::super::{
    CurveDerivatives, CurveEvaluatorId, CurveProjection, ExactCurveDefinition, ExactCurveEvaluator,
    ExactCurveImplementation, GeometryEvaluationControl, GeometryEvaluationError,
    GeometryEvaluationErrorKind, ParameterRange,
};
use super::integration::adaptive_arc_length;
use super::projection::{charge_seed_allocation, project_curve, uniform_seeds};
use super::spline::rational_derivatives;
use super::vector::{add_scaled, cross, distance, dot, norm, normalize, subtract};
use super::{invalid_result, kernel_owned, outside_domain, PortableExactEvaluator};

const MAX_NURBS_PROJECTION_SEEDS: usize = 1_000_000;

impl ExactCurveEvaluator for PortableExactEvaluator<'_> {
    fn parameter_range(
        &self,
        id: &CurveEvaluatorId,
    ) -> Result<ParameterRange, GeometryEvaluationError> {
        let record = self.curve_record(id)?;
        curve_domain(&record.implementation).ok_or_else(|| kernel_owned("curve"))
    }

    fn point(
        &self,
        id: &CurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError> {
        Ok(ExactCurveEvaluator::derivatives(self, id, parameter, control)?.point_m)
    }

    fn unit_tangent(
        &self,
        id: &CurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError> {
        normalize(&ExactCurveEvaluator::derivatives(self, id, parameter, control)?.first_m)
            .ok_or_else(|| invalid_result("curve tangent is undefined at a singular parameter"))
    }

    fn derivatives(
        &self,
        id: &CurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<CurveDerivatives, GeometryEvaluationError> {
        let record = self.curve_record(id)?;
        let ExactCurveImplementation::Portable { definition } = &record.implementation else {
            return Err(kernel_owned("curve"));
        };
        evaluate_curve(definition, parameter, control)
    }

    fn curvature_1_per_m(
        &self,
        id: &CurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<f64, GeometryEvaluationError> {
        let derivatives = ExactCurveEvaluator::derivatives(self, id, parameter, control)?;
        let speed = norm(&derivatives.first_m);
        if speed == 0.0 || !speed.is_finite() {
            return Err(invalid_result(
                "curve curvature is undefined at a singular parameter",
            ));
        }
        let curvature = norm(&cross(&derivatives.first_m, &derivatives.second_m)) / speed.powi(3);
        if !curvature.is_finite() {
            return Err(invalid_result("curve curvature is non-finite"));
        }
        Ok(curvature)
    }

    fn arc_length_m(
        &self,
        id: &CurveEvaluatorId,
        range: ParameterRange,
        absolute_error_m: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<f64, GeometryEvaluationError> {
        let record = self.curve_record(id)?;
        let ExactCurveImplementation::Portable { definition } = &record.implementation else {
            return Err(kernel_owned("curve"));
        };
        require_subrange(range, definition_range(definition))?;
        require_positive_error(absolute_error_m)?;
        control.checkpoint()?;
        match definition {
            ExactCurveDefinition::Line {
                direction_m_per_parameter,
                ..
            } => {
                control.consume_iterations(1)?;
                finite_length(norm(direction_m_per_parameter) * (range.end - range.start))
            }
            ExactCurveDefinition::Circle { radius_m, .. } => {
                control.consume_iterations(1)?;
                finite_length(radius_m * (range.end - range.start))
            }
            _ => adaptive_arc_length(range, absolute_error_m, control, |parameter| {
                Ok(norm(
                    &evaluate_curve(definition, parameter, control)?.first_m,
                ))
            }),
        }
    }

    fn inverse_project(
        &self,
        id: &CurveEvaluatorId,
        point_m: [f64; 3],
        absolute_error_m: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<CurveProjection, GeometryEvaluationError> {
        let record = self.curve_record(id)?;
        let ExactCurveImplementation::Portable { definition } = &record.implementation else {
            return Err(kernel_owned("curve"));
        };
        if point_m.iter().any(|value| !value.is_finite()) {
            return Err(invalid_result("curve projection point must be finite"));
        }
        require_positive_error(absolute_error_m)?;
        let range = definition_range(definition);
        if let ExactCurveDefinition::Line {
            origin_m,
            direction_m_per_parameter,
            ..
        } = definition
        {
            control.checkpoint()?;
            control.consume_search_work(1)?;
            let offset = subtract(&point_m, origin_m);
            let denominator = dot(direction_m_per_parameter, direction_m_per_parameter);
            let raw_parameter = dot(&offset, direction_m_per_parameter) / denominator;
            if !raw_parameter.is_finite() {
                return Err(invalid_result(
                    "line projection produced an invalid parameter",
                ));
            }
            let parameter = raw_parameter.clamp(range.start, range.end);
            let projected = add_scaled(
                origin_m,
                direction_m_per_parameter,
                parameter,
                &[0.0; 3],
                0.0,
            );
            let distance_m = distance(&projected, &point_m);
            if !distance_m.is_finite() {
                return Err(invalid_result(
                    "line projection produced an invalid distance",
                ));
            }
            return Ok(CurveProjection {
                parameter,
                point_m: projected,
                distance_m,
            });
        }
        let seeds = projection_seeds(definition, range)?;
        charge_seed_allocation(seeds.len(), std::mem::size_of::<f64>(), control)?;
        project_curve(
            range,
            point_m,
            absolute_error_m,
            seeds,
            control,
            |parameter| evaluate_curve(definition, parameter, control),
        )
    }
}

fn evaluate_curve(
    definition: &ExactCurveDefinition,
    parameter: f64,
    control: &dyn GeometryEvaluationControl,
) -> Result<CurveDerivatives, GeometryEvaluationError> {
    require_parameter(parameter, definition_range(definition))?;
    control.checkpoint()?;
    control.consume_iterations(1)?;
    let result = match definition {
        ExactCurveDefinition::Line {
            origin_m,
            direction_m_per_parameter,
            ..
        } => CurveDerivatives {
            point_m: add_scaled(
                origin_m,
                direction_m_per_parameter,
                parameter,
                &[0.0; 3],
                0.0,
            ),
            first_m: *direction_m_per_parameter,
            second_m: [0.0; 3],
        },
        ExactCurveDefinition::Circle {
            center_m,
            x_axis,
            y_axis,
            radius_m,
            ..
        } => {
            let (point_m, first_m, second_m) =
                trigonometric_curve(center_m, x_axis, y_axis, *radius_m, *radius_m, parameter);
            CurveDerivatives {
                point_m,
                first_m,
                second_m,
            }
        }
        ExactCurveDefinition::Ellipse {
            center_m,
            x_axis,
            y_axis,
            major_radius_m,
            minor_radius_m,
            ..
        } => {
            let (point_m, first_m, second_m) = trigonometric_curve(
                center_m,
                x_axis,
                y_axis,
                *major_radius_m,
                *minor_radius_m,
                parameter,
            );
            CurveDerivatives {
                point_m,
                first_m,
                second_m,
            }
        }
        ExactCurveDefinition::Nurbs { definition } => {
            let value = rational_derivatives(
                definition.degree,
                &definition.knots,
                &definition.control_points_m,
                &definition.weights,
                parameter,
                control,
            )?;
            CurveDerivatives {
                point_m: value.point,
                first_m: value.first,
                second_m: value.second,
            }
        }
    };
    if result
        .point_m
        .iter()
        .chain(&result.first_m)
        .chain(&result.second_m)
        .any(|value| !value.is_finite())
    {
        return Err(invalid_result(
            "curve evaluation produced a non-finite result",
        ));
    }
    Ok(result)
}

pub(super) fn trigonometric_curve<const N: usize>(
    center: &[f64; N],
    x_axis: &[f64; N],
    y_axis: &[f64; N],
    x_radius: f64,
    y_radius: f64,
    parameter: f64,
) -> ([f64; N], [f64; N], [f64; N]) {
    let (sine, cosine) = parameter.sin_cos();
    (
        add_scaled(center, x_axis, x_radius * cosine, y_axis, y_radius * sine),
        add_scaled(
            &[0.0; N],
            x_axis,
            -x_radius * sine,
            y_axis,
            y_radius * cosine,
        ),
        add_scaled(
            &[0.0; N],
            x_axis,
            -x_radius * cosine,
            y_axis,
            -y_radius * sine,
        ),
    )
}

fn definition_range(definition: &ExactCurveDefinition) -> ParameterRange {
    match definition {
        ExactCurveDefinition::Line { domain, .. }
        | ExactCurveDefinition::Circle { domain, .. }
        | ExactCurveDefinition::Ellipse { domain, .. } => *domain,
        ExactCurveDefinition::Nurbs { definition } => definition.domain,
    }
}

fn require_parameter(
    parameter: f64,
    domain: ParameterRange,
) -> Result<(), GeometryEvaluationError> {
    if !parameter.is_finite() || parameter < domain.start || parameter > domain.end {
        return Err(outside_domain(
            "curve parameter lies outside the admitted domain",
        ));
    }
    Ok(())
}

fn require_subrange(
    range: ParameterRange,
    domain: ParameterRange,
) -> Result<(), GeometryEvaluationError> {
    if !range.start.is_finite()
        || !range.end.is_finite()
        || range.start >= range.end
        || range.start < domain.start
        || range.end > domain.end
    {
        return Err(outside_domain(
            "requested parameter range is not an increasing subset of the curve domain",
        ));
    }
    Ok(())
}

fn require_positive_error(value: f64) -> Result<(), GeometryEvaluationError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid_result(
            "evaluation error bound must be finite and positive",
        ));
    }
    Ok(())
}

fn finite_length(value: f64) -> Result<f64, GeometryEvaluationError> {
    if !value.is_finite() || value < 0.0 {
        return Err(invalid_result("arc length is non-finite or negative"));
    }
    Ok(value)
}

fn projection_seeds(
    definition: &ExactCurveDefinition,
    range: ParameterRange,
) -> Result<Vec<f64>, GeometryEvaluationError> {
    match definition {
        ExactCurveDefinition::Nurbs { definition } => {
            let subdivisions = usize::from(definition.degree).saturating_mul(8).max(8);
            let mut seeds = Vec::new();
            for knots in definition.knots.windows(2) {
                let start = knots[0].max(range.start);
                let end = knots[1].min(range.end);
                if start >= end {
                    continue;
                }
                for index in 0..=subdivisions {
                    if seeds.len() >= MAX_NURBS_PROJECTION_SEEDS {
                        return Err(GeometryEvaluationError::new(
                            GeometryEvaluationErrorKind::BudgetExceeded,
                            "NURBS projection seed count exceeds its hard bound",
                        ));
                    }
                    seeds.push(start + (end - start) * index as f64 / subdivisions as f64);
                }
            }
            Ok(seeds)
        }
        _ => Ok(uniform_seeds(range, 256)),
    }
}
