use super::super::{
    surface_principal_curvature, surface_unit_normal, ExactSurfaceDefinition,
    ExactSurfaceEvaluator, ExactSurfaceImplementation, GeometryEvaluationControl,
    GeometryEvaluationError, GeometryEvaluationErrorKind, GeometryParameterAxis,
    GeometryParameterFrame, NurbsSurface3, ParameterRange, SurfaceCurvature,
    SurfaceDerivativeCoefficients, SurfaceDerivatives, SurfaceEvaluatorId, SurfaceProjection,
};
use super::projection::charge_seed_allocation;
use super::surface_projection::{project_surface, uniform_surface_seeds};
use super::surface_spline::rational_surface_derivatives;
use super::vector::{add_scaled, scale};
use super::{invalid_result, kernel_owned, outside_domain, PortableExactEvaluator};

impl ExactSurfaceEvaluator for PortableExactEvaluator<'_> {
    fn parameter_bounds(
        &self,
        id: &SurfaceEvaluatorId,
    ) -> Result<[ParameterRange; 2], GeometryEvaluationError> {
        let record = self.surface_record(id)?;
        let ExactSurfaceImplementation::Portable { definition } = &record.implementation else {
            return Err(kernel_owned("surface"));
        };
        Ok(definition_bounds(definition))
    }

    fn periodicity(
        &self,
        id: &SurfaceEvaluatorId,
    ) -> Result<[Option<f64>; 2], GeometryEvaluationError> {
        let record = self.surface_record(id)?;
        let ExactSurfaceImplementation::Portable { definition } = &record.implementation else {
            return Err(kernel_owned("surface"));
        };
        Ok(definition_periodicity(definition))
    }

    fn point(
        &self,
        id: &SurfaceEvaluatorId,
        uv: [f64; 2],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<crate::physical::Position, GeometryEvaluationError> {
        Ok(ExactSurfaceEvaluator::derivatives(self, id, uv, control)?.point_m)
    }

    fn derivatives(
        &self,
        id: &SurfaceEvaluatorId,
        uv: [f64; 2],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceDerivatives, GeometryEvaluationError> {
        let record = self.surface_record(id)?;
        let ExactSurfaceImplementation::Portable { definition } = &record.implementation else {
            return Err(kernel_owned("surface"));
        };
        let frames = surface_parameter_frames(definition, id);
        evaluate_surface(definition, uv, control)?.into_parameter_frames(frames)
    }

    fn unit_normal(
        &self,
        id: &SurfaceEvaluatorId,
        uv: [f64; 2],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError> {
        surface_unit_normal(&ExactSurfaceEvaluator::derivatives(self, id, uv, control)?)
    }

    fn principal_curvature(
        &self,
        id: &SurfaceEvaluatorId,
        uv: [f64; 2],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceCurvature, GeometryEvaluationError> {
        surface_principal_curvature(&ExactSurfaceEvaluator::derivatives(self, id, uv, control)?)
    }

    fn closest_point(
        &self,
        id: &SurfaceEvaluatorId,
        point_m: crate::physical::Position,
        absolute_error_m: crate::physical::Length,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceProjection, GeometryEvaluationError> {
        let point_m = point_m.canonical_value();
        let absolute_error_m = absolute_error_m.canonical_value();
        let record = self.surface_record(id)?;
        let ExactSurfaceImplementation::Portable { definition } = &record.implementation else {
            return Err(kernel_owned("surface"));
        };
        let bounds = definition_bounds(definition);
        let seeds = projection_seeds(definition, bounds)?;
        charge_seed_allocation(seeds.len(), std::mem::size_of::<[f64; 2]>(), control)?;
        project_surface(bounds, point_m, absolute_error_m, seeds, control, |uv| {
            evaluate_surface(definition, uv, control)
        })?
        .into_parameter_frames(surface_parameter_frames(definition, id))
    }
}

fn evaluate_surface(
    definition: &ExactSurfaceDefinition,
    uv: [f64; 2],
    control: &dyn GeometryEvaluationControl,
) -> Result<SurfaceDerivativeCoefficients, GeometryEvaluationError> {
    require_parameters(uv, definition_bounds(definition))?;
    control.checkpoint()?;
    control.consume_iterations(1)?;
    let result = match definition {
        ExactSurfaceDefinition::Plane {
            origin_m,
            u_axis_m_per_parameter,
            v_axis_m_per_parameter,
            ..
        } => {
            let origin_m = origin_m.canonical_values();
            SurfaceDerivativeCoefficients {
                point_m: add_scaled(
                    origin_m,
                    u_axis_m_per_parameter,
                    uv[0],
                    v_axis_m_per_parameter,
                    uv[1],
                ),
                du_m: *u_axis_m_per_parameter,
                dv_m: *v_axis_m_per_parameter,
                duu_m: [0.0; 3],
                duv_m: [0.0; 3],
                dvv_m: [0.0; 3],
            }
        }
        ExactSurfaceDefinition::Cylinder {
            origin_m,
            x_axis,
            y_axis,
            axis_m_per_v,
            radius_m,
            ..
        } => {
            let origin_m = origin_m.canonical_values();
            {
                let (radial, tangent) = radial_frame(x_axis, y_axis, uv[0]);
                SurfaceDerivativeCoefficients {
                    point_m: add_three(
                        origin_m,
                        &radial,
                        radius_m.canonical_value(),
                        axis_m_per_v,
                        uv[1],
                    ),
                    du_m: scale(&tangent, radius_m.canonical_value()),
                    dv_m: *axis_m_per_v,
                    duu_m: scale(&radial, -radius_m.canonical_value()),
                    duv_m: [0.0; 3],
                    dvv_m: [0.0; 3],
                }
            }
        }
        ExactSurfaceDefinition::Cone {
            apex_m,
            x_axis,
            y_axis,
            axis,
            semi_angle_rad,
            ..
        } => {
            let (radial, tangent) = radial_frame(x_axis, y_axis, uv[0]);
            let (sine, cosine) = semi_angle_rad.canonical_value().sin_cos();
            SurfaceDerivativeCoefficients {
                point_m: add_three(
                    apex_m.canonical_values(),
                    axis,
                    uv[1] * cosine,
                    &radial,
                    uv[1] * sine,
                ),
                du_m: scale(&tangent, uv[1] * sine),
                dv_m: add_scaled(&[0.0; 3], axis, cosine, &radial, sine),
                duu_m: scale(&radial, -uv[1] * sine),
                duv_m: scale(&tangent, sine),
                dvv_m: [0.0; 3],
            }
        }
        ExactSurfaceDefinition::Sphere {
            center_m,
            x_axis,
            y_axis,
            z_axis,
            radius_m,
            ..
        } => {
            let center_m = center_m.canonical_values();
            sphere_derivatives(
                center_m,
                x_axis,
                y_axis,
                z_axis,
                radius_m.canonical_value(),
                uv,
            )
        }
        ExactSurfaceDefinition::Torus {
            center_m,
            x_axis,
            y_axis,
            z_axis,
            major_radius_m,
            minor_radius_m,
            ..
        } => {
            let center_m = center_m.canonical_values();
            torus_derivatives(
                center_m,
                x_axis,
                y_axis,
                z_axis,
                major_radius_m.canonical_value(),
                minor_radius_m.canonical_value(),
                uv,
            )
        }
        ExactSurfaceDefinition::Nurbs { definition } => {
            rational_surface_derivatives(definition, uv, control)?
        }
    };
    validate_result(result)
}

fn sphere_derivatives(
    center: &[f64; 3],
    x_axis: &[f64; 3],
    y_axis: &[f64; 3],
    z_axis: &[f64; 3],
    radius: f64,
    uv: [f64; 2],
) -> SurfaceDerivativeCoefficients {
    let (radial, tangent) = radial_frame(x_axis, y_axis, uv[0]);
    let (sin_v, cos_v) = uv[1].sin_cos();
    SurfaceDerivativeCoefficients {
        point_m: add_three(center, &radial, radius * cos_v, z_axis, radius * sin_v),
        du_m: scale(&tangent, radius * cos_v),
        dv_m: add_scaled(&[0.0; 3], &radial, -radius * sin_v, z_axis, radius * cos_v),
        duu_m: scale(&radial, -radius * cos_v),
        duv_m: scale(&tangent, -radius * sin_v),
        dvv_m: add_scaled(&[0.0; 3], &radial, -radius * cos_v, z_axis, -radius * sin_v),
    }
}

fn torus_derivatives(
    center: &[f64; 3],
    x_axis: &[f64; 3],
    y_axis: &[f64; 3],
    z_axis: &[f64; 3],
    major_radius: f64,
    minor_radius: f64,
    uv: [f64; 2],
) -> SurfaceDerivativeCoefficients {
    let (radial, tangent) = radial_frame(x_axis, y_axis, uv[0]);
    let (sin_v, cos_v) = uv[1].sin_cos();
    let ring_radius = major_radius + minor_radius * cos_v;
    SurfaceDerivativeCoefficients {
        point_m: add_three(center, &radial, ring_radius, z_axis, minor_radius * sin_v),
        du_m: scale(&tangent, ring_radius),
        dv_m: add_scaled(
            &[0.0; 3],
            &radial,
            -minor_radius * sin_v,
            z_axis,
            minor_radius * cos_v,
        ),
        duu_m: scale(&radial, -ring_radius),
        duv_m: scale(&tangent, -minor_radius * sin_v),
        dvv_m: add_scaled(
            &[0.0; 3],
            &radial,
            -minor_radius * cos_v,
            z_axis,
            -minor_radius * sin_v,
        ),
    }
}

fn radial_frame(x_axis: &[f64; 3], y_axis: &[f64; 3], parameter: f64) -> ([f64; 3], [f64; 3]) {
    let (sine, cosine) = parameter.sin_cos();
    (
        add_scaled(&[0.0; 3], x_axis, cosine, y_axis, sine),
        add_scaled(&[0.0; 3], x_axis, -sine, y_axis, cosine),
    )
}

fn add_three(
    origin: &[f64; 3],
    first: &[f64; 3],
    first_scale: f64,
    second: &[f64; 3],
    second_scale: f64,
) -> [f64; 3] {
    add_scaled(origin, first, first_scale, second, second_scale)
}

fn definition_bounds(definition: &ExactSurfaceDefinition) -> [ParameterRange; 2] {
    match definition {
        ExactSurfaceDefinition::Plane { domains, .. }
        | ExactSurfaceDefinition::Cylinder { domains, .. }
        | ExactSurfaceDefinition::Cone { domains, .. }
        | ExactSurfaceDefinition::Sphere { domains, .. }
        | ExactSurfaceDefinition::Torus { domains, .. } => *domains,
        ExactSurfaceDefinition::Nurbs { definition } => definition.domains,
    }
}

fn definition_periodicity(definition: &ExactSurfaceDefinition) -> [Option<f64>; 2] {
    match definition {
        ExactSurfaceDefinition::Plane { .. } => [None, None],
        ExactSurfaceDefinition::Cylinder { .. }
        | ExactSurfaceDefinition::Cone { .. }
        | ExactSurfaceDefinition::Sphere { .. } => [Some(std::f64::consts::TAU), None],
        ExactSurfaceDefinition::Torus { .. } => {
            [Some(std::f64::consts::TAU), Some(std::f64::consts::TAU)]
        }
        ExactSurfaceDefinition::Nurbs { definition } => std::array::from_fn(|axis| {
            [definition.periodic_u, definition.periodic_v][axis]
                .then_some(definition.domains[axis].end - definition.domains[axis].start)
        }),
    }
}

fn require_parameters(
    uv: [f64; 2],
    bounds: [ParameterRange; 2],
) -> Result<(), GeometryEvaluationError> {
    if uv.iter().any(|value| !value.is_finite())
        || uv[0] < bounds[0].start
        || uv[0] > bounds[0].end
        || uv[1] < bounds[1].start
        || uv[1] > bounds[1].end
    {
        return Err(outside_domain(
            "surface parameters lie outside the admitted domain",
        ));
    }
    Ok(())
}

fn validate_result(
    result: SurfaceDerivativeCoefficients,
) -> Result<SurfaceDerivativeCoefficients, GeometryEvaluationError> {
    if result
        .point_m
        .iter()
        .chain(&result.du_m)
        .chain(&result.dv_m)
        .chain(&result.duu_m)
        .chain(&result.duv_m)
        .chain(&result.dvv_m)
        .any(|value| !value.is_finite())
    {
        return Err(invalid_result(
            "surface evaluation produced a non-finite result",
        ));
    }
    Ok(result)
}

fn projection_seeds(
    definition: &ExactSurfaceDefinition,
    bounds: [ParameterRange; 2],
) -> Result<Vec<[f64; 2]>, GeometryEvaluationError> {
    match definition {
        ExactSurfaceDefinition::Nurbs { definition } => nurbs_seeds(definition, bounds),
        _ => uniform_surface_seeds(bounds, 16),
    }
}

fn nurbs_seeds(
    definition: &NurbsSurface3,
    bounds: [ParameterRange; 2],
) -> Result<Vec<[f64; 2]>, GeometryEvaluationError> {
    let u = knot_axis_seeds(&definition.u_knots, bounds[0], definition.u_degree)?;
    let v = knot_axis_seeds(&definition.v_knots, bounds[1], definition.v_degree)?;
    let count = u
        .len()
        .checked_mul(v.len())
        .ok_or_else(|| invalid_result("surface projection seed-grid size overflow"))?;
    if count > 1_000_000 {
        return Err(GeometryEvaluationError::new(
            GeometryEvaluationErrorKind::BudgetExceeded,
            "NURBS surface projection seed count exceeds its hard bound",
        ));
    }
    let mut seeds = Vec::with_capacity(count);
    for u_value in u {
        for v_value in &v {
            seeds.push([u_value, *v_value]);
        }
    }
    Ok(seeds)
}

fn knot_axis_seeds(
    knots: &[f64],
    range: ParameterRange,
    degree: u8,
) -> Result<Vec<f64>, GeometryEvaluationError> {
    let subdivisions = usize::from(degree).saturating_mul(4).max(4);
    let mut values = Vec::new();
    for span in knots.windows(2) {
        let start = span[0].max(range.start);
        let end = span[1].min(range.end);
        if start >= end {
            continue;
        }
        for index in 0..=subdivisions {
            values.push(
                ParameterRange { start, end }
                    .parameter_at_fraction(index as f64 / subdivisions as f64)?,
            );
        }
    }
    values.sort_by(f64::total_cmp);
    values.dedup_by(|left, right| left.to_bits() == right.to_bits());
    Ok(values)
}

fn surface_parameter_frames(
    definition: &ExactSurfaceDefinition,
    id: &SurfaceEvaluatorId,
) -> [GeometryParameterFrame; 2] {
    match definition {
        ExactSurfaceDefinition::Sphere { .. } | ExactSurfaceDefinition::Torus { .. } => [
            GeometryParameterFrame::radians(),
            GeometryParameterFrame::radians(),
        ],
        ExactSurfaceDefinition::Cylinder { .. } | ExactSurfaceDefinition::Cone { .. } => [
            GeometryParameterFrame::radians(),
            GeometryParameterFrame::native(id.as_str(), GeometryParameterAxis::SurfaceV),
        ],
        _ => [
            GeometryParameterFrame::native(id.as_str(), GeometryParameterAxis::SurfaceU),
            GeometryParameterFrame::native(id.as_str(), GeometryParameterAxis::SurfaceV),
        ],
    }
}
