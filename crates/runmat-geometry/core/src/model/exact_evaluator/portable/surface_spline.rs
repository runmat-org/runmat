use super::super::{
    GeometryEvaluationControl, GeometryEvaluationError, NurbsSurface3, SurfaceDerivatives,
};
use super::invalid_result;
use super::spline::{charge_spline_allocation, homogeneous_derivatives, HomogeneousDerivatives};

pub(super) fn rational_surface_derivatives(
    definition: &NurbsSurface3,
    uv: [f64; 2],
    control: &dyn GeometryEvaluationControl,
) -> Result<SurfaceDerivatives, GeometryEvaluationError> {
    control.checkpoint()?;
    let u_degree = usize::from(definition.u_degree);
    let v_degree = usize::from(definition.v_degree);
    let u_count = usize::try_from(definition.u_control_count)
        .map_err(|_| invalid_result("surface u control count does not fit this platform"))?;
    let v_count = usize::try_from(definition.v_control_count)
        .map_err(|_| invalid_result("surface v control count does not fit this platform"))?;
    charge_spline_allocation(definition.control_points_m.len(), 4, 16, control)?;
    let work = (u_degree + 1)
        .saturating_mul(u_degree + 1)
        .saturating_mul(v_count)
        .saturating_add(
            (v_degree + 1)
                .saturating_mul(v_degree + 1)
                .saturating_mul(3),
        );
    control.consume_iterations(
        u64::try_from(work)
            .map_err(|_| invalid_result("surface spline work count does not fit u64"))?,
    )?;

    let homogeneous = definition
        .control_points_m
        .iter()
        .zip(&definition.weights)
        .map(|(point, weight)| {
            vec![
                point[0] * weight,
                point[1] * weight,
                point[2] * weight,
                *weight,
            ]
        })
        .collect::<Vec<_>>();
    let mut along_u = Vec::with_capacity(v_count);
    for v in 0..v_count {
        let controls = (0..u_count)
            .map(|u| homogeneous[u * v_count + v].clone())
            .collect::<Vec<_>>();
        along_u.push(homogeneous_derivatives(
            u_degree,
            &definition.u_knots,
            &controls,
            uv[0],
        )?);
    }

    let u_values = select(&along_u, |value| &value.value);
    let u_first = select(&along_u, |value| &value.first);
    let u_second = select(&along_u, |value| &value.second);
    let value_v = homogeneous_derivatives(v_degree, &definition.v_knots, &u_values, uv[1])?;
    let first_v = homogeneous_derivatives(v_degree, &definition.v_knots, &u_first, uv[1])?;
    let second_v = homogeneous_derivatives(v_degree, &definition.v_knots, &u_second, uv[1])?;
    rationalize_surface(&value_v, &first_v, &second_v)
}

fn select(
    values: &[HomogeneousDerivatives],
    field: impl Fn(&HomogeneousDerivatives) -> &Vec<f64>,
) -> Vec<Vec<f64>> {
    values.iter().map(|value| field(value).clone()).collect()
}

fn rationalize_surface(
    along_v: &HomogeneousDerivatives,
    u_first_along_v: &HomogeneousDerivatives,
    u_second_along_v: &HomogeneousDerivatives,
) -> Result<SurfaceDerivatives, GeometryEvaluationError> {
    let h = &along_v.value;
    let hu = &u_first_along_v.value;
    let hv = &along_v.first;
    let huu = &u_second_along_v.value;
    let huv = &u_first_along_v.first;
    let hvv = &along_v.second;
    let weight = h[3];
    if !weight.is_finite() || weight <= 0.0 {
        return Err(invalid_result(
            "rational surface produced a non-positive weight",
        ));
    }
    let point_m = std::array::from_fn(|index| h[index] / weight);
    let du_m = std::array::from_fn(|index| (hu[index] - point_m[index] * hu[3]) / weight);
    let dv_m = std::array::from_fn(|index| (hv[index] - point_m[index] * hv[3]) / weight);
    let duu_m = std::array::from_fn(|index| {
        (huu[index] - 2.0 * du_m[index] * hu[3] - point_m[index] * huu[3]) / weight
    });
    let duv_m = std::array::from_fn(|index| {
        (huv[index] - du_m[index] * hv[3] - dv_m[index] * hu[3] - point_m[index] * huv[3]) / weight
    });
    let dvv_m = std::array::from_fn(|index| {
        (hvv[index] - 2.0 * dv_m[index] * hv[3] - point_m[index] * hvv[3]) / weight
    });
    let result = SurfaceDerivatives {
        point_m,
        du_m,
        dv_m,
        duu_m,
        duv_m,
        dvv_m,
    };
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
            "rational surface produced a non-finite derivative",
        ));
    }
    Ok(result)
}
