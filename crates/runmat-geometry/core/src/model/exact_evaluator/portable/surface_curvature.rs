use super::super::{GeometryEvaluationError, SurfaceCurvature, SurfaceDerivatives};
use super::invalid_result;

struct FundamentalForms {
    metric: [f64; 3],
    shape: [f64; 3],
}
use super::vector::{cross, dot, normalize};

pub(super) fn unit_normal(
    derivatives: &SurfaceDerivatives,
) -> Result<[f64; 3], GeometryEvaluationError> {
    normalize(&cross(&derivatives.du_m, &derivatives.dv_m))
        .ok_or_else(|| invalid_result("surface normal is undefined at a singular parameter"))
}

pub(super) fn principal_curvature(
    derivatives: &SurfaceDerivatives,
) -> Result<SurfaceCurvature, GeometryEvaluationError> {
    let normal = unit_normal(derivatives)?;
    let forms = FundamentalForms {
        metric: [
            dot(&derivatives.du_m, &derivatives.du_m),
            dot(&derivatives.du_m, &derivatives.dv_m),
            dot(&derivatives.dv_m, &derivatives.dv_m),
        ],
        shape: [
            dot(&normal, &derivatives.duu_m),
            dot(&normal, &derivatives.duv_m),
            dot(&normal, &derivatives.dvv_m),
        ],
    };
    let [e_metric, f_metric, g_metric] = forms.metric;
    let determinant = e_metric * g_metric - f_metric * f_metric;
    if !determinant.is_finite() || determinant <= 0.0 {
        return Err(invalid_result("surface first fundamental form is singular"));
    }
    let [e_shape, f_shape, g_shape] = forms.shape;
    let trace = (e_shape * g_metric + g_shape * e_metric - 2.0 * f_shape * f_metric) / determinant;
    let gaussian = (e_shape * g_shape - f_shape * f_shape) / determinant;
    let raw_discriminant = trace * trace - 4.0 * gaussian;
    let roundoff_bound =
        f64::EPSILON * 64.0 * (trace * trace).abs().max((4.0 * gaussian).abs()).max(1.0);
    if !raw_discriminant.is_finite() || raw_discriminant < -roundoff_bound {
        return Err(invalid_result(
            "surface curvature eigenproblem has an invalid discriminant",
        ));
    }
    let discriminant = raw_discriminant.max(0.0).sqrt();
    let minimum = 0.5 * (trace - discriminant);
    let maximum = 0.5 * (trace + discriminant);
    if !minimum.is_finite() || !maximum.is_finite() {
        return Err(invalid_result("surface principal curvature is non-finite"));
    }
    let (minimum_direction_uv, maximum_direction_uv) =
        if discriminant <= f64::EPSILON * 64.0 * trace.abs().max(1.0) {
            ([1.0, 0.0], [0.0, 1.0])
        } else {
            (
                principal_direction(minimum, &forms)?,
                principal_direction(maximum, &forms)?,
            )
        };
    Ok(SurfaceCurvature {
        minimum_1_per_m: minimum,
        maximum_1_per_m: maximum,
        minimum_direction_uv,
        maximum_direction_uv,
    })
}

fn principal_direction(
    curvature: f64,
    forms: &FundamentalForms,
) -> Result<[f64; 2], GeometryEvaluationError> {
    let [e_metric, f_metric, g_metric] = forms.metric;
    let [e_shape, f_shape, g_shape] = forms.shape;
    let first = [
        -(f_shape - curvature * f_metric),
        e_shape - curvature * e_metric,
    ];
    let second = [
        g_shape - curvature * g_metric,
        -(f_shape - curvature * f_metric),
    ];
    let first_norm = dot(&first, &first);
    let second_norm = dot(&second, &second);
    let chosen = if first_norm >= second_norm {
        first
    } else {
        second
    };
    let mut direction = normalize(&chosen)
        .ok_or_else(|| invalid_result("surface principal direction is undefined"))?;
    if direction[0] < 0.0 || (direction[0] == 0.0 && direction[1] < 0.0) {
        direction[0] = -direction[0];
        direction[1] = -direction[1];
    }
    Ok(direction)
}
