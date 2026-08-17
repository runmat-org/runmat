use super::{
    GeometryEvaluationError, GeometryEvaluationErrorKind, SurfaceCurvature, SurfaceDerivatives,
};

struct FundamentalForms {
    metric: [f64; 3],
    shape: [f64; 3],
}

/// Derives the intrinsic unit normal from exact surface derivatives.
pub fn surface_unit_normal(
    derivatives: &SurfaceDerivatives,
) -> Result<[f64; 3], GeometryEvaluationError> {
    normalize(cross(derivatives.du_m, derivatives.dv_m))
        .ok_or_else(|| invalid("surface normal is undefined at a singular parameter"))
}

/// Solves the fundamental-form eigenproblem for exact principal curvatures.
pub fn surface_principal_curvature(
    derivatives: &SurfaceDerivatives,
) -> Result<SurfaceCurvature, GeometryEvaluationError> {
    let normal = surface_unit_normal(derivatives)?;
    let forms = FundamentalForms {
        metric: [
            dot(derivatives.du_m, derivatives.du_m),
            dot(derivatives.du_m, derivatives.dv_m),
            dot(derivatives.dv_m, derivatives.dv_m),
        ],
        shape: [
            dot(normal, derivatives.duu_m),
            dot(normal, derivatives.duv_m),
            dot(normal, derivatives.dvv_m),
        ],
    };
    let [e_metric, f_metric, g_metric] = forms.metric;
    let determinant = e_metric * g_metric - f_metric * f_metric;
    if !determinant.is_finite() || determinant <= 0.0 {
        return Err(invalid("surface first fundamental form is singular"));
    }
    let [e_shape, f_shape, g_shape] = forms.shape;
    let trace = (e_shape * g_metric + g_shape * e_metric - 2.0 * f_shape * f_metric) / determinant;
    let gaussian = (e_shape * g_shape - f_shape * f_shape) / determinant;
    let raw_discriminant = trace * trace - 4.0 * gaussian;
    let roundoff_bound =
        f64::EPSILON * 64.0 * (trace * trace).abs().max((4.0 * gaussian).abs()).max(1.0);
    if !raw_discriminant.is_finite() || raw_discriminant < -roundoff_bound {
        return Err(invalid(
            "surface curvature eigenproblem has an invalid discriminant",
        ));
    }
    let discriminant = raw_discriminant.max(0.0).sqrt();
    let minimum = 0.5 * (trace - discriminant);
    let maximum = 0.5 * (trace + discriminant);
    if !minimum.is_finite() || !maximum.is_finite() {
        return Err(invalid("surface principal curvature is non-finite"));
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
    let chosen = if dot(first, first) >= dot(second, second) {
        first
    } else {
        second
    };
    let mut direction =
        normalize(chosen).ok_or_else(|| invalid("surface principal direction is undefined"))?;
    if direction[0] < 0.0 || (direction[0] == 0.0 && direction[1] < 0.0) {
        direction[0] = -direction[0];
        direction[1] = -direction[1];
    }
    Ok(direction)
}

fn dot<const N: usize>(left: [f64; N], right: [f64; N]) -> f64 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}

fn normalize<const N: usize>(value: [f64; N]) -> Option<[f64; N]> {
    let magnitude = dot(value, value).sqrt();
    (magnitude.is_finite() && magnitude > 0.0).then(|| value.map(|component| component / magnitude))
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn invalid(reason: impl Into<String>) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::InvalidResult, reason)
}
