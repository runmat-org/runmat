use super::{
    definition_validation_math::{finite_vector, validate_range},
    NurbsCurve2, NurbsCurve3, NurbsSurface3, ParameterRange,
};
use crate::model::GeometryContractError;

const MAX_NURBS_DEGREE: u8 = 15;
const MAX_CONTROL_POINTS: usize = 1_000_000;

pub(super) fn validate_nurbs_curve3(definition: &NurbsCurve3) -> Result<(), GeometryContractError> {
    validate_nurbs_components(
        "3D NURBS curve",
        definition.degree,
        &definition.knots,
        definition.control_points_m.len(),
        &definition.weights,
        &definition.domain,
    )?;
    for point in &definition.control_points_m {
        finite_vector("3D NURBS control point", point)?;
    }
    Ok(())
}

pub(super) fn validate_nurbs_curve2(definition: &NurbsCurve2) -> Result<(), GeometryContractError> {
    validate_nurbs_components(
        "2D NURBS curve",
        definition.degree,
        &definition.knots,
        definition.control_points_uv.len(),
        &definition.weights,
        &definition.domain,
    )?;
    for point in &definition.control_points_uv {
        finite_vector("2D NURBS control point", point)?;
    }
    Ok(())
}

pub(super) fn validate_nurbs_surface(
    definition: &NurbsSurface3,
) -> Result<(), GeometryContractError> {
    let u_count = usize::try_from(definition.u_control_count).map_err(|_| {
        invalid(
            "NURBS surface",
            "u control count does not fit this platform",
        )
    })?;
    let v_count = usize::try_from(definition.v_control_count).map_err(|_| {
        invalid(
            "NURBS surface",
            "v control count does not fit this platform",
        )
    })?;
    let total = u_count
        .checked_mul(v_count)
        .ok_or_else(|| invalid("NURBS surface", "control-grid size overflow"))?;
    if total == 0
        || total > MAX_CONTROL_POINTS
        || definition.control_points_m.len() != total
        || definition.weights.len() != total
    {
        return Err(invalid(
            "NURBS surface control grid",
            "dimensions, controls, and weights must agree within the hard bound",
        ));
    }
    validate_knot_axis(
        "NURBS surface u axis",
        definition.u_degree,
        &definition.u_knots,
        u_count,
        &definition.domains[0],
    )?;
    validate_knot_axis(
        "NURBS surface v axis",
        definition.v_degree,
        &definition.v_knots,
        v_count,
        &definition.domains[1],
    )?;
    for point in &definition.control_points_m {
        finite_vector("NURBS surface control point", point)?;
    }
    validate_weights("NURBS surface weights", &definition.weights)
}

fn validate_nurbs_components(
    field: &str,
    degree: u8,
    knots: &[f64],
    control_count: usize,
    weights: &[f64],
    domain: &ParameterRange,
) -> Result<(), GeometryContractError> {
    if control_count == 0 || control_count > MAX_CONTROL_POINTS || weights.len() != control_count {
        return Err(invalid(
            field,
            "controls and weights must agree within the hard bound",
        ));
    }
    validate_knot_axis(field, degree, knots, control_count, domain)?;
    validate_weights(field, weights)
}

fn validate_knot_axis(
    field: &str,
    degree: u8,
    knots: &[f64],
    control_count: usize,
    domain: &ParameterRange,
) -> Result<(), GeometryContractError> {
    let degree = usize::from(degree);
    if degree == 0
        || degree > usize::from(MAX_NURBS_DEGREE)
        || control_count <= degree
        || knots.len() != control_count.saturating_add(degree).saturating_add(1)
        || knots.iter().any(|value| !value.is_finite())
        || knots.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(invalid(
            field,
            "degree, controls, and finite nondecreasing knot vector are inconsistent",
        ));
    }
    let mut run_start = 0;
    while run_start < knots.len() {
        let mut run_end = run_start + 1;
        while run_end < knots.len() && knots[run_end] == knots[run_start] {
            run_end += 1;
        }
        let maximum = if run_start == 0 || run_end == knots.len() {
            degree + 1
        } else {
            degree.saturating_sub(1)
        };
        if run_end - run_start > maximum {
            return Err(invalid(
                field,
                "interior knot multiplicity must preserve at least C1 continuity; split lower-continuity geometry into topological entities",
            ));
        }
        run_start = run_end;
    }
    validate_range(field, domain)?;
    let knot_domain = ParameterRange {
        start: knots[degree],
        end: knots[knots.len() - degree - 1],
    };
    if knot_domain.start >= knot_domain.end
        || domain.start < knot_domain.start
        || domain.end > knot_domain.end
    {
        return Err(invalid(
            field,
            "declared domain must lie within the nonempty basis domain",
        ));
    }
    Ok(())
}

fn validate_weights(field: &str, weights: &[f64]) -> Result<(), GeometryContractError> {
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return Err(invalid(
            field,
            "all rational weights must be finite and positive",
        ));
    }
    Ok(())
}

fn invalid(field: &str, reason: impl Into<String>) -> GeometryContractError {
    GeometryContractError::invalid(field, reason)
}
