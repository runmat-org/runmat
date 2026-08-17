use super::super::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind,
};
use super::invalid_result;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct RationalDerivatives<const N: usize> {
    pub point: [f64; N],
    pub first: [f64; N],
    pub second: [f64; N],
}

struct DerivativePolygon {
    degree: usize,
    knots: Vec<f64>,
    controls: Vec<Vec<f64>>,
}

pub(super) struct HomogeneousDerivatives {
    pub value: Vec<f64>,
    pub first: Vec<f64>,
    pub second: Vec<f64>,
}

pub(super) fn rational_derivatives<const N: usize>(
    degree: u8,
    knots: &[f64],
    control_points: &[[f64; N]],
    weights: &[f64],
    parameter: f64,
    control: &dyn GeometryEvaluationControl,
) -> Result<RationalDerivatives<N>, GeometryEvaluationError> {
    control.checkpoint()?;
    charge_spline_allocation(control_points.len(), N + 1, 3, control)?;
    let degree = usize::from(degree);
    let work = (degree + 1).saturating_mul(degree + 1).saturating_mul(3);
    let work = u64::try_from(work)
        .map_err(|_| invalid_result("spline evaluation work count does not fit u64"))?;
    control.consume_iterations(work)?;

    let homogeneous = control_points
        .iter()
        .zip(weights)
        .map(|(point, weight)| {
            let mut value = Vec::with_capacity(N + 1);
            value.extend(point.iter().map(|coordinate| coordinate * weight));
            value.push(*weight);
            value
        })
        .collect::<Vec<_>>();
    let homogeneous = homogeneous_derivatives(degree, knots, &homogeneous, parameter)?;
    rationalize(&homogeneous)
}

pub(super) fn charge_spline_allocation(
    control_count: usize,
    coordinate_count: usize,
    conservative_copies: usize,
    control: &dyn GeometryEvaluationControl,
) -> Result<(), GeometryEvaluationError> {
    let bytes_per_control = coordinate_count
        .checked_mul(std::mem::size_of::<f64>())
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Vec<f64>>()))
        .ok_or_else(|| invalid_result("spline allocation-byte count overflow"))?;
    let bytes = control_count
        .checked_mul(bytes_per_control)
        .and_then(|bytes| bytes.checked_mul(conservative_copies))
        .ok_or_else(|| invalid_result("spline allocation-byte count overflow"))?;
    control.consume_allocation_bytes(
        u64::try_from(bytes)
            .map_err(|_| invalid_result("spline allocation-byte count does not fit u64"))?,
    )
}

pub(super) fn homogeneous_derivatives(
    degree: usize,
    knots: &[f64],
    controls: &[Vec<f64>],
    parameter: f64,
) -> Result<HomogeneousDerivatives, GeometryEvaluationError> {
    let value = de_boor(degree, knots, controls, parameter)?;
    let first_polygon = derivative_polygon(degree, knots, controls)?;
    let first = de_boor(
        first_polygon.degree,
        &first_polygon.knots,
        &first_polygon.controls,
        parameter,
    )?;
    let second = if first_polygon.degree == 0 {
        vec![0.0; value.len()]
    } else {
        let second_polygon = derivative_polygon(
            first_polygon.degree,
            &first_polygon.knots,
            &first_polygon.controls,
        )?;
        de_boor(
            second_polygon.degree,
            &second_polygon.knots,
            &second_polygon.controls,
            parameter,
        )?
    };

    Ok(HomogeneousDerivatives {
        value,
        first,
        second,
    })
}

fn rationalize<const N: usize>(
    homogeneous: &HomogeneousDerivatives,
) -> Result<RationalDerivatives<N>, GeometryEvaluationError> {
    let value = &homogeneous.value;
    let first = &homogeneous.first;
    let second = &homogeneous.second;
    let weight = value[N];
    if !weight.is_finite() || weight <= 0.0 {
        return Err(invalid_result(
            "rational spline produced a non-positive weight",
        ));
    }
    let point = std::array::from_fn(|index| value[index] / weight);
    let first_result =
        std::array::from_fn(|index| (first[index] - point[index] * first[N]) / weight);
    let second_result = std::array::from_fn(|index| {
        (second[index] - 2.0 * first_result[index] * first[N] - point[index] * second[N]) / weight
    });
    if point
        .iter()
        .chain(&first_result)
        .chain(&second_result)
        .any(|value| !value.is_finite())
    {
        return Err(invalid_result(
            "rational spline produced a non-finite derivative",
        ));
    }
    Ok(RationalDerivatives {
        point,
        first: first_result,
        second: second_result,
    })
}

fn derivative_polygon(
    degree: usize,
    knots: &[f64],
    controls: &[Vec<f64>],
) -> Result<DerivativePolygon, GeometryEvaluationError> {
    if degree == 0 || controls.len() < 2 || knots.len() < 2 {
        return Err(invalid_result("spline derivative polygon is undefined"));
    }
    let mut derivative = Vec::with_capacity(controls.len() - 1);
    for index in 0..controls.len() - 1 {
        let denominator = knots[index + degree + 1] - knots[index + 1];
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(invalid_result("spline derivative has a zero knot span"));
        }
        let factor = degree as f64 / denominator;
        derivative.push(
            controls[index + 1]
                .iter()
                .zip(&controls[index])
                .map(|(next, current)| (next - current) * factor)
                .collect(),
        );
    }
    Ok(DerivativePolygon {
        degree: degree - 1,
        knots: knots[1..knots.len() - 1].to_vec(),
        controls: derivative,
    })
}

fn de_boor(
    degree: usize,
    knots: &[f64],
    controls: &[Vec<f64>],
    parameter: f64,
) -> Result<Vec<f64>, GeometryEvaluationError> {
    let span = find_span(degree, knots, controls.len(), parameter)?;
    let mut values = (0..=degree)
        .map(|offset| controls[span - degree + offset].clone())
        .collect::<Vec<_>>();
    for level in 1..=degree {
        for offset in (level..=degree).rev() {
            let knot_index = span - degree + offset;
            let denominator = knots[knot_index + degree - level + 1] - knots[knot_index];
            let alpha = if denominator == 0.0 {
                0.0
            } else {
                (parameter - knots[knot_index]) / denominator
            };
            for coordinate in 0..values[offset].len() {
                values[offset][coordinate] = (1.0 - alpha) * values[offset - 1][coordinate]
                    + alpha * values[offset][coordinate];
            }
        }
    }
    Ok(values[degree].clone())
}

fn find_span(
    degree: usize,
    knots: &[f64],
    control_count: usize,
    parameter: f64,
) -> Result<usize, GeometryEvaluationError> {
    let last_control = control_count - 1;
    let domain_start = knots[degree];
    let domain_end = knots[control_count];
    if !parameter.is_finite() || parameter < domain_start || parameter > domain_end {
        return Err(GeometryEvaluationError::new(
            GeometryEvaluationErrorKind::ParameterOutsideDomain,
            "spline parameter lies outside its basis domain",
        ));
    }
    if parameter == domain_end {
        return Ok(last_control);
    }
    let (mut low, mut high) = (degree, control_count);
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        if parameter < knots[middle] {
            high = middle;
        } else {
            low = middle;
        }
    }
    Ok(low)
}
