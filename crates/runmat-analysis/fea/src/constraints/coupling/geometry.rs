//! Original rotational-geometry admissibility, borrowing coordinate authority.
use super::*;

pub(super) fn validate_rotational_geometry(
    weights: &[ConstraintCouplingWeight],
    coordinates: &impl CoordinateLookup,
    characteristic_length: f64,
    constraint_id: &str,
) -> Result<(), ConstraintPlanError> {
    if !characteristic_length.is_finite() || characteristic_length <= 1.0e-12 {
        return Err(ConstraintPlanError::Invalid(format!(
            "constraint `{constraint_id}` coupling has no rotational lever arm"
        )));
    }
    let centroid = weights.iter().try_fold([0.0; 3], |sum, weight| {
        coordinates
            .position(&weight.node_identity)
            .map(|position| super::super::vector::add(sum, scale(position, weight.weight)))
            .ok_or_else(|| {
                ConstraintPlanError::Invalid(format!(
                    "constraint `{constraint_id}` coupling node is absent from solver topology"
                ))
            })
    })?;
    let mut inertia = [[0.0; 3]; 3];
    for weight in weights {
        check_cancelled()?;
        let arm = subtract(
            coordinates
                .position(&weight.node_identity)
                .ok_or_else(|| degenerate_coupling(constraint_id))?,
            centroid,
        );
        let squared_radius = arm.iter().map(|value| value * value).sum::<f64>();
        for row in 0..3 {
            for column in 0..3 {
                inertia[row][column] += weight.weight
                    * if row == column {
                        squared_radius - arm[row] * arm[column]
                    } else {
                        -arm[row] * arm[column]
                    };
            }
        }
    }
    let scale = inertia
        .iter()
        .flatten()
        .map(|value| value.abs())
        .sum::<f64>();
    let tolerance = scale.max(f64::MIN_POSITIVE) * 1.0e-12;
    let first = inertia[0][0];
    if !first.is_finite() || first <= tolerance {
        return Err(degenerate_coupling(constraint_id));
    }
    let second = inertia[1][1] - inertia[1][0] * inertia[1][0] / first;
    if !second.is_finite() || second <= tolerance {
        return Err(degenerate_coupling(constraint_id));
    }
    let lower_20 = inertia[2][0] / first.sqrt();
    let lower_21 = (inertia[2][1] - lower_20 * inertia[1][0] / first.sqrt()) / second.sqrt();
    let third = inertia[2][2] - lower_20 * lower_20 - lower_21 * lower_21;
    if !third.is_finite() || third <= tolerance {
        return Err(degenerate_coupling(constraint_id));
    }
    Ok(())
}

fn degenerate_coupling(constraint_id: &str) -> ConstraintPlanError {
    ConstraintPlanError::Invalid(format!(
        "constraint `{constraint_id}` coupling target must contain non-collinear nodes"
    ))
}
