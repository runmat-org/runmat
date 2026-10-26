//! Numerical points are only proposals. Incumbents retain an exact original-model witness.
use super::*;
use common::conic::{self, linear};
use num_rational::BigRational;
use num_traits::ToPrimitive;

pub(super) struct Incumbent {
    pub point: Vec<BigRational>,
    pub objective: BigRational,
}
pub(super) async fn incumbent(
    model: &conic::Model<'_>,
    integers: &[usize],
    semis: &[domains::Semi],
    proposal: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Incumbent>> {
    let Some(point) = conic::terminal::verified_point_satisfying(
        model,
        proposal,
        cooperation,
        async |point: &[BigRational], cooperation: &mut ScalarCooperation| {
            admitted(point, integers, semis, cooperation).await
        },
    )
    .await?
    else {
        return Ok(None);
    };
    from_verified(model, integers, semis, point, cooperation).await
}
pub(super) async fn from_verified(
    model: &conic::Model<'_>,
    integers: &[usize],
    semis: &[domains::Semi],
    point: conic::terminal::VerifiedPoint,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Incumbent>> {
    if !admitted(point.values(), integers, semis, cooperation).await? {
        return Ok(None);
    }
    let Some(objective) = linear::objective(model, point.values(), cooperation).await? else {
        return Ok(None);
    };
    Ok(Some(Incumbent {
        point: point.into_values(),
        objective,
    }))
}

async fn admitted(
    point: &[BigRational],
    integers: &[usize],
    semis: &[domains::Semi],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    if !domains::admitted(point, semis, cooperation).await? {
        return Ok(false);
    }
    for &column in integers {
        if point.get(column).is_none_or(|v| !v.is_integer()) {
            return Ok(false);
        }
        cooperation.completed(1).await?;
    }
    Ok(true)
}
/// The pair comes from one shared Candidate and its one original node model.
/// Any positive common denominator scales a rational recession direction to
/// an integer displacement on selected coordinates, preserving all homogeneous
/// cone conditions and strict objective descent. This adds no domain cap.
pub(super) async fn integral_unbounded(
    point: &conic::terminal::VerifiedPoint,
    ray: &conic::terminal::VerifiedRay,
    integers: &[usize],
    semis: &[domains::Semi],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    use num_bigint::BigInt;
    use num_traits::{One, Signed};
    if point.values().len() != ray.values().len() {
        return Ok(false);
    }
    if !domains::admitted(point.values(), semis, cooperation).await? {
        return Ok(false);
    }
    // Each semi domain is bounded. A direction cannot leave its admitted
    // zero/active disjunct, even if its convex relaxation has a valid ray.
    for domain in semis {
        if ray
            .values()
            .get(domain.column)
            .is_none_or(|value| !num_traits::Zero::is_zero(value))
        {
            return Ok(false);
        }
        cooperation.completed(1).await?;
    }
    let mut scale = BigInt::one();
    for &column in integers {
        if point.values().get(column).is_none_or(|v| !v.is_integer()) {
            return Ok(false);
        }
        let Some(direction) = ray.values().get(column) else {
            return Ok(false);
        };
        let denominator = direction.denom();
        // Ratio normalization uses the existing exact gcd authority. Its
        // reduced numerator is scale/gcd(scale,denominator), so this is lcm.
        scale = BigRational::new(scale, denominator.clone()).numer() * denominator;
        cooperation.completed(1).await?;
    }
    if !scale.is_positive() {
        return Ok(false);
    }
    let scale = BigRational::from_integer(scale);
    for &column in integers {
        if !(&ray.values()[column] * &scale).is_integer() {
            return Ok(false);
        }
        cooperation.completed(1).await?;
    }
    Ok(true)
}
pub(super) async fn output_point(
    point: &[BigRational],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Vec<f64>>> {
    let mut result = runmat_value::exact_capacity_vec(point.len()).map_err(allocation)?;
    for chunk in point.chunks(1024) {
        for value in chunk {
            let Some(converted) = value.to_f64().filter(|v| v.is_finite()) else {
                return Ok(None);
            };
            // This alone grants no representability fact: the output owner
            // compares integer coordinates exactly before publishing.
            result.push(converted);
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(Some(result))
}
