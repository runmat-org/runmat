//! Exact original-row step bounds, followed by verification of the returned
//! binary64 generation. A numeric direction is not itself a feasible witness.
use super::super::super::intlinprog::feasibility::Repair;
use super::super::{common, invalid, ScalarCooperation, POLICY};
use crate::BuiltinResult;
use common::{global_time::TimeLimit, solver_timing::SolverTiming};
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};

pub(in crate::builtins::math::optim) enum Step {
    Point(Vec<f64>),
    Blocked,
    TimeLimit,
    Unverified,
}
pub(in crate::builtins::math::optim) struct Controls<'a> {
    pub maximum: f64,
    pub draw: f64,
    pub tolerance: f64,
    pub timing: &'a SolverTiming,
    pub deadline: TimeLimit,
}
pub(super) async fn candidate<G: super::Geometry + ?Sized>(
    domain: &mut G,
    parent: &[f64],
    direction: &[BigRational],
    controls: Controls<'_>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Step> {
    let Controls {
        maximum,
        draw,
        tolerance,
        timing,
        deadline,
    } = controls;
    if parent.len() != direction.len()
        || parent.len() != domain.bounds().0.len()
        || !maximum.is_finite()
        || maximum < 0.
        || !draw.is_finite()
        || !(0.0..1.0).contains(&draw)
        || !tolerance.is_finite()
        || tolerance < 0.
    {
        return Err(invalid("invalid original-domain step geometry or controls"));
    }
    if domain.time_limit_reached(timing, deadline) {
        return Ok(Step::TimeLimit);
    }
    if !domain
        .classify(parent, tolerance, cooperation)
        .await?
        .feasible
    {
        return Err(invalid(
            "feasible genetic mutation requires an admitted parent",
        ));
    }
    let mut exact = POLICY.capacity(parent.len(), cooperation.runtime())?;
    for &value in parent {
        exact.push(
            BigRational::from_float(value)
                .ok_or_else(|| invalid("mutation parent must be finite"))?,
        );
        cooperation.completed(1).await?;
    }
    let mut length = BigRational::from_float(maximum).expect("finite step maximum");
    let (lower, upper) = domain.bounds();
    for (column, delta) in direction.iter().enumerate() {
        let boundary = if delta.is_positive() {
            upper[column]
        } else {
            lower[column]
        };
        if !delta.is_zero() && boundary.is_finite() {
            let limit = (BigRational::from_float(boundary).expect("finite original bound")
                - &exact[column])
                / delta;
            length = length.min(limit);
        }
        cooperation.completed(1).await?;
    }
    let tolerance = BigRational::from_float(tolerance).expect("finite original tolerance");
    let mut row = domain.row_scratch(cooperation)?;
    for index in 0..domain.rhs().len() {
        if domain.time_limit_reached(timing, deadline) {
            return Ok(Step::TimeLimit);
        }
        domain.original_row(index, &mut row, cooperation).await?;
        let mut value = BigRational::zero();
        let mut delta = BigRational::zero();
        for &(column, coefficient) in &row {
            let coefficient =
                BigRational::from_float(coefficient).expect("validated original coefficient");
            value += &coefficient * &exact[column];
            delta += coefficient * &direction[column];
            cooperation.completed(1).await?;
        }
        let rhs = BigRational::from_float(domain.rhs()[index]).expect("validated original rhs");
        if delta.is_positive() {
            length = length.min((&rhs + &tolerance - &value) / &delta);
        } else if delta.is_negative() && index < domain.equalities() {
            length = length.min((&rhs - &tolerance - &value) / &delta);
        }
        cooperation.completed(1).await?;
    }
    length *= BigRational::from_float(draw).expect("uniform draw");
    let tolerance = tolerance.to_f64().expect("original finite tolerance");
    let half = BigRational::new(1.into(), 2.into());
    while length.is_positive() {
        if domain.time_limit_reached(timing, deadline) {
            return Ok(Step::TimeLimit);
        }
        let mut point = POLICY.capacity(parent.len(), cooperation.runtime())?;
        let mut changed = false;
        let mut representable = true;
        for (column, delta) in direction.iter().enumerate() {
            if delta.is_zero() {
                point.push(parent[column]);
                cooperation.completed(1).await?;
                continue;
            }
            let value = &exact[column] + &length * delta;
            let Some(value) = value.to_f64().filter(|value| value.is_finite()) else {
                representable = false;
                break;
            };
            changed |= value.to_bits() != parent[column].to_bits();
            point.push(value);
            cooperation.completed(1).await?;
        }
        if representable {
            if !changed {
                return Ok(Step::Blocked);
            }
            if domain
                .classify(&point, tolerance, cooperation)
                .await?
                .feasible
            {
                return Ok(Step::Point(point));
            }
            if !domain.integers().is_empty() {
                // Integer feasibility is the existing original-model proof
                // owner, never rounded fractional coordinates or ray status.
                match domain.repair(&point, cooperation).await? {
                    Repair::Point(point) => {
                        if domain
                            .classify(&point, tolerance, cooperation)
                            .await?
                            .feasible
                        {
                            return Ok(Step::Point(point));
                        }
                    }
                    Repair::TimeLimit => return Ok(Step::TimeLimit),
                    Repair::Unverified => return Ok(Step::Unverified),
                    Repair::Infeasible => return Ok(Step::Blocked),
                    Repair::Unchanged => (),
                }
            }
        }
        // Rounding can put a boundary trial just outside the user's original
        // tolerance. Shorten the same exact step; do not clamp a coordinate.
        length *= &half;
        cooperation.completed(1).await?;
    }
    Ok(Step::Blocked)
}
