//! One actual Latin-hypercube candidate set, frozen before user callbacks.
use super::super::{buffer, invalid, random::shuffle_in_turn};
use crate::builtins::{common::random, math::optim::common::ScalarCooperation};
use crate::BuiltinResult;

pub(super) async fn generate(
    lower: &[f64],
    upper: &[f64],
    count: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<Vec<f64>>> {
    if lower.is_empty() || lower.len() != upper.len() || count == 0 {
        return Err(invalid(
            "Latin hypercube sample dimensions must be positive",
        ));
    }
    lower
        .len()
        .checked_mul(count)
        .ok_or_else(|| invalid("Latin hypercube extent overflow"))?;
    for (&a, &b) in lower.iter().zip(upper) {
        if !a.is_finite() || !b.is_finite() || a > b {
            return Err(invalid(
                "Latin hypercube sampling requires finite ordered intervals",
            ));
        }
        cooperation.completed(1).await?;
    }
    let mut points = buffer(count)?;
    let mut order = buffer(count)?;
    for index in 0..count {
        let mut point = buffer(lower.len())?;
        for _ in lower {
            point.push(0.0);
            cooperation.completed(1).await?;
        }
        points.push(point);
        order.push(index);
    }
    let runtime = cooperation.runtime().cloned();
    let mut turn = random::acquire_original_turn(runtime.as_ref(), "patternsearch").await?;
    for (coordinate, (&a, &b)) in lower.iter().zip(upper).enumerate() {
        // Each coordinate gets a new independent permutation under this SAME
        // ordered original stream, not a reused diagonal or rank-one sample.
        for (index, target) in order.iter_mut().enumerate() {
            *target = index;
            cooperation.completed(1).await?;
        }
        shuffle_in_turn(&mut turn, &mut order, cooperation).await?;
        let jitter = {
            let (context, budget) = cooperation.runtime_budget();
            if let Some(budget) = budget {
                random::generate_uniform_cooperative(
                    &mut turn,
                    count,
                    "patternsearch",
                    context,
                    budget,
                )
                .await?
            } else {
                random::generate_uniform(&mut turn, count, "patternsearch")?
            }
        };
        for ((point, &stratum), draw) in points.iter_mut().zip(&order).zip(jitter) {
            let position = ((stratum as f64 + 1.0) - draw) / (count as f64);
            let value = if a == b {
                a
            } else if a.signum() == b.signum() {
                (b - a).mul_add(position, a)
            } else {
                a.mul_add(1.0 - position, b * position)
            };
            if !value.is_finite() || value < a || value > b {
                return Err(invalid(
                    "Latin hypercube coordinate is numerically unrepresentable",
                ));
            }
            point[coordinate] = value;
            cooperation.completed(1).await?;
        }
    }
    drop(turn);
    Ok(points)
}
