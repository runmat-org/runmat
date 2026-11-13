//! Host-side modeled merit refinement with the same original SQP session.
//! Only one best and one active working point exist; starts are not retained
//! as another population, and none invoke the expensive objective source.
use super::{invalid, rbf::Training, refinement::Session, region::Region, NAME};
use crate::builtins::{
    common::random,
    math::optim::{
        common::{affine_coordinate, ScalarCooperation},
        fmincon::internal::Refinement,
    },
};
use crate::BuiltinResult;

pub(super) enum Start {
    Point(Vec<f64>),
    /// Already solved by this same frozen-model session and derived region.
    /// This is never an InitialPoints or expensive user-observation carrier.
    Refined(Refinement),
}
pub(super) async fn search<T: Training, E: Training>(
    seed: Start,
    region: &mut Region<'_>,
    session: &mut Session<'_, T, E>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Refinement>> {
    cooperation.completed(0).await?;
    if !session.matches_region(region) {
        return Err(invalid(
            "multistart region has a different original model owner",
        ));
    }
    let initial = match &seed {
        Start::Point(point) => point,
        Start::Refined(result) => &result.point,
    };
    if initial.len() != region.variables() {
        return Err(invalid(
            "multistart point changed its original variable extent",
        ));
    }
    for (column, &value) in initial.iter().enumerate() {
        let (lower, upper) = region.bounds();
        if !value.is_finite() || value < lower[column] || value > upper[column] {
            return Err(invalid("multistart point is outside its derived region"));
        }
        cooperation.completed(1).await?;
    }
    for index in 0..region.integers().len() {
        let column = region.integers()[index];
        if !region.pin_integer(column, initial[column])? {
            return Ok(None);
        }
        cooperation.completed(1).await?;
    }
    let (lower, upper) = region.bounds();
    let free = free_count(lower, upper, cooperation).await?;
    let starts = if free == 0 {
        1
    } else if region.dimensions() <= 10 {
        5
    } else {
        20
    };
    let (mut first, mut best, completed) = match seed {
        Start::Point(point) => (Some(point), None, 0),
        Start::Refined(result) => {
            if !session.preferred(&result, None) {
                return Ok(None);
            }
            (None, Some(result), 1)
        }
    };
    for _ in completed..starts {
        cooperation.completed(0).await?;
        if session.expired() {
            break;
        }
        let point = match first.take() {
            Some(point) => point,
            None => random_start(lower, upper, free, cooperation).await?,
        };
        let candidate = session.solve(point, lower, upper, cooperation).await?;
        if session.preferred(&candidate, best.as_ref()) {
            best = Some(candidate);
        }
    }
    Ok(best)
}

async fn free_count(
    lower: &[f64],
    upper: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<usize> {
    let mut count = 0usize;
    for (&lower, &upper) in lower.iter().zip(upper) {
        count += usize::from(lower != upper);
        cooperation.completed(1).await?;
    }
    Ok(count)
}
async fn random_start(
    lower: &[f64],
    upper: &[f64],
    free: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    crate::context::host_allocation::require_unbounded_workspace(
        cooperation.runtime(),
        NAME,
        "finite host-memory execution has no complete multistart workspace bound",
    )?;
    let mut point = runmat_value::exact_capacity_vec(lower.len()).map_err(|source| {
        crate::build_runtime_error("surrogateopt: multistart point allocation failed")
            .with_builtin(NAME)
            .with_source(source)
            .build()
    })?;
    let mut turn = random::acquire_original_turn(cooperation.runtime(), NAME).await?;
    let values = {
        let (runtime, budget) = cooperation.runtime_budget();
        match budget {
            Some(budget) => {
                random::generate_uniform_cooperative(&mut turn, free, NAME, runtime, budget).await?
            }
            None => random::generate_uniform(&mut turn, free, NAME)?,
        }
    };
    drop(turn);
    let mut values = values.into_iter();
    for (&lower, &upper) in lower.iter().zip(upper) {
        point.push(if lower == upper {
            lower
        } else {
            affine_coordinate::interpolate(
                lower,
                upper,
                values.next().expect("validated free start count"),
            )
            .ok_or_else(|| invalid("finite multistart coordinate is unrepresentable"))?
        });
        cooperation.completed(1).await?;
    }
    Ok(point)
}
