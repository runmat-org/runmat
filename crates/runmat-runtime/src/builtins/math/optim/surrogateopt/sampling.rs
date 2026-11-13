//! Adaptive sampler cycle and local random proposals. Integer sampling uses
//! original-stream words over the actual integer interval, not a rounded
//! continuous draw with biased half-width endpoints.
use super::{domain::BoxView, invalid, NAME};
use crate::builtins::{common::random, math::optim::common::ScalarCooperation};
use crate::BuiltinResult;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
mod metric;
pub(super) use metric::Metric;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Kind {
    Random,
    Orthogonal,
    Coordinate,
    Crossover,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Problem {
    Continuous,
    Binary,
    BinaryLinear,
    Integer,
}

impl Problem {
    pub(super) async fn classify(
        domain: &BoxView<'_>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut integers = 0usize;
        let mut binary = true;
        for &column in domain.free_columns() {
            let integer = domain.integers().binary_search(&column).is_ok();
            if integer {
                integers += 1;
                let (lower, upper) = domain.effective_bounds(column);
                binary &= lower == 0.0 && upper == 1.0;
            }
            cooperation.completed(1).await?;
        }
        Ok(if integers == 0 {
            Self::Continuous
        } else if integers == domain.dimensions() && binary {
            if domain.linear().rows() == 0 {
                Self::Binary
            } else {
                Self::BinaryLinear
            }
        } else {
            Self::Integer
        })
    }
    pub(super) fn sampler(self, cycle: usize) -> BuiltinResult<Kind> {
        if cycle >= 4 {
            return Err(invalid("invalid adaptive weight cycle"));
        }
        let kinds = match self {
            Self::Continuous => [
                Kind::Random,
                Kind::Random,
                Kind::Orthogonal,
                Kind::Coordinate,
            ],
            Self::Binary => [Kind::Random, Kind::Random, Kind::Crossover, Kind::Crossover],
            Self::BinaryLinear => [
                Kind::Orthogonal,
                Kind::Orthogonal,
                Kind::Crossover,
                Kind::Crossover,
            ],
            Self::Integer => [
                Kind::Orthogonal,
                Kind::Crossover,
                Kind::Orthogonal,
                Kind::Coordinate,
            ],
        };
        Ok(kinds[cycle])
    }
}

pub(super) async fn random_point(
    domain: &BoxView<'_>,
    anchor: &[f64],
    scale: f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    cooperation.completed(0).await?;
    if anchor.len() != domain.variables()
        || !scale.is_finite()
        || !(0.0..=0.8).contains(&scale)
        || scale == 0.0
    {
        return Err(invalid("invalid adaptive random sampling geometry"));
    }
    let mut continuous = 0usize;
    for (column, &value) in anchor.iter().enumerate() {
        let (lower, upper) = domain.effective_bounds(column);
        let integer = domain.integers().binary_search(&column).is_ok();
        if !value.is_finite() || value < lower || value > upper || integer && value.fract() != 0.0 {
            return Err(invalid(
                "adaptive anchor is outside its original box/integer domain",
            ));
        }
        continuous += usize::from(lower != upper && !integer);
        cooperation.completed(1).await?;
    }
    crate::context::host_allocation::require_unbounded_workspace(
        cooperation.runtime(),
        NAME,
        "finite host-memory execution has no complete adaptive random workspace bound",
    )?;
    let mut point = runmat_value::exact_capacity_vec(domain.variables()).map_err(|source| {
        crate::build_runtime_error("surrogateopt: random point allocation failed")
            .with_builtin(NAME)
            .with_source(source)
            .build()
    })?;
    let mut turn = random::acquire_original_turn(cooperation.runtime(), NAME).await?;
    let normals = {
        let (runtime, budget) = cooperation.runtime_budget();
        match budget {
            Some(budget) => {
                random::generate_normal_cooperative(&mut turn, continuous, NAME, runtime, budget)
                    .await?
            }
            None => random::generate_normal(&mut turn, continuous, NAME)?,
        }
    };
    let mut normals = normals.into_iter();
    let mut axis = 0usize;
    for (column, &center) in anchor.iter().enumerate() {
        let (lower, upper) = domain.effective_bounds(column);
        let value = if lower == upper {
            lower
        } else if domain.integers().binary_search(&column).is_ok() {
            let (lower, upper) = domain.integer_interval(axis, center, scale)?;
            let integer = uniform_integer(&lower, &upper, &mut turn, cooperation).await?;
            integer
                .to_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| invalid("sampled original integer is unrepresentable"))?
        } else {
            let z = normals.next().expect("validated continuous normal count");
            let unit = scale
                .mul_add(z, domain.unit_coordinate(anchor, axis))
                .clamp(0.0, 1.0);
            domain.interpolate_coordinate(axis, unit)?
        };
        point.push(value);
        axis += usize::from(lower != upper);
        cooperation.completed(1).await?;
    }
    drop(turn);
    Ok(point)
}

async fn uniform_integer(
    lower: &BigInt,
    upper: &BigInt,
    turn: &mut random::RandomTurn,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<BigInt> {
    let span = (upper - lower + BigInt::one())
        .to_biguint()
        .filter(|value| !value.is_zero())
        .ok_or_else(|| invalid("empty adaptive integer interval"))?;
    if span.is_one() {
        return Ok(lower.clone());
    }
    let bits = (&span - BigUint::one()).bits() as usize;
    // Finite binary64 integer endpoints have at most 1024 magnitude bits;
    // the full opposite-sign interval requires at most 1025 sample bits.
    if bits > 1025 {
        return Err(invalid("adaptive integer interval exceeds finite geometry"));
    }
    let words = bits.div_ceil(64);
    let mask = (BigUint::one() << bits) - BigUint::one();
    let mut storage = [0u64; 17];
    loop {
        cooperation.completed(0).await?;
        random::fill_integer_offsets(turn, 1u128 << 64, &mut storage[..words], NAME)?;
        let mut offset = BigUint::zero();
        for &word in storage[..words].iter().rev() {
            offset = (offset << 64usize) + BigUint::from(word);
            cooperation.completed(1).await?;
        }
        offset &= &mask;
        if offset < span {
            return Ok(lower + BigInt::from(offset));
        }
    }
}
