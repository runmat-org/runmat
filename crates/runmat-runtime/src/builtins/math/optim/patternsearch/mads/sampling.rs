//! Integer sampling from the existing ordered random word owner.
use super::{
    super::{buffer, invalid},
    level::Level,
};
use crate::builtins::{common::random, math::optim::common::ScalarCooperation};
use crate::BuiltinResult;
use num_bigint::{BigInt, Sign};
use num_traits::{One, Zero};

/// Uniform integer in [-q+1,q-1], q=2^ell. Binary64 mesh levels need at most
/// 538 random bits; this bound comes from the floating exponent, not a solver
/// dimension cutoff. No floating modulo/floor or independent RNG is used.
pub(super) async fn off_diagonal(
    turn: &mut random::RandomTurn,
    level: Level,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<BigInt> {
    if level.bits() == 0 {
        return Ok(BigInt::zero());
    }
    let bits = level.bits() + 1;
    let count = bits.div_ceil(64);
    let mut words = [0u64; 9];
    if count > words.len() {
        return Err(invalid("invalid binary64 MADS level"));
    }
    let mut digits = buffer(count * 2)?;
    let tail = bits % 64;
    let mask = if tail == 0 {
        u64::MAX
    } else {
        (1u64 << tail) - 1
    };
    loop {
        random::fill_integer_offsets(turn, 1u128 << 64, &mut words[..count], "patternsearch")?;
        words[count - 1] &= mask;
        cooperation.completed(count as u64).await?;
        // The only rejected code is 2q-1. Every other exact integer code has
        // the same number of underlying source words.
        if words[..count - 1].iter().all(|&word| word == u64::MAX) && words[count - 1] == mask {
            continue;
        }
        for &word in &words[..count] {
            digits.push(word as u32);
            digits.push((word >> 32) as u32);
        }
        let sampled = BigInt::new(Sign::Plus, digits);
        return Ok(sampled - (level.integer() - BigInt::one()));
    }
}
pub(super) async fn diagonal(
    turn: &mut random::RandomTurn,
    level: Level,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<BigInt> {
    let mut sign = [0u64; 1];
    random::fill_integer_offsets(turn, 2, &mut sign, "patternsearch")?;
    cooperation.completed(1).await?;
    let q = level.integer();
    Ok(if sign[0] == 0 { -q } else { q })
}
pub(super) async fn pivot(
    turn: &mut random::RandomTurn,
    dimension: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<usize> {
    if dimension == 0 {
        return Err(invalid("MADS variable dimension must be positive"));
    }
    let mut value = [0u64; 1];
    random::fill_integer_offsets(turn, dimension as u128, &mut value, "patternsearch")?;
    cooperation.completed(1).await?;
    usize::try_from(value[0]).map_err(|_| invalid("MADS pivot exceeds platform extent"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn coefficients_keep_exact_wide_integer_range_and_original_seed_replay() {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let mut cooperation = ScalarCooperation::for_builtin(
            runmat_builtins::PATTERNSEARCH_CATALOG_ENTRY.identity,
            Some(&runtime),
        );
        for mesh in [1.0, 0.0625, 2f64.powi(-120), f64::from_bits(1)] {
            let level = Level::for_mesh(mesh).unwrap();
            let q = level.integer();
            let mut first = None;
            for _ in 0..2 {
                let mut turn = random::acquire_turn(Some(&runtime), "patternsearch")
                    .await
                    .unwrap();
                random::set_seed_in_turn(&mut turn, 17).unwrap();
                let value = off_diagonal(&mut turn, level, &mut cooperation)
                    .await
                    .unwrap();
                assert!(value > -&q && value < q);
                let sign = diagonal(&mut turn, level, &mut cooperation).await.unwrap();
                assert!(sign == q || sign == -&q);
                let actual_pivot = pivot(&mut turn, 7, &mut cooperation).await.unwrap();
                assert!(actual_pivot < 7);
                if let Some(prior) = &first {
                    assert_eq!(prior, &(value, sign, actual_pivot));
                } else {
                    first = Some((value, sign, actual_pivot));
                }
            }
        }
    }
}
