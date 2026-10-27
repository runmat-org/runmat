//! Index-only poll permutations under the original ordered random stream.
use super::super::buffer;
use crate::builtins::{common::random, math::optim::common::ScalarCooperation};
use crate::BuiltinResult;
use runmat_builtins::OptimPatternOrder;

pub(super) enum Order {
    Consecutive,
    Random(Vec<usize>),
}
impl Order {
    pub async fn new(
        selected: OptimPatternOrder,
        count: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        match selected {
            OptimPatternOrder::Consecutive | OptimPatternOrder::Success => {
                return Ok(Self::Consecutive)
            }
            OptimPatternOrder::Random if count < 2 => return Ok(Self::Consecutive),
            OptimPatternOrder::Random => {}
        }
        // This is an index permutation, never a dense direction matrix.
        // Finish fallible storage admission before consuming stream words.
        let mut indices = buffer(count)?;
        for index in 0..count {
            indices.push(index);
            cooperation.completed(1).await?;
        }
        let runtime = cooperation.runtime().cloned();
        let mut turn = random::acquire_original_turn(runtime.as_ref(), "patternsearch").await?;
        super::super::random::shuffle_in_turn(&mut turn, &mut indices, cooperation).await?;
        // No arbitrary objective, nonlcon, observer or worker call owns a turn.
        drop(turn);
        Ok(Self::Random(indices))
    }
    pub fn at(&self, index: usize) -> usize {
        match self {
            Self::Consecutive => index,
            Self::Random(indices) => indices[index],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn permutation_uses_original_stream_is_reproducible_and_releases_before_callbacks() {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let foreign = crate::user_functions::test_support::exact_fixture_runtime();
        let _foreign = foreign.enter();
        let mut cooperation = ScalarCooperation::for_builtin(
            runmat_builtins::PATTERNSEARCH_CATALOG_ENTRY.identity,
            Some(&runtime),
        );
        let mut first = Vec::new();
        for pass in 0..2 {
            let mut turn = random::acquire_turn(Some(&runtime), "patternsearch")
                .await
                .unwrap();
            random::set_seed_in_turn(&mut turn, 42).unwrap();
            drop(turn);
            let order = Order::new(OptimPatternOrder::Random, 600, &mut cooperation)
                .await
                .unwrap();
            let actual: Vec<_> = (0..600).map(|index| order.at(index)).collect();
            let mut sorted = actual.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..600).collect::<Vec<_>>());
            assert_ne!(actual, sorted);
            if pass == 0 {
                first = actual;
            } else {
                assert_eq!(actual, first);
            }
            let turn = random::try_acquire_turn(Some(&runtime), "callback").unwrap();
            drop(turn);
        }
        let mut turn = random::acquire_turn(Some(&runtime), "patternsearch")
            .await
            .unwrap();
        random::set_seed_in_turn(&mut turn, 42).unwrap();
        let before = random::snapshot_in_turn(&mut turn).unwrap();
        drop(turn);
        let order = Order::new(OptimPatternOrder::Consecutive, 600, &mut cooperation)
            .await
            .unwrap();
        assert_eq!(order.at(499), 499);
        let mut turn = random::try_acquire_turn(Some(&runtime), "callback").unwrap();
        let after = random::snapshot_in_turn(&mut turn).unwrap();
        assert_eq!(before, after, "Consecutive must not draw random words");
        drop(turn);
        let mut foreign_turn = random::acquire_turn(Some(&foreign), "foreign snapshot")
            .await
            .unwrap();
        let foreign_before = random::snapshot_in_turn(&mut foreign_turn).unwrap();
        drop(foreign_turn);
        let mut context_free = ScalarCooperation::for_builtin(
            runmat_builtins::PATTERNSEARCH_CATALOG_ENTRY.identity,
            None,
        );
        Order::new(OptimPatternOrder::Random, 17, &mut context_free)
            .await
            .unwrap();
        let mut foreign_turn = random::acquire_turn(Some(&foreign), "foreign snapshot")
            .await
            .unwrap();
        assert_eq!(
            random::snapshot_in_turn(&mut foreign_turn).unwrap(),
            foreign_before,
            "context-free Pattern must not consume a foreign ambient stream"
        );
    }
}
