#[cfg(any(feature = "plot-core", test))]
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};

use super::KeyAtom;

mod builder;
use builder::{BuildProgress, GroupIndexBuilder};

/// The semantic group grammar and execution service remain separate failure
/// domains. Callers adapt each at their public builtin boundary.
#[cfg(any(feature = "plot-core", test))]
#[derive(Debug)]
pub(crate) enum GroupIndexBuildError {
    Invalid(String),
    Service(ExecutionServiceError),
}

/// A tuple key; each component retains its typed identity until presentation.
pub(crate) type GroupKey = Vec<KeyAtom>;

/// The ordering contract selected by the builtin accepting the grouping value.
#[derive(Clone, Debug)]
pub(crate) enum KeyOrder {
    Sorted,
    FirstAppearance,
    Explicit(Vec<GroupKey>),
}

/// A deterministic index over nonmissing grouping observations.
#[derive(Clone, Debug)]
pub(crate) struct GroupIndex {
    pub(crate) ids: Vec<f64>,
    pub(crate) keys: Vec<GroupKey>,
    pub(crate) first_rows: Vec<Option<usize>>,
    pub(crate) row_groups: Vec<Vec<usize>>,
}

impl GroupIndex {
    pub(crate) fn build(rows: Vec<Option<GroupKey>>, order: KeyOrder) -> Result<Self, String> {
        Self::build_from_builder(GroupIndexBuilder::new(rows, order))
    }

    /// The synchronous and cooperative drivers run the same bounded exact-key
    /// state machine; neither driver enters a nested executor.
    #[cfg(any(feature = "plot-core", test))]
    pub(crate) async fn build_cooperative(
        rows: Vec<Option<GroupKey>>,
        order: KeyOrder,
        budget: Option<&mut CpuCooperationBudget>,
    ) -> Result<Self, GroupIndexBuildError> {
        let mut builder = GroupIndexBuilder::new(rows, order);
        let Some(budget) = budget else {
            return Self::build_from_builder(builder).map_err(GroupIndexBuildError::Invalid);
        };
        budget
            .checkpoint(0)
            .await
            .map_err(GroupIndexBuildError::Service)?;
        loop {
            let mut units = 0;
            match builder
                .step(Some(&mut units))
                .map_err(GroupIndexBuildError::Invalid)?
            {
                BuildProgress::Work => budget
                    .checkpoint(units)
                    .await
                    .map_err(GroupIndexBuildError::Service)?,
                BuildProgress::Complete(index) => return Ok(index),
            }
        }
    }

    fn build_from_builder(mut builder: GroupIndexBuilder) -> Result<Self, String> {
        loop {
            match builder.step(None)? {
                BuildProgress::Work => {}
                BuildProgress::Complete(index) => return Ok(index),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::CpuCooperationPolicy;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use runmat_value::IntValue;
    use std::cell::Cell;
    use std::future::Future;
    use std::num::NonZeroU64;
    use std::pin::Pin;
    use std::rc::Rc;
    use std::task::{Context, Poll};

    fn text(value: &str) -> GroupKey {
        vec![KeyAtom::Text(value.into())]
    }

    #[test]
    fn sorted_and_first_appearance_are_distinct_policies() {
        let rows = vec![Some(text("b")), Some(text("a")), Some(text("b"))];
        let sorted = GroupIndex::build(rows.clone(), KeyOrder::Sorted).unwrap();
        let stable = GroupIndex::build(rows, KeyOrder::FirstAppearance).unwrap();
        assert_eq!(sorted.ids, vec![2.0, 1.0, 2.0]);
        assert_eq!(stable.ids, vec![1.0, 2.0, 1.0]);
    }

    #[test]
    fn explicit_order_retains_empty_levels_and_missing_rows() {
        let rows = vec![Some(text("b")), None, Some(text("b"))];
        let index = GroupIndex::build(
            rows,
            KeyOrder::Explicit(vec![text("a"), text("b"), text("c")]),
        )
        .unwrap();
        assert_eq!(index.ids[0], 2.0);
        assert!(index.ids[1].is_nan());
        assert_eq!(index.ids[2], 2.0);
        assert_eq!(index.first_rows, vec![None, Some(0), None]);
        assert_eq!(index.row_groups, vec![vec![], vec![0, 2], vec![]]);
    }

    #[test]
    fn synchronous_consumer_is_safe_inside_an_async_executor() {
        let rows = vec![Some(text("b")), Some(text("a")), Some(text("b"))];
        let (sorted, stable) = futures::executor::block_on(async {
            (
                GroupIndex::build(rows.clone(), KeyOrder::Sorted).unwrap(),
                GroupIndex::build(rows, KeyOrder::FirstAppearance).unwrap(),
            )
        });
        assert_eq!(sorted.ids, vec![2.0, 1.0, 2.0]);
        assert_eq!(stable.ids, vec![1.0, 2.0, 1.0]);
    }

    #[test]
    fn cooperative_budget_charges_text_bytes_and_factor_arity() {
        let source = CancellationSource::new();
        let turns = Rc::new(Cell::new(0));
        let counted = turns.clone();
        let mut budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(16).unwrap()),
            source.token(),
            move || {
                counted.set(counted.get() + 1);
                Box::pin(async { Ok(()) })
            },
        );
        let key = vec![KeyAtom::Text("x".repeat(32)), KeyAtom::Logical(true)];
        let index = futures::executor::block_on(GroupIndex::build_cooperative(
            vec![Some(key.clone())],
            KeyOrder::Sorted,
            Some(&mut budget),
        ))
        .unwrap();
        assert_eq!(index.keys, vec![key]);
        assert!(turns.get() >= 1, "long text must charge more than one row");
    }

    #[test]
    fn cooperative_index_preserves_exact_order_and_row_membership() {
        let rows = vec![Some(text("b")), None, Some(text("a")), Some(text("b"))];
        for order in [KeyOrder::Sorted, KeyOrder::FirstAppearance] {
            let expected = GroupIndex::build(rows.clone(), order.clone()).unwrap();
            let source = CancellationSource::new();
            let turns = Rc::new(Cell::new(0));
            let counted = turns.clone();
            let mut budget = CpuCooperationBudget::with_test_turn(
                CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
                source.token(),
                move || {
                    counted.set(counted.get() + 1);
                    Box::pin(async { Ok(()) })
                },
            );
            let actual = futures::executor::block_on(GroupIndex::build_cooperative(
                rows.clone(),
                order,
                Some(&mut budget),
            ))
            .unwrap();
            for (actual, expected) in actual.ids.iter().zip(&expected.ids) {
                if expected.is_nan() {
                    assert!(actual.is_nan());
                } else {
                    assert_eq!(actual, expected);
                }
            }
            assert_eq!(actual.row_groups, expected.row_groups);
            assert_eq!(actual.first_rows, expected.first_rows);
            assert_eq!(actual.keys, expected.keys);
            assert!(turns.get() >= rows.len());
        }
    }

    #[test]
    fn explicit_cooperative_order_matches_sync_and_preserves_diagnostics() {
        let rows = vec![Some(text("b")), None, Some(text("b"))];
        let order = KeyOrder::Explicit(vec![text("a"), text("b"), text("c")]);
        let expected = GroupIndex::build(rows.clone(), order.clone()).unwrap();
        let actual =
            futures::executor::block_on(GroupIndex::build_cooperative(rows, order, None)).unwrap();
        assert_eq!(actual.keys, expected.keys);
        assert_eq!(actual.first_rows, expected.first_rows);
        assert_eq!(actual.row_groups, expected.row_groups);
        assert_eq!(actual.ids[0], expected.ids[0]);
        assert!(actual.ids[1].is_nan());
        assert_eq!(actual.ids[2], expected.ids[2]);

        let duplicate = KeyOrder::Explicit(vec![text("a"), text("a")]);
        let sync = GroupIndex::build(vec![], duplicate.clone()).unwrap_err();
        let cooperative =
            futures::executor::block_on(GroupIndex::build_cooperative(vec![], duplicate, None))
                .unwrap_err();
        assert_eq!(sync, "explicit group order contains a duplicate key");
        assert!(matches!(cooperative, GroupIndexBuildError::Invalid(message) if message == sync));

        let missing = KeyOrder::Explicit(vec![text("a")]);
        let sync = GroupIndex::build(vec![Some(text("b"))], missing.clone()).unwrap_err();
        let cooperative = futures::executor::block_on(GroupIndex::build_cooperative(
            vec![Some(text("b"))],
            missing,
            None,
        ))
        .unwrap_err();
        assert_eq!(sync, "group observation is absent from the explicit order");
        assert!(matches!(cooperative, GroupIndexBuildError::Invalid(message) if message == sync));
    }

    #[test]
    fn empty_tuples_and_mixed_wide_integer_atoms_keep_exact_identity() {
        let wide = (1u64 << 53) + 1;
        let keys = vec![
            vec![],
            vec![
                KeyAtom::Integer(IntValue::U64(wide)),
                KeyAtom::Text("x".into()),
            ],
            vec![
                KeyAtom::Integer(IntValue::I64(wide as i64)),
                KeyAtom::Text("x".into()),
            ],
            vec![
                KeyAtom::Integer(IntValue::U64(wide + 1)),
                KeyAtom::Text("x".into()),
            ],
        ];
        let rows = keys.iter().cloned().map(Some).collect();
        let sync = GroupIndex::build(rows, KeyOrder::FirstAppearance).unwrap();
        assert_eq!(sync.keys.len(), 3);
        assert_eq!(sync.ids, vec![1.0, 2.0, 2.0, 3.0]);
        assert_eq!(sync.row_groups, vec![vec![0], vec![1, 2], vec![3]]);
        let cooperative = futures::executor::block_on(GroupIndex::build_cooperative(
            keys.into_iter().map(Some).collect(),
            KeyOrder::FirstAppearance,
            None,
        ))
        .unwrap();
        assert_eq!(cooperative.keys, sync.keys);
        assert_eq!(cooperative.ids, sync.ids);
    }

    struct PendingTurn {
        dropped: Rc<Cell<bool>>,
    }

    impl Future for PendingTurn {
        type Output = Result<(), ExecutionServiceError>;

        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Pending
        }
    }

    impl Drop for PendingTurn {
        fn drop(&mut self) {
            self.dropped.set(true);
        }
    }

    #[test]
    fn abandoning_pending_cooperative_turn_drops_it_without_publishing() {
        let source = CancellationSource::new();
        let entered = Rc::new(Cell::new(false));
        let dropped = Rc::new(Cell::new(false));
        let turn_entered = entered.clone();
        let turn_dropped = dropped.clone();
        let mut budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || {
                turn_entered.set(true);
                Box::pin(PendingTurn {
                    dropped: turn_dropped.clone(),
                })
            },
        );
        let mut pending = Box::pin(GroupIndex::build_cooperative(
            vec![Some(text("a"))],
            KeyOrder::Sorted,
            Some(&mut budget),
        ));
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(matches!(pending.as_mut().poll(&mut context), Poll::Pending));
        assert!(entered.get());
        assert!(!dropped.get());
        drop(pending);
        assert!(
            dropped.get(),
            "pending service turn must be retired on drop"
        );
    }

    #[test]
    fn cooperative_index_cancellation_keeps_service_error_typed() {
        let source = Rc::new(CancellationSource::new());
        let cancel_on_turn = source.clone();
        let mut budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || {
                cancel_on_turn.cancel(CancellationReason::User);
                Box::pin(async { Ok(()) })
            },
        );
        let error = futures::executor::block_on(GroupIndex::build_cooperative(
            vec![Some(text("a")), Some(text("b"))],
            KeyOrder::Sorted,
            Some(&mut budget),
        ))
        .expect_err("cancelled grouping cannot publish an index");
        assert!(matches!(
            error,
            GroupIndexBuildError::Service(ExecutionServiceError::Cancelled)
        ));
    }
}
