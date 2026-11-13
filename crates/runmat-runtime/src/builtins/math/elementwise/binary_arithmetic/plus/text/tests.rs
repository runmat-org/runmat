use super::*;
use futures::executor::block_on;
use runmat_value::{CellArray, CharArray, IntValue, NumericStorage, Tensor};

mod support;
use support::{assert_text, texts};

#[test]
fn append_converts_and_expands_without_losing_class_or_column_major_order() {
    let unicode_rows = Value::CharArray(
        CharArray::from_unicode_scalars("ab cd ".chars().collect(), 2, 3).unwrap(),
    );
    let cells = Value::Cell(
        CellArray::new_with_shape(
            vec![
                Value::Num(1.0),
                Value::Num(2.0),
                Value::Num(3.0),
                Value::Num(4.0),
            ],
            vec![2, 2, 1],
        )
        .unwrap(),
    );
    let wide = Value::Tensor(
        Tensor::from_numeric_storage(
            NumericStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![1, 2],
        )
        .unwrap(),
    );
    let cases = [
        (
            Value::from("left"),
            Value::from("right"),
            vec!["leftright"],
            vec![1, 1],
        ),
        (
            Value::from("λ"),
            Value::CharArray(CharArray::from_unicode_scalars(vec!['🙂'], 1, 2).unwrap()),
            vec!["λ🙂"],
            vec![1, 1],
        ),
        (
            Value::from("id="),
            wide,
            vec!["id=9007199254740993", "id=18446744073709551615"],
            vec![1, 2],
        ),
        (
            Value::Int(IntValue::I64(i64::MIN)),
            Value::from("!"),
            vec!["-9223372036854775808!"],
            vec![1, 1],
        ),
        (
            texts(&["A", "B"], &[1, 2]),
            unicode_rows,
            vec!["Aab ", "Acd ", "Bab ", "Bcd "],
            vec![2, 2],
        ),
        (
            Value::from("x"),
            cells,
            vec!["x1", "x3", "x2", "x4"],
            vec![2, 2, 1],
        ),
        (
            Value::from("v="),
            Value::Bool(true),
            vec!["v=true"],
            vec![1, 1],
        ),
        (
            Value::Complex(2.0, -3.0),
            Value::from("i"),
            vec!["2-3ii"],
            vec![1, 1],
        ),
        (
            texts(&["<missing>"], &[1, 1]),
            Value::from("suffix"),
            vec!["<missing>suffix"],
            vec![1, 1],
        ),
        (Value::from("x"), texts(&[], &[0, 2]), vec![], vec![0, 2]),
    ];
    for (left, right, expected, shape) in cases {
        let output = block_on(crate::call_builtin_async("plus", &[left, right])).unwrap();
        assert_text(output.value().clone(), &expected, &shape);
    }
    assert!(block_on(concatenate(
        None,
        &texts(&["a", "b"], &[1, 2]),
        &texts(&["x", "y", "z"], &[1, 3])
    ))
    .is_err());
    let nonscalar = Value::Cell(CellArray::new(vec![texts(&["a", "b"], &[1, 2])], 1, 1).unwrap());
    assert!(block_on(concatenate(None, &Value::from("x"), &nonscalar)).is_err());
}

#[test]
fn managed_append_retains_exact_output_claims_and_releases_every_owner() {
    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use std::rc::Rc;
    struct Parallel(ManagedHostEffectLedger);
    impl RuntimeParallelService for Parallel {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..Default::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    let ledger = ManagedHostEffectLedger::new(1_048_576);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
        );
    let left = texts(&["λ", ""], &[2, 1]);
    let right = texts(&["A", "B"], &[1, 2]);
    let result = block_on(concatenate(Some(&runtime), &left, &right)).unwrap();
    let Value::StringArray(array) = &result else {
        panic!("string array")
    };
    assert!(array.shape.is_admitted());
    assert!(array.data().iter().all(StringElement::has_retained_claim));
    let live = ledger.snapshot().live_reserved_bytes;
    assert!(live > 0);
    let clone = result.clone();
    assert_eq!(
        ledger.snapshot().live_reserved_bytes,
        live,
        "clones share the original owners"
    );
    drop(result);
    assert_eq!(ledger.snapshot().live_reserved_bytes, live);
    drop(clone);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_text(
        block_on(concatenate(
            Some(&runtime),
            &Value::from("x"),
            &texts(&[], &[0, 1]),
        ))
        .unwrap(),
        &[],
        &[0, 1],
    );
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    let tiny = ManagedHostEffectLedger::new(1);
    let limited = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(tiny.clone()))),
        );
    assert!(block_on(concatenate(Some(&limited), &left, &right)).is_err());
    assert_eq!(
        tiny.snapshot().live_reserved_bytes,
        0,
        "failed admission leaves no stage or output claims"
    );
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let cancelled = runtime.with_cancellation_token(cancellation.token());
    cancellation.cancel(runmat_execution::CancellationReason::User);
    assert!(block_on(concatenate(Some(&cancelled), &left, &right)).is_err());
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn typed_missing_addition_distinguishes_literal_display_text() {
    let source = Value::StringArray(
        StringArray::from_elements_unbounded(
            vec![
                StringElement::text("<missing>"),
                StringElement::text("<Missing>"),
                StringElement::text(""),
                StringElement::Missing,
            ],
            vec![1, 4],
        )
        .unwrap(),
    );
    let result = block_on(concatenate(None, &source, &Value::string("!"))).unwrap();
    let Value::StringArray(result) = result else {
        panic!("string array result")
    };
    assert_eq!(
        result.data(),
        &[
            StringElement::text("<missing>!"),
            StringElement::text("<Missing>!"),
            StringElement::text("!"),
            StringElement::Missing
        ]
    );
    assert!(!result.data()[3].has_retained_claim());
}
