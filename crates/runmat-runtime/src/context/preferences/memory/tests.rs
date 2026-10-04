use super::*;
use crate::execution::RuntimeExecutionService;
use futures::executor::block_on;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_value::{CellArray, IntValue, StructValue};

fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cloned_authority_shares_commits_but_separate_authority_is_isolated() {
    let service = IsolatedMemoryPreferenceService::new();
    let clone = service.clone();
    let isolated = IsolatedMemoryPreferenceService::new();
    block_on(service.set_many(
        runtime(),
        "app".into(),
        vec![("wide".into(), Value::Int(IntValue::U64(u64::MAX)))],
    ))
    .expect("commit");
    assert_eq!(
        block_on(clone.group_values(runtime(), "app".into()))
            .expect("shared read")
            .unwrap()["wide"],
        Value::Int(IntValue::U64(u64::MAX))
    );
    assert!(block_on(isolated.group_values(runtime(), "app".into()))
        .expect("isolated read")
        .is_none());
    let reopened = IsolatedMemoryPreferenceService::from_document_bytes(
        &block_on(service.document_bytes()).unwrap(),
    )
    .expect("versioned document");
    assert_eq!(
        block_on(reopened.group_values(runtime(), "app".into()))
            .unwrap()
            .unwrap()["wide"],
        Value::Int(IntValue::U64(u64::MAX))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn late_unsupported_value_and_missing_name_leave_transaction_unchanged() {
    let service = IsolatedMemoryPreferenceService::new();
    let error = block_on(service.set_many(
        runtime(),
        "app".into(),
        vec![
            ("first".into(), Value::Num(2.0)),
            (
                "later".into(),
                Value::FunctionHandle("session_function".into()),
            ),
        ],
    ))
    .expect_err("session-bound callable cannot persist");
    assert!(matches!(error, PreferenceServiceError::ValueEncoding(_)));
    assert!(block_on(service.group_values(runtime(), "app".into()))
        .unwrap()
        .is_none());

    let error = block_on(service.read_or_create_many(
        runtime(),
        "app".into(),
        vec!["first".into(), "missing".into()],
        None,
    ))
    .expect_err("missing later name");
    assert!(matches!(error, PreferenceServiceError::Missing { .. }));
    assert!(block_on(service.group_values(runtime(), "app".into()))
        .unwrap()
        .is_none());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn defaults_are_one_ordered_transaction_and_nested_bits_round_trip() {
    let service = IsolatedMemoryPreferenceService::new();
    let mut structure = StructValue::new();
    structure.insert("wide", Value::Int(IntValue::I64(i64::MIN)));
    structure.insert("nan", Value::Num(f64::from_bits(0x7ff8_0000_0000_0042)));
    let nested = Value::Cell(CellArray::new(vec![Value::Struct(structure)], 1, 1).unwrap());
    let values = block_on(service.read_or_create_many(
        runtime(),
        "app".into(),
        vec!["same".into(), "same".into()],
        Some(vec![nested.clone(), Value::Num(0.0)]),
    ))
    .expect("commit first default once");
    assert_eq!(values, vec![nested.clone(), nested.clone()]);
    let stored = block_on(service.group_values(runtime(), "app".into()))
        .unwrap()
        .unwrap();
    let Value::Cell(cell) = &stored["same"] else {
        panic!("nested cell");
    };
    let Value::Struct(structure) = &cell.data[0] else {
        panic!("nested structure");
    };
    assert_eq!(
        structure.fields["wide"],
        Value::Int(IntValue::I64(i64::MIN))
    );
    let Value::Num(number) = &structure.fields["nan"] else {
        panic!("nested number");
    };
    assert_eq!(number.to_bits(), 0x7ff8_0000_0000_0042);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn unpolled_and_dropped_mutations_have_no_publication() {
    let service = IsolatedMemoryPreferenceService::new();
    let future = service.set_many(
        runtime(),
        "app".into(),
        vec![("name".into(), Value::Num(1.0))],
    );
    drop(future);
    assert!(block_on(service.group_values(runtime(), "app".into()))
        .unwrap()
        .is_none());
    assert_eq!(
        service.visit_retained_values(&mut |_| panic!("no runtime value retained")),
        Some(())
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cancelled_generation_cannot_publish_or_reuse_another_authority() {
    let service = IsolatedMemoryPreferenceService::new();
    let cancellation = CancellationSource::new();
    cancellation.cancel(CancellationReason::User);
    let cancelled = runtime().with_cancellation_token(cancellation.token());
    let error = block_on(service.set_many(
        cancelled,
        "app".into(),
        vec![("name".into(), Value::Num(1.0))],
    ))
    .unwrap_err();
    assert!(matches!(error, PreferenceServiceError::Storage(_)));
    assert!(block_on(service.group_values(runtime(), "app".into()))
        .unwrap()
        .is_none());
}
