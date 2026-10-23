use super::*;
use crate::execution::value_codec::encode_durable_value;
use crate::execution::RuntimeExecutionService;
use futures::executor::block_on;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_value::{CellArray, CharArray, DynamicPropertyDef, IntValue};

#[path = "tests/lifecycle.rs"]
mod lifecycle;

fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn isolated_user_path_is_distinct_and_reopens_with_exact_text_and_revision() {
    let service = IsolatedMemoryPreferenceService::new();
    let initial = block_on(RuntimeUserPathService::read(&service, runtime())).unwrap();
    assert_eq!(initial.setting, crate::context::UserPathSetting::Default);
    assert_eq!(initial.revision, 0);
    let set = block_on(RuntimeUserPathService::commit(
        &service,
        runtime(),
        UserPathMutation::Set("relative/MiXeD".into()),
    ))
    .unwrap();
    assert_eq!(set.revision, 1);
    assert_eq!(
        set.setting,
        crate::context::UserPathSetting::Custom("relative/MiXeD".into())
    );
    let reopened = IsolatedMemoryPreferenceService::from_document_bytes(
        &block_on(service.document_bytes()).unwrap(),
    )
    .unwrap();
    assert_eq!(
        block_on(RuntimeUserPathService::read(&reopened, runtime())).unwrap(),
        set
    );
    assert!(block_on(reopened.all_groups(runtime())).unwrap().is_empty());
    let clear = block_on(RuntimeUserPathService::commit(
        &reopened,
        runtime(),
        UserPathMutation::Clear,
    ))
    .unwrap();
    assert_eq!(clear.revision, 2);
    assert_eq!(clear.setting, crate::context::UserPathSetting::Cleared);
    let reset = block_on(RuntimeUserPathService::commit(
        &reopened,
        runtime(),
        UserPathMutation::Reset,
    ))
    .unwrap();
    assert_eq!(reset.revision, 3);
    assert_eq!(reset.setting, crate::context::UserPathSetting::Default);
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
fn ignored_defaults_are_not_encoded_and_missing_defaults_publish_atomically() {
    let service = IsolatedMemoryPreferenceService::new();
    block_on(service.set_many(
        runtime(),
        "app".into(),
        vec![("existing".into(), Value::Num(7.0))],
    ))
    .unwrap();
    let ignored = Value::FunctionHandle("session_function".into());
    let values = block_on(service.read_or_create_many(
        runtime(),
        "app".into(),
        vec!["existing".into(), "new".into(), "new".into()],
        Some(vec![ignored.clone(), Value::Num(9.0), ignored.clone()]),
    ))
    .expect("only first missing default is encoded");
    assert_eq!(
        values,
        vec![Value::Num(7.0), Value::Num(9.0), Value::Num(9.0)]
    );
    let existing = block_on(service.read_or_create_many(
        runtime(),
        "app".into(),
        vec!["existing".into()],
        Some(vec![ignored.clone()]),
    ))
    .expect("ignored callable default does not narrow existing-value form");
    assert_eq!(existing, vec![Value::Num(7.0)]);
    let error = block_on(service.read_or_create_many(
        runtime(),
        "app".into(),
        vec!["would-insert".into(), "invalid".into()],
        Some(vec![Value::Num(2.0), ignored]),
    ))
    .expect_err("later invalid missing default blocks entire batch");
    assert!(matches!(error, PreferenceServiceError::ValueEncoding(_)));
    let stored = block_on(service.group_values(runtime(), "app".into()))
        .unwrap()
        .unwrap();
    assert_eq!(stored.len(), 2);
    assert!(!stored.contains_key("would-insert"));
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
    let mut structure = runmat_value::UnboundedStructValueBuilder::new();
    structure.insert("wide", Value::Int(IntValue::I64(i64::MIN)));
    structure.insert("nan", Value::Num(f64::from_bits(0x7ff8_0000_0000_0042)));
    let nested =
        Value::Cell(CellArray::new(vec![Value::Struct(structure.finish())], 1, 1).unwrap());
    let values = block_on(service.read_or_create_many(
        runtime(),
        "app".into(),
        vec!["same".into(), "same".into()],
        Some(vec![nested.clone(), Value::Num(0.0)]),
    ))
    .expect("commit first default once");
    assert_eq!(values.len(), 2);
    let expected = encode_durable_value(&nested).unwrap();
    for value in &values {
        assert_eq!(encode_durable_value(value).unwrap(), expected);
    }
    let stored = block_on(service.group_values(runtime(), "app".into()))
        .unwrap()
        .unwrap();
    let Value::Cell(cell) = &stored["same"] else {
        panic!("nested cell");
    };
    let Value::Struct(structure) = &cell.data()[0] else {
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
    assert!(matches!(
        error,
        PreferenceServiceError::Execution(crate::execution::ExecutionServiceError::Cancelled)
    ));
    assert!(block_on(service.group_values(runtime(), "app".into()))
        .unwrap()
        .is_none());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn admitted_character_ranks_and_dynamic_definition_names_survive_reopen() {
    let service = IsolatedMemoryPreferenceService::new();
    let mut object = runmat_value::UnboundedObjectBuilder::new("PreferencePayload");
    let definition = DynamicPropertyDef::new(
        "definition_name".into(),
        runmat_types::ClassIdentity::new("PreferencePayload").unwrap(),
    );
    object.insert_dynamic("map_key".into(), definition);
    let entries = vec![
        (
            "rank_one".into(),
            Value::CharArray(
                CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![2]).unwrap(),
            ),
        ),
        (
            "empty_rank".into(),
            Value::CharArray(
                CharArray::from_code_units_with_shape(Vec::new(), Vec::new()).unwrap(),
            ),
        ),
        (
            "paged".into(),
            Value::CharArray(
                CharArray::from_unicode_scalars_with_shape(vec!['x', 'y'], vec![1, 1, 2]).unwrap(),
            ),
        ),
        ("object".into(), Value::Object(object.finish())),
    ];
    block_on(service.set_many(runtime(), "app".into(), entries.clone())).unwrap();
    let current = block_on(service.group_values(runtime(), "app".into()))
        .unwrap()
        .unwrap();
    let reopened = IsolatedMemoryPreferenceService::from_document_bytes(
        &block_on(service.document_bytes()).unwrap(),
    )
    .unwrap();
    let durable = block_on(reopened.group_values(runtime(), "app".into()))
        .unwrap()
        .unwrap();
    for (name, value) in entries {
        assert_eq!(current[&name], value);
        assert_eq!(durable[&name], value);
    }
}
