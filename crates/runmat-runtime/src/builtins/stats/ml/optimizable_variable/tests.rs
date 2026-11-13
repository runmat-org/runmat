use super::*;
use futures::executor::block_on;
use runmat_value::{IntegerStorage, ObjectInstance};

use crate::builtins::stats::ml::bayesian_test_provider as provider_fixture;

async fn optimizable_variable_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    name: Value,
    range: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    super::optimizable_variable_builtin(
        runtime,
        name,
        range,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
}

fn object(value: Value) -> ObjectInstance {
    let Value::Object(object) = value else {
        panic!("expected object");
    };
    object
}

fn typed_integer_range(storage: IntegerStorage) -> Value {
    let len = storage.len();
    let tensor = Tensor::new_integer(storage, vec![1, len]).expect("integer range");
    Value::Tensor(tensor)
}

#[test]
fn builds_real_variable_with_defaults() {
    let out = block_on(optimizable_variable_builtin(
        None,
        Value::String("depth".into()),
        Value::Tensor(Tensor::new(vec![1.0, 10.0], vec![1, 2]).unwrap()),
        vec![],
    ))
    .expect("optimizableVariable");
    let object = object(out);
    assert!(object.class_name().is(CLASS_IDENTITY));
    assert_eq!(
        object.properties().get("Name"),
        Some(&Value::String("depth".into()))
    );
    assert_eq!(
        object.properties().get("Type"),
        Some(&Value::String("real".into()))
    );
    assert_eq!(
        object.properties().get("Transform"),
        Some(&Value::String("none".into()))
    );
    assert_eq!(
        object.properties().get("Optimize"),
        Some(&Value::Bool(true))
    );
}

#[test]
fn last_effective_option_value_retains_previous_override_semantics() {
    let result = block_on(optimizable_variable_builtin(
        None,
        Value::String("depth".into()),
        Value::Tensor(Tensor::new(vec![1.0, 10.0], vec![1, 2]).unwrap()),
        vec![
            Value::String("Type".into()),
            Value::String("invalid-but-overridden".into()),
            Value::String("Type".into()),
            Value::String("real".into()),
        ],
    ))
    .expect("only the last effective Type is validated");
    assert_eq!(
        object(result).properties().get("Type"),
        Some(&Value::String("real".into()))
    );
}

#[test]
fn invalid_host_type_precedes_resident_range_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = provider_fixture::RejectDownloadProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let resident = provider.allocation();
    let arguments = vec![
        Value::String("depth".into()),
        Value::GpuTensor(resident.handle.clone()),
        Value::String("Type".into()),
        Value::String("invalid".into()),
    ];
    let error = block_on(crate::call_builtin_async("optimizableVariable", &arguments))
        .expect_err("invalid host Type before Range download");
    assert_eq!(error.identifier(), ERROR_INVALID_OPTION.identifier);
    assert_eq!(provider.downloads(), 0);
    resident
        .probe
        .with_data(&resident.handle, |_| ())
        .expect("borrowed Range remains live");
}

#[test]
fn numeric_resident_text_options_fail_before_range_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = provider_fixture::RejectDownloadProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let resident = provider.allocation();
    for option in ["Type", "Transform"] {
        let error = block_on(optimizable_variable_builtin(
            None,
            Value::String("depth".into()),
            Value::GpuTensor(resident.handle.clone()),
            vec![
                Value::String("Optimize".into()),
                Value::GpuTensor(resident.handle.clone()),
                Value::String(option.into()),
                Value::GpuTensor(resident.handle.clone()),
            ],
        ))
        .expect_err("numeric resident text control must be rejected");
        assert_eq!(error.identifier(), ERROR_INVALID_OPTION.identifier);
        assert!(error.message().contains(option));
        assert_eq!(provider.downloads(), 0);
        resident
            .probe
            .with_data(&resident.handle, |_| ())
            .expect("borrowed range and controls remain live");
    }
}

#[test]
fn overridden_numeric_resident_type_does_not_change_effective_option() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = provider_fixture::RejectDownloadProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let resident = provider.allocation();
    let result = block_on(optimizable_variable_builtin(
        None,
        Value::String("depth".into()),
        Value::Tensor(Tensor::new(vec![1.0, 10.0], vec![1, 2]).unwrap()),
        vec![
            Value::String("Type".into()),
            Value::GpuTensor(resident.handle.clone()),
            Value::String("Type".into()),
            Value::String("real".into()),
        ],
    ))
    .expect("last effective Type determines admission");
    assert_eq!(
        object(result).properties().get("Type"),
        Some(&Value::String("real".into()))
    );
    assert_eq!(provider.downloads(), 0);
    resident
        .probe
        .with_data(&resident.handle, |_| ())
        .expect("ignored resident control remains live");
}

#[test]
fn invalid_host_control_precedes_another_deferred_control() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = provider_fixture::RejectDownloadProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let resident = provider.allocation();
    let arguments = vec![
        Value::String("depth".into()),
        Value::Tensor(Tensor::new(vec![1.0, 10.0], vec![1, 2]).unwrap()),
        Value::String("Optimize".into()),
        Value::GpuTensor(resident.handle.clone()),
        Value::String("Type".into()),
        Value::String("invalid".into()),
    ];
    let error = block_on(crate::call_builtin_async("optimizableVariable", &arguments))
        .expect_err("invalid host Type before resident Optimize download");
    assert_eq!(error.identifier(), ERROR_INVALID_OPTION.identifier);
    assert_eq!(provider.downloads(), 0);
    resident
        .probe
        .with_data(&resident.handle, |_| ())
        .expect("borrowed option remains live");
}

#[test]
fn invalid_host_range_precedes_resident_option_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = provider_fixture::RejectDownloadProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let resident = provider.allocation();
    let arguments = vec![
        Value::String("depth".into()),
        Value::Tensor(Tensor::new(vec![10.0, 1.0], vec![1, 2]).unwrap()),
        Value::String("Optimize".into()),
        Value::GpuTensor(resident.handle.clone()),
    ];
    let error = block_on(crate::call_builtin_async("optimizableVariable", &arguments))
        .expect_err("host-invalid Range before resident Optimize download");
    assert_eq!(error.identifier(), ERROR_INVALID_RANGE.identifier);
    assert_eq!(provider.downloads(), 0);
    resident
        .probe
        .with_data(&resident.handle, |_| ())
        .expect("borrowed option remains live");
}

#[test]
fn failed_range_download_keeps_nested_cause_and_borrowed_allocation() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = provider_fixture::RejectDownloadProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let resident = provider.allocation();
    let error = block_on(optimizable_variable_builtin(
        None,
        Value::String("depth".into()),
        Value::GpuTensor(resident.handle.clone()),
        vec![],
    ))
    .expect_err("fixture download failure");
    assert_eq!(error.identifier(), ERROR_FLOW.identifier);
    let source = error.source.as_ref().expect("retained gather error");
    let mut found = source
        .to_string()
        .contains("Bayesian fixture download rejected");
    let mut cause = source.source();
    while let Some(next) = cause {
        found |= next
            .to_string()
            .contains("Bayesian fixture download rejected");
        cause = next.source();
    }
    assert!(
        found,
        "original provider cause must remain in the source chain"
    );
    assert_eq!(provider.downloads(), 1);
    resident
        .probe
        .with_data(&resident.handle, |_| ())
        .expect("failed gather must not retire caller allocation");
}

#[test]
fn builds_integer_log_variable_with_optimize_false() {
    let out = block_on(optimizable_variable_builtin(
        None,
        Value::String("trees".into()),
        Value::Tensor(Tensor::new(vec![0.0, 1000.0], vec![1, 2]).unwrap()),
        vec![
            Value::String("Type".into()),
            Value::String("integer".into()),
            Value::String("Transform".into()),
            Value::String("log".into()),
            Value::String("Optimize".into()),
            Value::Bool(false),
        ],
    ))
    .expect("optimizableVariable integer");
    let object = object(out);
    assert_eq!(
        object.properties().get("Type"),
        Some(&Value::String("integer".into()))
    );
    assert_eq!(
        object.properties().get("Transform"),
        Some(&Value::String("log".into()))
    );
    assert_eq!(
        object.properties().get("Optimize"),
        Some(&Value::Bool(false))
    );
}

#[test]
fn typed_integer_range_bounds_are_read_from_exact_storage() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let range = typed_integer_range(IntegerStorage::I16(vec![0, 1000]));
    let out = block_on(optimizable_variable_builtin(
        None,
        Value::String("trees".into()),
        range,
        vec![
            Value::String("Type".into()),
            Value::String("integer".into()),
            Value::String("Transform".into()),
            Value::String("log".into()),
        ],
    ))
    .expect("optimizableVariable integer");
    let object = object(out);
    assert_eq!(
        object.properties().get("Type"),
        Some(&Value::String("integer".into()))
    );
    let Some(Value::Tensor(range)) = object.properties().get("Range") else {
        panic!("expected tensor Range");
    };
    assert_eq!(
        range.integer_storage(),
        Some(&IntegerStorage::I16(vec![0, 1000]))
    );
    assert_eq!(range.materialize_f64(), vec![0.0, 1000.0]);
}

#[test]
fn builds_categorical_variable_from_cellstr() {
    let categories = Value::Cell(
        CellArray::new(
            vec![Value::String("linear".into()), Value::String("rbf".into())],
            1,
            2,
        )
        .unwrap(),
    );
    let out = block_on(optimizable_variable_builtin(
        None,
        Value::String("kernel".into()),
        categories.clone(),
        vec![
            Value::String("Type".into()),
            Value::String("categorical".into()),
        ],
    ))
    .expect("optimizableVariable categorical");
    let object = object(out);
    assert_eq!(
        object.properties().get("Type"),
        Some(&Value::String("categorical".into()))
    );
    assert_eq!(object.properties().get("Range"), Some(&categories));
}

#[test]
fn infers_categorical_type_from_text_range() {
    let out = block_on(optimizable_variable_builtin(
        None,
        Value::String("method".into()),
        Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
        vec![],
    ))
    .expect("optimizableVariable categorical inferred");
    let object = object(out);
    assert_eq!(
        object.properties().get("Type"),
        Some(&Value::String("categorical".into()))
    );
}

#[test]
fn accepts_explicit_categorical_scalar_string_range() {
    let out = block_on(optimizable_variable_builtin(
        None,
        Value::String("method".into()),
        Value::String("rbf".into()),
        vec![
            Value::String("Type".into()),
            Value::String("categorical".into()),
        ],
    ))
    .expect("optimizableVariable scalar categorical");
    let object = object(out);
    assert_eq!(
        object.properties().get("Range"),
        Some(&Value::String("rbf".into()))
    );
    assert_eq!(
        object.properties().get("Type"),
        Some(&Value::String("categorical".into()))
    );
}

#[test]
fn rejects_invalid_ranges_and_options() {
    let err = block_on(optimizable_variable_builtin(
        None,
        Value::String("bad".into()),
        Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap()),
        vec![],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("finite and increasing"));

    let err = block_on(optimizable_variable_builtin(
        None,
        Value::String("bad".into()),
        Value::Tensor(Tensor::new(vec![1.5, 3.0], vec![1, 2]).unwrap()),
        vec![
            Value::String("Type".into()),
            Value::String("integer".into()),
        ],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("whole numbers"));

    let err = block_on(optimizable_variable_builtin(
        None,
        Value::String("bad".into()),
        Value::Tensor(Tensor::new(vec![0.0, 3.0], vec![1, 2]).unwrap()),
        vec![
            Value::String("Transform".into()),
            Value::String("log".into()),
        ],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("positive real"));

    let err = block_on(optimizable_variable_builtin(
        None,
        Value::String("bad".into()),
        Value::Tensor(Tensor::new(vec![-1.0, 3.0], vec![1, 2]).unwrap()),
        vec![
            Value::String("Type".into()),
            Value::String("integer".into()),
            Value::String("Transform".into()),
            Value::String("log".into()),
        ],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("nonnegative integer"));

    let err = block_on(optimizable_variable_builtin(
        None,
        Value::String("bad".into()),
        Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap()),
        vec![Value::String("Type".into()), Value::String("int".into())],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("unsupported Type"));

    let err = block_on(optimizable_variable_builtin(
        None,
        Value::String("bad".into()),
        Value::StringArray(StringArray::new(vec!["a".into(), "a".into()], vec![1, 2]).unwrap()),
        vec![
            Value::String("Type".into()),
            Value::String("categorical".into()),
        ],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("unique"));
}

#[test]
fn wide_integer_bounds_require_exact_binary64_conversion() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let exact_lower = 1_u64 << 54;
    let exact_upper = exact_lower + 4;
    let accepted = block_on(optimizable_variable_builtin(
        None,
        Value::String("wide".into()),
        typed_integer_range(IntegerStorage::U64(vec![exact_lower, exact_upper])),
        vec![
            Value::String("Type".into()),
            Value::String("integer".into()),
        ],
    ))
    .expect("aligned wide bounds remain exact");
    assert!(object(accepted).class_name().is(CLASS_IDENTITY));

    let error = block_on(optimizable_variable_builtin(
        None,
        Value::String("rounded".into()),
        typed_integer_range(IntegerStorage::U64(vec![
            (1_u64 << 53) + 1,
            (1_u64 << 53) + 3,
        ])),
        vec![
            Value::String("Type".into()),
            Value::String("integer".into()),
        ],
    ))
    .expect_err("rounded bounds must not become a different search domain");
    assert_eq!(error.identifier(), ERROR_INVALID_RANGE.identifier);
}

#[test]
fn matlab_compatibility_rejects_typed_integer_range_at_admission() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(optimizable_variable_builtin(
        None,
        Value::String("trees".into()),
        typed_integer_range(IntegerStorage::I16(vec![0, 1000])),
        vec![],
    ))
    .expect_err("typed integer bounds are a RunMat-only form");
    assert_eq!(
        error.identifier(),
        OPTIMIZABLE_VARIABLE_INTEGER_RANGE_EXTENSION.error_identifier
    );
}
