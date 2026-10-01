use super::*;
use futures::executor::block_on;
use runmat_value::IntegerStorage;

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
        Value::String("depth".into()),
        Value::Tensor(Tensor::new(vec![1.0, 10.0], vec![1, 2]).unwrap()),
        vec![],
    ))
    .expect("optimizableVariable");
    let object = object(out);
    assert!(object.class_name.is(CLASS_IDENTITY));
    assert_eq!(
        object.properties.get("Name"),
        Some(&Value::String("depth".into()))
    );
    assert_eq!(
        object.properties.get("Type"),
        Some(&Value::String("real".into()))
    );
    assert_eq!(
        object.properties.get("Transform"),
        Some(&Value::String("none".into()))
    );
    assert_eq!(object.properties.get("Optimize"), Some(&Value::Bool(true)));
}

#[test]
fn builds_integer_log_variable_with_optimize_false() {
    let out = block_on(optimizable_variable_builtin(
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
        object.properties.get("Type"),
        Some(&Value::String("integer".into()))
    );
    assert_eq!(
        object.properties.get("Transform"),
        Some(&Value::String("log".into()))
    );
    assert_eq!(object.properties.get("Optimize"), Some(&Value::Bool(false)));
}

#[test]
fn typed_integer_range_bounds_are_read_from_exact_storage() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let range = typed_integer_range(IntegerStorage::I16(vec![0, 1000]));
    let out = block_on(optimizable_variable_builtin(
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
        object.properties.get("Type"),
        Some(&Value::String("integer".into()))
    );
    let Some(Value::Tensor(range)) = object.properties.get("Range") else {
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
        object.properties.get("Type"),
        Some(&Value::String("categorical".into()))
    );
    assert_eq!(object.properties.get("Range"), Some(&categories));
}

#[test]
fn infers_categorical_type_from_text_range() {
    let out = block_on(optimizable_variable_builtin(
        Value::String("method".into()),
        Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
        vec![],
    ))
    .expect("optimizableVariable categorical inferred");
    let object = object(out);
    assert_eq!(
        object.properties.get("Type"),
        Some(&Value::String("categorical".into()))
    );
}

#[test]
fn accepts_explicit_categorical_scalar_string_range() {
    let out = block_on(optimizable_variable_builtin(
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
        object.properties.get("Range"),
        Some(&Value::String("rbf".into()))
    );
    assert_eq!(
        object.properties.get("Type"),
        Some(&Value::String("categorical".into()))
    );
}

#[test]
fn rejects_invalid_ranges_and_options() {
    let err = block_on(optimizable_variable_builtin(
        Value::String("bad".into()),
        Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap()),
        vec![],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("finite and increasing"));

    let err = block_on(optimizable_variable_builtin(
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
        Value::String("bad".into()),
        Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap()),
        vec![Value::String("Type".into()), Value::String("int".into())],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("unsupported Type"));

    let err = block_on(optimizable_variable_builtin(
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
        Value::String("wide".into()),
        typed_integer_range(IntegerStorage::U64(vec![exact_lower, exact_upper])),
        vec![
            Value::String("Type".into()),
            Value::String("integer".into()),
        ],
    ))
    .expect("aligned wide bounds remain exact");
    assert!(object(accepted).class_name.is(CLASS_IDENTITY));

    let error = block_on(optimizable_variable_builtin(
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
