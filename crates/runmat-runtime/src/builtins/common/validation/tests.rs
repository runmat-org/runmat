use super::*;
mod class_shape;
use crate::builtins::common::identifiers::MATLAB_NAME_LENGTH_MAX;
use crate::builtins::common::test_support;
use runmat_value::{
    ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray, StringArray,
    StructArray, StructValue, Tensor,
};

fn ok(builtin: &str, args: Vec<Value>) {
    dispatch_validator(builtin, args).unwrap_or_else(|err| {
        panic!("{builtin} unexpectedly failed: {err}");
    });
}

fn err(builtin: &str, args: Vec<Value>) {
    assert!(
        dispatch_validator(builtin, args).is_err(),
        "{builtin} unexpectedly passed"
    );
}

fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new_2d(data, rows, cols).unwrap())
}

fn sparse(values: Vec<f64>) -> Value {
    Value::SparseTensor(SparseTensor::new(2, 2, vec![0, 1, 1], vec![0], values).unwrap())
}

#[test]
fn fixed_width_integer_binary64_exactness_uses_significant_bits() {
    for value in [
        IntValue::I8(i8::MIN),
        IntValue::I16(i16::MIN),
        IntValue::I32(i32::MIN),
        IntValue::U8(u8::MAX),
        IntValue::U16(u16::MAX),
        IntValue::U32(u32::MAX),
        IntValue::U64(1_u64 << 54),
    ] {
        assert!(integer_is_exact_f64(&value), "{value:?} must be exact");
    }
    assert!(!integer_is_exact_f64(&IntValue::U64((1_u64 << 53) + 1)));
    assert!(!integer_is_exact_f64(&IntValue::I64(-((1_i64 << 53) + 1))));
    assert!(integer_is_exact_f64(&IntValue::U64(0)));
}

#[test]
fn resident_integer_exactness_is_class_conservative() {
    use runmat_accelerate_api::{GpuTensorHandle, IntegerElementType};

    let narrow = GpuTensorHandle {
        shape: vec![1, 1],
        device_id: u32::MAX,
        buffer_id: u64::MAX - 1,
        descriptor: Default::default(),
    }
    .with_numeric_descriptor(
        IntegerElementType::U32.into(),
        runmat_accelerate_api::GpuTensorStorage::Real,
    );
    assert!(native_integer_value_is_exact_f64(&Value::GpuTensor(narrow)));
    let wide = GpuTensorHandle {
        shape: vec![1, 1],
        device_id: u32::MAX,
        buffer_id: u64::MAX - 2,
        descriptor: Default::default(),
    }
    .with_numeric_descriptor(
        IntegerElementType::I64.into(),
        runmat_accelerate_api::GpuTensorStorage::Real,
    );
    assert!(!native_integer_value_is_exact_f64(&Value::GpuTensor(wide)));
}

#[test]
fn resident_wide_integer_exactness_is_decided_from_gathered_values() {
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

    test_support::with_test_provider(|provider| {
        for (value, expected) in [
            (9_007_199_254_740_992_u64, true),
            (9_007_199_254_740_993_u64, false),
        ] {
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(std::slice::from_ref(&value)),
                    shape: &[1, 1],
                })
                .expect("upload resident uint64");
            assert_eq!(
                block_on(native_integer_value_is_exact_f64_async(&Value::GpuTensor(
                    handle.clone()
                )))
                .expect("exactness check"),
                expected
            );
            provider.free(&handle).expect("free resident uint64");
            runmat_accelerate_api::clear_handle_metadata(&handle);
        }
    });
}

fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape).unwrap())
}

#[test]
fn structural_validators_cover_all_native_integer_classes_without_conversion() {
    let cases = [
        (IntegerStorage::I8(vec![1]), "int8"),
        (IntegerStorage::I16(vec![1]), "int16"),
        (IntegerStorage::I32(vec![1]), "int32"),
        (IntegerStorage::I64(vec![1]), "int64"),
        (IntegerStorage::U8(vec![1]), "uint8"),
        (IntegerStorage::U16(vec![1]), "uint16"),
        (IntegerStorage::U32(vec![1]), "uint32"),
        (IntegerStorage::U64(vec![1]), "uint64"),
    ];

    for (storage, class_name) in cases {
        let value = integer_tensor(storage, vec![1, 1]);
        ok("mustBeReal", vec![value.clone()]);
        ok("mustBeScalarOrEmpty", vec![value.clone()]);
        err("mustBeSparse", vec![value.clone()]);
        ok(
            "mustBeUnderlyingType",
            vec![value.clone(), Value::String(class_name.into())],
        );
        ok("mustBeVector", vec![value.clone()]);
        err("mustBeText", vec![value.clone()]);
        err("mustBeTextScalar", vec![value.clone()]);
        err("mustBeValidVariableName", vec![value]);
    }
}

#[test]
fn sparse_and_vector_validators_apply_documented_empty_shape_rules() {
    let empty = integer_tensor(IntegerStorage::U16(vec![]), vec![0, 3]);
    ok("mustBeScalarOrEmpty", vec![empty.clone()]);
    ok("mustBeSparse", vec![empty.clone()]);
    err("mustBeVector", vec![empty.clone()]);
    ok(
        "mustBeVector",
        vec![empty, Value::String("allow-all-empties".into())],
    );

    let empty_vector = integer_tensor(IntegerStorage::I8(vec![]), vec![0, 1]);
    ok("mustBeVector", vec![empty_vector]);

    let multidimensional = integer_tensor(IntegerStorage::U32(vec![1, 2]), vec![1, 1, 2]);
    err("mustBeVector", vec![multidimensional]);
    let trailing_singleton = integer_tensor(IntegerStorage::U32(vec![1, 2]), vec![1, 2, 1]);
    ok("mustBeVector", vec![trailing_singleton]);

    err(
        "mustBeVector",
        vec![
            Value::Int(IntValue::U8(1)),
            Value::String("unsupported".into()),
        ],
    );
}

#[test]
fn sparse_integer_storage_satisfies_sparse_validation_without_materialization() {
    let sparse = SparseTensor::new_integer(
        2,
        2,
        vec![0, 1, 1],
        vec![0],
        IntegerStorage::I64(vec![9_007_199_254_740_993]),
    )
    .expect("sparse integer");
    ok("mustBeSparse", vec![Value::SparseTensor(sparse)]);
}

#[test]
fn resident_text_validators_reject_before_provider_lookup() {
    use runmat_accelerate_api::{GpuTensorHandle, IntegerElementType};

    let handle = GpuTensorHandle {
        shape: vec![1, 1],
        device_id: u32::MAX - 1,
        buffer_id: u64::MAX - 2,
        descriptor: Default::default(),
    }
    .with_numeric_descriptor(
        IntegerElementType::I32.into(),
        runmat_accelerate_api::GpuTensorStorage::Real,
    );
    for builtin in ["mustBeText", "mustBeTextScalar", "mustBeValidVariableName"] {
        let error = dispatch_validator(builtin, vec![Value::GpuTensor(handle.clone())])
            .expect_err("resident integer must fail text validation");
        let expected_identifier = format!("RunMat:{builtin}:ValidationFailed");
        assert_eq!(error.identifier(), Some(expected_identifier.as_str()));
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    }
    runmat_accelerate_api::clear_handle_metadata(&handle);
}

#[test]
fn resident_integer_structural_validators_do_not_read_freed_payloads() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![1, u64::MAX]), vec![1, 2])
            .expect("resident integer");
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
            .expect("upload resident integer");
        provider
            .free(&handle)
            .expect("free payload before predicates");
        runmat_accelerate_api::set_handle_logical(&handle, false);
        let value = Value::GpuTensor(handle.clone());

        ok("mustBeReal", vec![value.clone()]);
        err("mustBeScalarOrEmpty", vec![value.clone()]);
        err("mustBeSparse", vec![value.clone()]);
        ok(
            "mustBeUnderlyingType",
            vec![value.clone(), Value::String("uint64".into())],
        );
        ok("mustBeVector", vec![value]);
        runmat_accelerate_api::clear_handle_metadata(&handle);
    });
}

#[test]
fn numeric_validators_check_all_elements() {
    let ok = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    assert!(dispatch_validator("mustBePositive", vec![Value::Tensor(ok)]).is_ok());

    let bad = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
    assert!(dispatch_validator("mustBePositive", vec![Value::Tensor(bad)]).is_err());
}

#[test]
fn numeric_validators_read_typed_integer_storage_exactly() {
    let positive =
        Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![1, 2]).expect("positive");
    ok("mustBePositive", vec![Value::Tensor(positive)]);

    let negative =
        Tensor::new_integer(IntegerStorage::I16(vec![-1, -2]), vec![1, 2]).expect("negative");
    ok("mustBeNegative", vec![Value::Tensor(negative)]);

    let zero = Tensor::new_integer(IntegerStorage::I16(vec![0]), vec![1, 1]).expect("zero");
    err("mustBeNonzero", vec![Value::Tensor(zero)]);

    let wide = 9_007_199_254_740_993_u64;
    let adjacent = wide - 1;
    assert_eq!(wide as f64, adjacent as f64);
    let wide_nonzero =
        Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).expect("wide");
    ok("mustBeNonzero", vec![Value::Tensor(wide_nonzero)]);

    let complex_nonzero = ComplexTensor::new_integer(
        IntegerComplexStorage::new(
            IntegerStorage::U64(vec![0]),
            IntegerStorage::U64(vec![wide]),
        )
        .expect("complex integer storage"),
        vec![1, 1],
    )
    .expect("complex integer tensor");
    ok("mustBeNonzero", vec![Value::ComplexTensor(complex_nonzero)]);
}

#[test]
fn value_is_empty_uses_typed_integer_storage_length() {
    let scalar = Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("scalar");
    assert!(!value_is_empty(&Value::Tensor(scalar)));

    let empty =
        Tensor::new_integer(IntegerStorage::U16(Vec::new()), vec![0, 0]).expect("empty tensor");
    assert!(value_is_empty(&Value::Tensor(empty)));

    let complex = ComplexTensor::new_integer(
        IntegerComplexStorage::new(IntegerStorage::I16(vec![1]), IntegerStorage::I16(vec![0]))
            .expect("complex integer storage"),
        vec![1, 1],
    )
    .expect("complex integer tensor");
    assert!(!value_is_empty(&Value::ComplexTensor(complex)));

    let empty_complex = ComplexTensor::new_integer(
        IntegerComplexStorage::new(IntegerStorage::I16(vec![]), IntegerStorage::I16(vec![]))
            .expect("empty complex integer storage"),
        vec![0, 0],
    )
    .expect("empty complex integer tensor");
    assert!(value_is_empty(&Value::ComplexTensor(empty_complex)));
}

fn structure_array(shape: Vec<usize>) -> Value {
    let mut element = StructValue::new();
    element.insert("value", Value::Num(1.0));
    Value::StructArray(
        StructArray::new(vec![element; shape.iter().product()], shape).expect("structure array"),
    )
}

#[test]
fn structural_predicates_use_structure_array_shape() {
    let matrix = structure_array(vec![2, 2]);
    assert_eq!(value_shape_2d(&matrix), (2, 2));
    assert!(!value_is_empty(&matrix));
    assert!(!value_is_column(&matrix));
    assert!(!value_is_vector(&matrix).unwrap());
    assert!(value_matches_class(&matrix, "struct"));

    let column = structure_array(vec![3, 1]);
    assert!(value_is_column(&column));
    assert!(value_is_vector(&column).unwrap());

    let nd = structure_array(vec![1, 2, 2]);
    assert_eq!(value_shape_2d(&nd), (1, 2));
    assert!(!value_is_column(&nd));
    assert!(!value_is_vector(&nd).unwrap());

    let empty = Value::StructArray(
        StructArray::empty(vec!["value".into()], vec![0, 4]).expect("empty structure array"),
    );
    assert_eq!(value_shape_2d(&empty), (0, 4));
    assert!(value_is_empty(&empty));
    assert!(!value_is_vector(&empty).unwrap());
}

#[test]
fn finite_integer_and_nan_predicates_read_typed_integer_storage_exactly() {
    let tensor = Tensor::new_integer(IntegerStorage::I16(vec![-1, 0, 2]), vec![1, 3]).unwrap();
    let value = Value::Tensor(tensor);

    assert!(value_is_finite(&value));
    assert!(value_is_integer(&value));
    assert!(value_is_non_nan(&value));
    ok("mustBeFinite", vec![value.clone()]);
    ok("mustBeInteger", vec![value.clone()]);
    ok("mustBeNonNan", vec![value]);

    let sparse = Value::SparseTensor(
        SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            IntegerStorage::U8(vec![1, 2]),
        )
        .unwrap(),
    );
    assert!(value_is_finite(&sparse));
    assert!(value_is_integer(&sparse));
    assert!(value_is_non_nan(&sparse));
}

#[test]
fn real_and_integer_predicates_read_authoritative_complex_integer_storage() {
    let real_storage = IntegerStorage::I16(vec![1, -2]);
    let zero_imag = IntegerStorage::I16(vec![0, 0]);
    let real_complex = ComplexTensor::new_integer(
        IntegerComplexStorage::new(real_storage, zero_imag).unwrap(),
        vec![1, 2],
    )
    .unwrap();
    let value = Value::ComplexTensor(real_complex);
    assert!(value_is_finite(&value));
    assert!(value_is_real(&value));
    assert!(value_is_integer(&value));
    assert!(value_is_non_nan(&value));

    let nonreal_complex = ComplexTensor::new_integer(
        IntegerComplexStorage::new(IntegerStorage::I16(vec![1]), IntegerStorage::I16(vec![1]))
            .unwrap(),
        vec![1, 1],
    )
    .unwrap();
    let value = Value::ComplexTensor(nonreal_complex);
    assert!(!value_is_real(&value));
    assert!(value_is_integer(&value));
    ok("mustBeInteger", vec![value]);
}

#[test]
fn complex_sparse_validators_keep_both_components() {
    let value = Value::SparseTensor(
        SparseTensor::new_complex(
            2,
            2,
            vec![0, 2, 3],
            vec![0, 1, 1],
            vec![(1.0, 0.0), (2.0, -3.0), (f64::NAN, 4.0)],
        )
        .expect("complex sparse"),
    );
    assert!(!value_is_finite(&value));
    assert!(!value_is_real(&value));
    assert!(!value_is_integer(&value));
    assert!(!value_is_non_nan(&value));

    let integral = Value::SparseTensor(
        SparseTensor::new_complex(
            1,
            2,
            vec![0, 1, 2],
            vec![0, 0],
            vec![(1.0, 0.0), (2.0, -3.0)],
        )
        .expect("integral complex sparse"),
    );
    assert!(value_is_finite(&integral));
    assert!(!value_is_real(&integral));
    assert!(value_is_integer(&integral));
    assert!(value_is_non_nan(&integral));
}

#[test]
fn complex_validators_use_component_integrality_real_ordering_and_exact_membership() {
    ok("mustBeInteger", vec![Value::Complex(1.0, 2.0)]);
    err("mustBeInteger", vec![Value::Complex(1.0, 2.5)]);
    ok("mustBeNegative", vec![Value::Complex(-1.0, 9.0)]);
    ok("mustBePositive", vec![Value::Complex(1.0, -9.0)]);
    ok(
        "mustBeMember",
        vec![Value::Complex(1.0, 2.0), Value::Complex(1.0, 2.0)],
    );

    let wide = Value::ComplexTensor(
        ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![9_007_199_254_740_993]),
                IntegerStorage::U64(vec![7]),
            )
            .expect("integer components"),
            vec![1, 1],
        )
        .expect("complex integer tensor"),
    );
    ok("mustBeMember", vec![wide.clone(), wide]);
    ok("mustBeMember", vec![Value::Bool(true), Value::Num(1.0)]);
    ok("mustBeMember", vec![Value::Num(0.0), Value::Bool(false)]);
    ok(
        "mustBeInteger",
        vec![Value::CharArray(
            CharArray::new(vec!['a', 'b'], 1, 2).expect("character row"),
        )],
    );
    ok("mustBeNonzero", vec![Value::Num(f64::INFINITY)]);
    ok("mustBeNonzero", vec![Value::Num(f64::NAN)]);
    ok("mustBePositive", vec![Value::Num(f64::INFINITY)]);
    ok("mustBeNegative", vec![Value::Num(f64::NEG_INFINITY)]);
    ok(
        "mustBeNonnegative",
        vec![Value::Complex(f64::INFINITY, 7.0)],
    );
    ok(
        "mustBeNonpositive",
        vec![Value::Complex(f64::NEG_INFINITY, 7.0)],
    );
}

#[test]
fn resident_secondary_validator_operands_require_coherent_metadata() {
    test_support::with_test_provider(|provider| {
        let value = Tensor::new_integer(IntegerStorage::U8(vec![2]), vec![1, 1]).expect("value");
        let lower = Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).expect("lower");
        let value_handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &value)
            .expect("upload value");
        let mut lower_handle =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &lower)
                .expect("upload lower");
        lower_handle.descriptor.storage =
            Some(runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved);
        let error = dispatch_validator(
            "mustBeGreaterThan",
            vec![
                Value::GpuTensor(value_handle.clone()),
                Value::GpuTensor(lower_handle.clone()),
            ],
        )
        .expect_err("contradictory bound metadata must reject");
        assert_eq!(
            error.identifier(),
            Some("RunMat:validators:ProviderPayloadMismatch")
        );
        provider.free(&value_handle).ok();
        provider.free(&lower_handle).ok();
        runmat_accelerate_api::clear_handle_metadata(&value_handle);
        runmat_accelerate_api::clear_handle_metadata(&lower_handle);
    });
}

#[test]
fn threshold_validators_read_typed_sparse_and_complex_integer_storage() {
    let sparse =
        SparseTensor::new_integer(1, 2, vec![0, 1, 1], vec![0], IntegerStorage::U8(vec![1]))
            .unwrap();
    ok("mustBeNonnegative", vec![Value::SparseTensor(sparse)]);

    let complex = ComplexTensor::new_integer(
        IntegerComplexStorage::new(IntegerStorage::I16(vec![2]), IntegerStorage::I16(vec![0]))
            .unwrap(),
        vec![1, 1],
    )
    .unwrap();
    ok("mustBePositive", vec![Value::ComplexTensor(complex)]);
}

#[test]
fn member_validator_accepts_numeric_and_text_sets() {
    let allowed = Tensor::new(vec![1.0, 3.0, 5.0], vec![1, 3]).unwrap();
    assert!(dispatch_validator(
        "mustBeMember",
        vec![Value::Num(3.0), Value::Tensor(allowed)]
    )
    .is_ok());

    let allowed = StringArray::new(vec!["on".into(), "off".into()], vec![1, 2]).unwrap();
    assert!(dispatch_validator(
        "mustBeMember",
        vec![Value::String("on".into()), Value::StringArray(allowed)]
    )
    .is_ok());
}

#[test]
fn class_validators_use_native_integer_storage_metadata() {
    let dense = integer_tensor(
        IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
        vec![1, 2],
    );
    ok(
        "mustBeA",
        vec![dense.clone(), Value::String("integer".into())],
    );
    err("mustBeFloat", vec![dense.clone()]);
    err("mustBeA", vec![dense, Value::String("double".into())]);

    let sparse = Value::SparseTensor(
        SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 1],
            vec![1],
            IntegerStorage::I64(vec![i64::MIN]),
        )
        .unwrap(),
    );
    ok(
        "mustBeA",
        vec![sparse.clone(), Value::String("integer".into())],
    );
    err("mustBeFloat", vec![sparse.clone()]);
    err("mustBeA", vec![sparse, Value::String("double".into())]);

    let typed_complex = Value::ComplexTensor(
        ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::I16(vec![1]), IntegerStorage::I16(vec![2]))
                .unwrap(),
            vec![1, 1],
        )
        .unwrap(),
    );
    ok(
        "mustBeA",
        vec![typed_complex.clone(), Value::String("integer".into())],
    );
    err("mustBeFloat", vec![typed_complex]);
}

#[test]
fn class_validators_use_gpu_integer_metadata_without_gather() {
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

    test_support::with_test_provider(|provider| {
        let values = [u64::MAX, 9_007_199_254_740_993];
        let shape = [1usize, 2usize];
        let handle = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&values),
                shape: &shape,
            })
            .expect("upload integer gpu tensor");
        let gpu = Value::GpuTensor(handle);

        ok(
            "mustBeA",
            vec![gpu.clone(), Value::String("integer".into())],
        );
        ok("mustBeInteger", vec![gpu.clone()]);
        err("mustBeFloat", vec![gpu.clone()]);
        err("mustBeA", vec![gpu, Value::String("double".into())]);
    });
}

#[test]
fn must_be_a_distinguishes_explicit_gpuarray_from_automatic_residency() {
    test_support::with_test_provider(|provider| {
        let tensor =
            Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("integer source");
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
            .expect("upload integer source");
        let handle = handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic);
        ok(
            "mustBeA",
            vec![
                Value::GpuTensor(handle.clone()),
                Value::String("uint16".into()),
            ],
        );
        err(
            "mustBeA",
            vec![
                Value::GpuTensor(handle.clone()),
                Value::String("gpuArray".into()),
            ],
        );
        let handle = handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        ok(
            "mustBeA",
            vec![
                Value::GpuTensor(handle.clone()),
                Value::String("gpuArray".into()),
            ],
        );
        err(
            "mustBeA",
            vec![
                Value::GpuTensor(handle.clone()),
                Value::String("uint16".into()),
            ],
        );
        provider.free(&handle).expect("free resident source");
        runmat_accelerate_api::clear_handle_metadata(&handle);
    });
}

#[test]
fn member_validator_compares_native_integers_exactly() {
    let wide = 9_007_199_254_740_993_u64;
    let adjacent = wide - 1;
    assert_eq!(wide as f64, adjacent as f64);

    let allowed = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap();
    ok(
        "mustBeMember",
        vec![
            Value::Int(IntValue::U64(wide)),
            Value::Tensor(allowed.clone()),
        ],
    );
    err(
        "mustBeMember",
        vec![Value::Int(IntValue::U64(adjacent)), Value::Tensor(allowed)],
    );

    let sparse_allowed = SparseTensor::new_integer(
        2,
        2,
        vec![0, 1, 1],
        vec![1],
        IntegerStorage::U64(vec![wide]),
    )
    .unwrap();
    ok(
        "mustBeMember",
        vec![
            Value::Int(IntValue::U64(0)),
            Value::SparseTensor(sparse_allowed),
        ],
    );
}

#[test]
fn member_validator_does_not_equate_wide_integer_with_rounded_double() {
    let wide = 9_007_199_254_740_993_u64;
    let rounded = (wide - 1) as f64;
    assert_eq!(wide as f64, rounded);

    err(
        "mustBeMember",
        vec![Value::Int(IntValue::U64(wide)), Value::Num(rounded)],
    );
}

#[test]
fn text_and_varname_validators_follow_core_shapes() {
    assert!(dispatch_validator(
        "mustBeNonzeroLengthText",
        vec![Value::CharArray(CharArray::new_row("alpha"))]
    )
    .is_ok());
    assert!(isvarname_value(&Value::String("alpha_1".into())));
    assert!(!isvarname_value(&Value::String("1alpha".into())));
}

#[test]
fn isvarname_returns_false_for_all_integer_classes() {
    for value in [
        IntValue::I8(-1),
        IntValue::I16(-2),
        IntValue::I32(-3),
        IntValue::I64(i64::MIN),
        IntValue::U8(1),
        IntValue::U16(2),
        IntValue::U32(3),
        IntValue::U64(u64::MAX),
    ] {
        assert!(!isvarname_value(&Value::Int(value)));
    }
}

#[test]
fn isvarname_returns_false_for_resident_integer_without_gather() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
            .expect("integer tensor");
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
            .expect("upload integer");
        assert!(!isvarname_value(&Value::GpuTensor(handle)));
    });
}

#[test]
fn namedargs2cell_preserves_field_order() {
    let mut st = StructValue::new();
    st.insert("Name", Value::String("entry-a".into()));
    st.insert("Value", Value::Num(7.0));
    let out = namedargs2cell_value(Value::Struct(st)).expect("namedargs2cell");
    let Value::Cell(cell) = out else {
        panic!("expected cell");
    };
    assert_eq!(cell.rows, 1);
    assert_eq!(cell.cols, 4);
    assert_eq!(cell.data[0], Value::String("Name".into()));
    assert_eq!(cell.data[2], Value::String("Value".into()));
}

#[test]
fn namedargs2cell_rejects_top_level_integers_and_preserves_integer_fields_exactly() {
    let cases = [
        IntValue::I8(i8::MIN),
        IntValue::I16(i16::MIN),
        IntValue::I32(i32::MIN),
        IntValue::I64(i64::MIN),
        IntValue::U8(u8::MAX),
        IntValue::U16(u16::MAX),
        IntValue::U32(u32::MAX),
        IntValue::U64(u64::MAX),
    ];
    for value in cases {
        assert!(namedargs2cell_value(Value::Int(value.clone())).is_err());
        let mut structure = StructValue::new();
        structure.insert("Exact", Value::Int(value.clone()));
        let output = namedargs2cell_value(Value::Struct(structure))
            .expect("scalar struct with integer field");
        let Value::Cell(cell) = output else {
            panic!("expected name-value cell");
        };
        assert_eq!(cell.data[1], Value::Int(value));
    }

    let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 1],
        device_id: 0,
        buffer_id: 9_419_006,
        descriptor: Default::default(),
    });
    assert!(namedargs2cell_value(resident).is_err());
}

#[test]
fn validator_surface_accepts_and_rejects_representative_values() {
    let temp_dir = tempfile::tempdir().unwrap();
    let file_path = temp_dir.path().join("data.txt");
    std::fs::write(&file_path, "ok").unwrap();
    let dir_text = Value::String(temp_dir.path().to_string_lossy().into_owned());
    let file_text = Value::String(file_path.to_string_lossy().into_owned());

    ok(
        "mustBeA",
        vec![Value::Num(1.0), Value::String("double".into())],
    );
    err(
        "mustBeA",
        vec![Value::String("x".into()), Value::String("double".into())],
    );
    ok("mustBeColumn", vec![tensor(vec![1.0, 2.0], 2, 1)]);
    err("mustBeColumn", vec![tensor(vec![1.0, 2.0], 1, 2)]);
    ok("mustBeFile", vec![file_text.clone()]);
    err("mustBeFile", vec![dir_text.clone()]);
    ok("mustBeFolder", vec![dir_text.clone()]);
    err("mustBeFolder", vec![file_text.clone()]);
    ok("mustBeFinite", vec![tensor(vec![1.0, 2.0], 1, 2)]);
    err("mustBeFinite", vec![Value::Num(f64::INFINITY)]);
    ok("mustBeFloat", vec![Value::Num(1.0)]);
    err("mustBeFloat", vec![Value::Int(IntValue::I32(1))]);
    ok("mustBeInteger", vec![tensor(vec![1.0, 2.0], 1, 2)]);
    ok("mustBeInteger", vec![Value::Bool(true)]);
    ok(
        "mustBeInteger",
        vec![Value::LogicalArray(
            LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap(),
        )],
    );
    err("mustBeInteger", vec![Value::Num(1.5)]);
    ok("mustBeNumeric", vec![Value::Complex(1.0, 2.0)]);
    err("mustBeNumeric", vec![Value::String("1".into())]);
    ok(
        "mustBeNumericOrLogical",
        vec![Value::LogicalArray(
            LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap(),
        )],
    );
    err("mustBeNumericOrLogical", vec![Value::String("true".into())]);
    ok("mustBeReal", vec![Value::Complex(1.0, 0.0)]);
    err("mustBeReal", vec![Value::Complex(1.0, 1.0)]);
    ok("mustBeVector", vec![tensor(vec![1.0, 2.0], 1, 2)]);
    err("mustBeVector", vec![tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2)]);
    ok("mustBeScalarOrEmpty", vec![Value::Num(1.0)]);
    err("mustBeScalarOrEmpty", vec![tensor(vec![1.0, 2.0], 1, 2)]);
    ok("mustBeSparse", vec![sparse(vec![1.0])]);
    err("mustBeSparse", vec![Value::Num(1.0)]);
    ok("mustBeNonsparse", vec![Value::Num(1.0)]);
    err("mustBeNonsparse", vec![sparse(vec![1.0])]);
    ok(
        "mustBeText",
        vec![Value::CharArray(CharArray::new_row("abc"))],
    );
    err("mustBeText", vec![Value::Num(1.0)]);
    ok("mustBeTextScalar", vec![Value::String("abc".into())]);
    err(
        "mustBeTextScalar",
        vec![Value::StringArray(
            StringArray::new_2d(vec!["a".into(), "b".into()], 1, 2).unwrap(),
        )],
    );
    ok(
        "mustBeNonzeroLengthText",
        vec![Value::StringArray(
            StringArray::new_2d(vec!["a".into(), "b".into()], 1, 2).unwrap(),
        )],
    );
    err(
        "mustBeNonzeroLengthText",
        vec![Value::String(String::new())],
    );
    ok("mustBeNonempty", vec![Value::String("x".into())]);
    err(
        "mustBeNonempty",
        vec![Value::StringArray(
            StringArray::new_2d(vec![], 0, 0).unwrap(),
        )],
    );
    ok("mustBeNonmissing", vec![Value::Num(1.0)]);
    err("mustBeNonmissing", vec![Value::Num(f64::NAN)]);
    ok("mustBeNonNan", vec![Value::Complex(1.0, 0.0)]);
    err("mustBeNonNan", vec![Value::Complex(f64::NAN, 0.0)]);
    ok(
        "mustBeUnderlyingType",
        vec![Value::Int(IntValue::I16(1)), Value::String("int16".into())],
    );
    err(
        "mustBeUnderlyingType",
        vec![Value::Bool(true), Value::String("double".into())],
    );
    ok(
        "mustBeValidVariableName",
        vec![Value::String("alpha_1".into())],
    );
    err(
        "mustBeValidVariableName",
        vec![Value::String("_alpha".into())],
    );
    ok(
        "mustBeMember",
        vec![
            Value::String("on".into()),
            Value::Cell(
                CellArray::new(
                    vec![Value::String("on".into()), Value::String("off".into())],
                    1,
                    2,
                )
                .unwrap(),
            ),
        ],
    );
    err(
        "mustBeMember",
        vec![
            Value::String("bad".into()),
            Value::Cell(
                CellArray::new(
                    vec![Value::String("on".into()), Value::String("off".into())],
                    1,
                    2,
                )
                .unwrap(),
            ),
        ],
    );
}

#[test]
fn numeric_threshold_validators_cover_boundaries() {
    ok("mustBePositive", vec![Value::Num(1.0)]);
    ok("mustBePositive", vec![Value::Bool(true)]);
    err("mustBePositive", vec![Value::Num(0.0)]);
    ok("mustBeNegative", vec![Value::Num(-1.0)]);
    err("mustBeNegative", vec![Value::Num(0.0)]);
    ok("mustBeNonnegative", vec![Value::Num(0.0)]);
    ok(
        "mustBeNonnegative",
        vec![Value::LogicalArray(
            LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap(),
        )],
    );
    err("mustBeNonnegative", vec![Value::Num(-1.0)]);
    ok("mustBeNonpositive", vec![Value::Num(0.0)]);
    err("mustBeNonpositive", vec![Value::Num(1.0)]);
    ok("mustBeNonzero", vec![Value::Complex(0.0, 1.0)]);
    err("mustBeNonzero", vec![Value::Num(0.0)]);
    ok("mustBeGreaterThan", vec![Value::Num(2.0), Value::Num(1.0)]);
    err("mustBeGreaterThan", vec![Value::Num(1.0), Value::Num(1.0)]);
    ok(
        "mustBeGreaterThanOrEqual",
        vec![Value::Num(1.0), Value::Num(1.0)],
    );
    err(
        "mustBeGreaterThanOrEqual",
        vec![Value::Num(0.0), Value::Num(1.0)],
    );
    ok("mustBeLessThan", vec![Value::Num(0.0), Value::Num(1.0)]);
    err("mustBeLessThan", vec![Value::Num(1.0), Value::Num(1.0)]);
    ok(
        "mustBeLessThanOrEqual",
        vec![Value::Num(1.0), Value::Num(1.0)],
    );
    err(
        "mustBeLessThanOrEqual",
        vec![Value::Num(2.0), Value::Num(1.0)],
    );
}

#[test]
fn threshold_validators_read_typed_integer_storage_exactly() {
    let lower = Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("lower");
    ok(
        "mustBeGreaterThan",
        vec![Value::Num(2.0), Value::Tensor(lower)],
    );

    let upper = Tensor::new_integer(IntegerStorage::U16(vec![3]), vec![1, 1]).expect("upper");
    ok(
        "mustBeLessThan",
        vec![Value::Num(2.0), Value::Tensor(upper)],
    );

    let range_lower =
        Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("range lower");
    let range_upper =
        Tensor::new_integer(IntegerStorage::U16(vec![3]), vec![1, 1]).expect("range upper");
    ok(
        "mustBeInRange",
        vec![
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1]).expect("range value"),
            ),
            Value::Tensor(range_lower),
            Value::Tensor(range_upper),
        ],
    );

    let wide = 9_007_199_254_740_993_u64;
    let adjacent = wide - 1;
    let rounded = adjacent as f64;
    assert_eq!(wide as f64, rounded);

    let wide_value =
        Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).expect("wide value");
    ok(
        "mustBeGreaterThan",
        vec![Value::Tensor(wide_value.clone()), Value::Num(rounded)],
    );
    err(
        "mustBeLessThanOrEqual",
        vec![Value::Tensor(wide_value.clone()), Value::Num(rounded)],
    );
    err(
        "mustBeInRange",
        vec![
            Value::Tensor(wide_value.clone()),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![0]), vec![1, 1]).expect("wide lower"),
            ),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![adjacent]), vec![1, 1])
                    .expect("wide upper"),
            ),
        ],
    );

    let complex_value = ComplexTensor::new_integer(
        IntegerComplexStorage::new(
            IntegerStorage::U64(vec![wide]),
            IntegerStorage::U64(vec![0]),
        )
        .expect("complex integer storage"),
        vec![1, 1],
    )
    .expect("complex integer tensor");
    ok(
        "mustBeGreaterThan",
        vec![Value::ComplexTensor(complex_value), Value::Num(rounded)],
    );

    let sparse_value = Value::SparseTensor(
        SparseTensor::new_integer(1, 1, vec![0, 1], vec![0], IntegerStorage::U64(vec![wide]))
            .expect("sparse integer value"),
    );
    ok(
        "mustBeGreaterThan",
        vec![sparse_value.clone(), Value::Num(rounded)],
    );
    err(
        "mustBeInRange",
        vec![
            sparse_value,
            Value::SparseTensor(
                SparseTensor::new_integer(1, 1, vec![0, 0], vec![], IntegerStorage::U64(vec![]))
                    .expect("sparse lower"),
            ),
            Value::SparseTensor(
                SparseTensor::new_integer(
                    1,
                    1,
                    vec![0, 1],
                    vec![0],
                    IntegerStorage::U64(vec![adjacent]),
                )
                .expect("sparse upper"),
            ),
        ],
    );
}

#[test]
fn in_range_supports_interval_flags_and_rejects_extra_inputs() {
    ok(
        "mustBeInRange",
        vec![Value::Num(1.0), Value::Num(1.0), Value::Num(2.0)],
    );
    err(
        "mustBeInRange",
        vec![
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(2.0),
            Value::String("exclusive".into()),
        ],
    );
    ok(
        "mustBeInRange",
        vec![
            Value::Num(1.5),
            Value::Num(1.0),
            Value::Num(2.0),
            Value::String("exclusive".into()),
        ],
    );
    err(
        "mustBeInRange",
        vec![
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(2.0),
            Value::String("exclusive".into()),
            Value::String("inclusive".into()),
        ],
    );
    ok(
        "mustBeInRange",
        vec![
            Value::Num(2.0),
            Value::Num(1.0),
            Value::Num(2.0),
            Value::String("exclude-lower".into()),
        ],
    );
    err(
        "mustBeInRange",
        vec![
            Value::Num(2.0),
            Value::Num(1.0),
            Value::Num(2.0),
            Value::String("inclusive".into()),
            Value::String("exclusive".into()),
            Value::String("extra".into()),
        ],
    );
}

#[test]
fn varname_rules_reject_keywords_underscores_and_overlong_names() {
    assert!(isvarname_value(&Value::String("alpha_1".into())));
    assert!(!isvarname_value(&Value::String("_alpha".into())));
    assert!(!isvarname_value(&Value::String("1alpha".into())));
    assert!(!isvarname_value(&Value::String("for".into())));
    assert!(!isvarname_value(&Value::String("end".into())));
    assert!(isvarname_value(&Value::String(
        "a".repeat(MATLAB_NAME_LENGTH_MAX)
    )));
    assert!(!isvarname_value(&Value::String(
        "a".repeat(MATLAB_NAME_LENGTH_MAX + 1)
    )));
}

#[test]
fn callable_validators_reject_unexpected_extra_arguments() {
    err("mustBePositive", vec![Value::Num(1.0), Value::Num(2.0)]);
    err("mustBeMember", vec![Value::String("on".into())]);
    err(
        "mustBeA",
        vec![
            Value::Num(1.0),
            Value::String("double".into()),
            Value::String("extra".into()),
        ],
    );
}

#[test]
fn ordered_validators_support_exact_compatible_integer_bounds() {
    let value = integer_tensor(
        IntegerStorage::U64(vec![9_007_199_254_740_993, 4]),
        vec![2, 1],
    );
    let bounds = integer_tensor(
        IntegerStorage::U64(vec![9_007_199_254_740_992, 3]),
        vec![2, 1],
    );
    ok("mustBeGreaterThan", vec![value.clone(), bounds]);
    err(
        "mustBeLessThanOrEqual",
        vec![value, Value::Num(9_007_199_254_740_992.0)],
    );

    let matrix = integer_tensor(IntegerStorage::I16(vec![1, 2]), vec![2, 1]);
    let row_bounds = integer_tensor(IntegerStorage::I16(vec![0, 0]), vec![1, 2]);
    ok("mustBeGreaterThan", vec![matrix, row_bounds]);

    let huge_sparse = SparseTensor::new_integer(
        1_000_000,
        1_000_000,
        {
            let mut pointers = vec![0; 1_000_001];
            pointers[1..].fill(1);
            pointers
        },
        vec![0],
        IntegerStorage::I64(vec![1]),
    )
    .expect("huge sparse sentinel");
    ok("mustBeNonnegative", vec![Value::SparseTensor(huge_sparse)]);
}

#[test]
fn must_be_in_range_requires_same_class_and_compares_wide_integers_exactly() {
    let value = integer_tensor(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1]);
    let lower = integer_tensor(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1]);
    let upper = integer_tensor(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]);
    ok("mustBeInRange", vec![value.clone(), lower, upper]);
    err(
        "mustBeInRange",
        vec![value, Value::Num(0.0), Value::Num(f64::INFINITY)],
    );
}

#[test]
fn must_be_member_enforces_nondouble_class_compatibility_without_rounding() {
    let wide = integer_tensor(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1]);
    let rounded_double = tensor(vec![9_007_199_254_740_992.0], 1, 1);
    err("mustBeMember", vec![wide.clone(), rounded_double]);
    let exact = integer_tensor(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1]);
    ok("mustBeMember", vec![wide.clone(), exact]);
    let unlike = integer_tensor(IntegerStorage::I64(vec![9]), vec![1, 1]);
    err("mustBeMember", vec![wide, unlike]);
}

#[test]
fn resident_content_validators_download_exactly_and_preserve_source() {
    use futures::executor::block_on;

    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new_integer(IntegerStorage::I64(vec![-1, 2]), vec![2, 1])
            .expect("resident integer");
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
            .expect("upload resident integer");
        let handle = handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic);
        let value = Value::GpuTensor(handle.clone());
        assert!(block_on(dispatch_validator_async(
            "mustBeNonzero",
            vec![value.clone()],
        ))
        .is_ok());
        assert!(block_on(dispatch_validator_async("mustBePositive", vec![value],)).is_err());
        assert!(crate::builtins::common::gpu_helpers::exact_provider_for_handle(&handle).is_some());
        let gathered = block_on(provider.download_integer(&handle)).expect("source survives");
        assert_eq!(
            gathered.data,
            runmat_accelerate_api::HostIntegerDataOwned::I64(vec![-1, 2])
        );
        provider.free(&handle).expect("free resident source");
        runmat_accelerate_api::clear_handle_metadata(&handle);
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_resident_integer_validators_preserve_exact_source_and_provenance() {
    use futures::executor::block_on;
    use runmat_accelerate_api::AccelProvider;

    let _lock = test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("register WGPU provider for integer validator coverage");
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let tensor = Tensor::new_integer(
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        vec![1, 2],
    )
    .expect("wide integer source");
    let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
        .expect("upload wide integer source");
    let handle = handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic);
    let value = Value::GpuTensor(handle.clone());

    for name in [
        "mustBeFinite",
        "mustBeInteger",
        "mustBeNonNan",
        "mustBeNonzero",
        "mustBePositive",
    ] {
        block_on(dispatch_validator_async(name, vec![value.clone()]))
            .unwrap_or_else(|error| panic!("{name} unexpectedly failed: {error}"));
    }
    let handle = handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
    let value = Value::GpuTensor(handle.clone());
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    for name in ["mustBeReal", "mustBeVector"] {
        block_on(dispatch_validator_async(name, vec![value.clone()]))
            .unwrap_or_else(|error| panic!("{name} unexpectedly failed: {error}"));
    }
    block_on(dispatch_validator_async(
        "mustBeUnderlyingType",
        vec![value.clone(), Value::String("uint64".into())],
    ))
    .expect("resident underlying type metadata");
    assert!(block_on(dispatch_validator_async(
        "mustBeScalarOrEmpty",
        vec![value.clone()],
    ))
    .is_err());
    assert!(block_on(dispatch_validator_async(
        "mustBeSparse",
        vec![value.clone()],
    ))
    .is_err());
    for name in ["mustBeText", "mustBeTextScalar", "mustBeValidVariableName"] {
        let error = block_on(dispatch_validator_async(name, vec![value.clone()]))
            .expect_err("resident integer must reject text validation");
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    }
    assert_eq!(
        runmat_accelerate_api::handle_provenance(&handle),
        Some(runmat_accelerate_api::GpuHandleProvenance::Explicit)
    );
    let gathered = block_on(provider.download_integer(&handle)).expect("source survives");
    assert_eq!(
        gathered.data,
        runmat_accelerate_api::HostIntegerDataOwned::U64(vec![9_007_199_254_740_993, u64::MAX,])
    );
    provider.free(&handle).expect("free source");
    runmat_accelerate_api::clear_handle_metadata(&handle);
}

#[test]
fn undocumented_explicit_resident_validators_are_mode_gated() {
    test_support::with_test_provider(|provider| {
        let tensor =
            Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).expect("resident integer");
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
            .expect("upload resident integer");
        let handle = handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let value = Value::GpuTensor(handle.clone());
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        for name in [
            "mustBeFinite",
            "mustBeInteger",
            "mustBeNonNan",
            "mustBeNonzero",
        ] {
            let error = dispatch_validator(name, vec![value.clone()])
                .expect_err("explicit resident extension must reject in compatibility mode");
            assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        }
        drop(_strict);
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        for name in [
            "mustBeFinite",
            "mustBeInteger",
            "mustBeNonNan",
            "mustBeNonzero",
        ] {
            ok(name, vec![value.clone()]);
        }
        provider.free(&handle).expect("free resident source");
        runmat_accelerate_api::clear_handle_metadata(&handle);
    });
}

#[test]
fn validate_function_signatures_json_checks_json_syntax() {
    assert_eq!(
        VALIDATE_FUNCTION_SIGNATURES_JSON_INTEGER_AUDIT.kind,
        BuiltinIntegerAuditKind::NotApplicable
    );
    ok(
        "validateFunctionSignaturesJSON",
        vec![Value::String(r#"{"functions":[]}"#.into())],
    );
    err(
        "validateFunctionSignaturesJSON",
        vec![Value::String("{not json}".into())],
    );
    err(
        "validateFunctionSignaturesJSON",
        vec![Value::Int(IntValue::U64(u64::MAX))],
    );
    let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 1],
        device_id: u32::MAX,
        buffer_id: u64::MAX,
        descriptor: Default::default(),
    });
    let error = dispatch_validator("validateFunctionSignaturesJSON", vec![resident])
        .expect_err("resident numeric input must reject as invalid text");
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
}
