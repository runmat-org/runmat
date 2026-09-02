use futures::executor::block_on;
use runmat_accelerate_api::ProviderPrecision;
use runmat_value::{IntegerStorage, LogicalArray, NumericDType, Tensor, Value};

use super::*;
use crate::builtins::common::{gpu_helpers, random};

fn reset() -> impl Drop {
    let guard = random::test_guard();
    runmat_accelerate_api::clear_provider();
    random::reset_rng();
    guard
}

fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape).expect("integer tensor"))
}

fn all_integer_storages(value: u64) -> [IntegerStorage; 8] {
    [
        IntegerStorage::I8(vec![value as i8]),
        IntegerStorage::I16(vec![value as i16]),
        IntegerStorage::I32(vec![value as i32]),
        IntegerStorage::I64(vec![value as i64]),
        IntegerStorage::U8(vec![value as u8]),
        IntegerStorage::U16(vec![value as u16]),
        IntegerStorage::U32(vec![value as u32]),
        IntegerStorage::U64(vec![value]),
    ]
}

fn tensor_data(value: Value) -> Vec<f64> {
    match value {
        Value::Num(value) => vec![value],
        Value::Tensor(tensor) => tensor.materialize_f64(),
        other => panic!("expected host numeric output, got {other:?}"),
    }
}

#[test]
fn accepts_scalar_expansion_and_explicit_size_forms() {
    let _guard = reset();
    let output = block_on(wblrnd_builtin(vec![
        Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
        Value::Num(1.0),
    ]))
    .expect("scalar-expanded wblrnd");
    let Value::Tensor(output) = output else {
        panic!("expected tensor output");
    };
    assert_eq!(output.shape, vec![1, 2]);
    assert!(output.materialize_f64().iter().all(|value| *value > 0.0));

    let output = block_on(wblrnd_builtin(vec![
        Value::Num(2.0),
        Value::Num(3.0),
        Value::Num(2.0),
        Value::Num(4.0),
    ]))
    .expect("explicit-size wblrnd");
    let Value::Tensor(output) = output else {
        panic!("expected tensor output");
    };
    assert_eq!(output.shape, vec![2, 4]);
}

#[test]
fn validates_domains_representations_and_output_count() {
    let _guard = reset();
    for arguments in [
        vec![Value::Num(0.0), Value::Num(1.0)],
        vec![Value::Num(1.0), Value::Num(-1.0)],
        vec![Value::Num(f64::NAN), Value::Num(1.0)],
    ] {
        let error = block_on(wblrnd_builtin(arguments)).expect_err("invalid parameter domain");
        assert_eq!(error.identifier(), WBLRND_ERROR_INVALID_ARGUMENT.identifier);
    }
    for value in [
        Value::CharArray(runmat_value::CharArray::new(vec!['2'], 1, 1).unwrap()),
        Value::Complex(2.0, 0.0),
        Value::SparseTensor(runmat_value::SparseTensor::zeros(1, 1)),
    ] {
        let error = block_on(wblrnd_builtin(vec![value, Value::Num(1.0)]))
            .expect_err("unsupported representation");
        assert_eq!(error.identifier(), WBLRND_ERROR_INVALID_ARGUMENT.identifier);
    }

    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = block_on(wblrnd_builtin(vec![Value::Num(2.0), Value::Num(1.0)]))
        .expect_err("second output");
    assert_eq!(error.identifier(), WBLRND_ERROR_TOO_MANY_OUTPUTS.identifier);
}

#[test]
fn compatibility_gates_integer_and_logical_roles_before_gather() {
    let _guard = reset();
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let cases = [
        (
            vec![
                integer_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]),
                Value::Num(3.0),
            ],
            "RunMat:compatibility:WblrndIntegerScaleExtension",
        ),
        (
            vec![
                Value::Num(2.0),
                integer_tensor(IntegerStorage::U16(vec![3]), vec![1, 1]),
            ],
            "RunMat:compatibility:WblrndIntegerShapeExtension",
        ),
        (
            vec![
                Value::Num(2.0),
                Value::Num(3.0),
                integer_tensor(IntegerStorage::U16(vec![2, 3]), vec![1, 2]),
            ],
            "RunMat:compatibility:WblrndIntegerSizeExtension",
        ),
        (
            vec![Value::Bool(true), Value::Num(3.0)],
            "RunMat:compatibility:WblrndLogicalInputExtension",
        ),
    ];
    for (arguments, identifier) in cases {
        let error = block_on(wblrnd_builtin(arguments)).expect_err("extension gate");
        assert_eq!(error.identifier(), Some(identifier));
    }
}

#[test]
fn runmat_extensions_preserve_exact_values_and_single_selection() {
    let _guard = reset();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let output = block_on(wblrnd_builtin(vec![
        integer_tensor(IntegerStorage::U16(vec![2, 3]), vec![1, 2]),
        Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap()),
    ]))
    .expect("integer and single wblrnd");
    let Value::Tensor(output) = output else {
        panic!("expected single tensor");
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(output.shape, vec![1, 2]);

    let logical = block_on(wblrnd_builtin(vec![
        Value::LogicalArray(LogicalArray::new(vec![1, 1], vec![1, 2]).unwrap()),
        Value::Bool(true),
    ]))
    .expect("logical parameters");
    let Value::Tensor(logical) = logical else {
        panic!("expected logical-extension tensor output");
    };
    assert_eq!(logical.shape, vec![1, 2]);
    assert_eq!(logical.numeric_dtype(), NumericDType::F64);

    let exact = block_on(wblrnd_builtin(vec![
        integer_tensor(IntegerStorage::U64(vec![1_u64 << 54]), vec![1, 1]),
        Value::Num(1.0),
    ]))
    .expect("exact wide integer parameter");
    assert!(tensor_data(exact)[0].is_finite());

    let error = block_on(wblrnd_builtin(vec![
        integer_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]),
        Value::Num(1.0),
    ]))
    .expect_err("inexact integer parameter");
    assert!(error.message().contains("exactly representable as double"));
}

#[test]
fn every_integer_class_works_in_each_supported_role() {
    let _guard = reset();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    for storage in all_integer_storages(1) {
        assert!(tensor_data(
            block_on(wblrnd_builtin(vec![
                integer_tensor(storage.clone(), vec![1, 1]),
                Value::Num(1.0),
            ]))
            .expect("integer scale")
        )[0]
        .is_finite());
        assert!(tensor_data(
            block_on(wblrnd_builtin(vec![
                Value::Num(1.0),
                integer_tensor(storage.clone(), vec![1, 1]),
            ]))
            .expect("integer shape")
        )[0]
        .is_finite());
        assert!(tensor_data(
            block_on(wblrnd_builtin(vec![
                Value::Num(1.0),
                Value::Num(1.0),
                integer_tensor(storage, vec![1, 1]),
            ]))
            .expect("integer size")
        )[0]
        .is_finite());
    }

    let logical_size = block_on(wblrnd_builtin(vec![
        Value::Bool(true),
        Value::Bool(true),
        Value::Bool(true),
    ]))
    .expect("logical size");
    assert_eq!(tensor_data(logical_size).len(), 1);
}

#[test]
fn normalizes_sizes_and_checks_each_nonscalar_parameter() {
    let _guard = reset();
    let square = block_on(wblrnd_builtin(vec![
        Value::Num(2.0),
        Value::Num(1.0),
        Value::Num(3.0),
    ]))
    .expect("scalar size");
    let Value::Tensor(square) = square else {
        panic!("expected square tensor");
    };
    assert_eq!(square.shape, vec![3, 3]);

    let empty = block_on(wblrnd_builtin(vec![
        Value::Num(2.0),
        Value::Num(1.0),
        Value::Num(-2.0),
        Value::Num(4.0),
    ]))
    .expect("empty shape");
    let Value::Tensor(empty) = empty else {
        panic!("expected empty tensor");
    };
    assert_eq!(empty.shape, vec![0, 4]);

    for arguments in [
        vec![
            Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        ],
        vec![
            Value::Num(2.0),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::Num(2.0),
            Value::Num(2.0),
        ],
    ] {
        let error = block_on(wblrnd_builtin(arguments)).expect_err("mismatched explicit size");
        assert_eq!(error.identifier(), WBLRND_ERROR_INVALID_ARGUMENT.identifier);
    }
}

#[test]
fn random_state_reset_reproduces_samples() {
    let _guard = reset();
    let arguments = || vec![Value::Num(2.0), Value::Num(3.0), Value::Num(2.0)];
    let first = tensor_data(block_on(wblrnd_builtin(arguments())).expect("first sample"));
    random::reset_rng();
    let second = tensor_data(block_on(wblrnd_builtin(arguments())).expect("second sample"));
    assert_eq!(first, second);
}

#[test]
fn documented_gpu_inputs_restore_residency_and_precision() {
    use crate::builtins::common::test_support;

    let _guard = reset();
    test_support::with_test_provider(|provider| {
        for (parameter, precision) in [
            (
                Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap(),
                ProviderPrecision::F64,
            ),
            (
                Tensor::from_f32(vec![2.0, 3.0], vec![1, 2]).unwrap(),
                ProviderPrecision::F32,
            ),
        ] {
            let handle = gpu_helpers::upload_tensor(provider, &parameter)
                .expect("upload")
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let output = block_on(wblrnd_builtin(vec![
                Value::GpuTensor(handle),
                Value::Num(1.0),
            ]))
            .expect("documented resident form");
            let Value::GpuTensor(output) = output else {
                panic!("expected resident output");
            };
            assert!(runmat_accelerate_api::handle_is_explicit(&output));
            assert_eq!(
                runmat_accelerate_api::handle_precision(&output),
                Some(precision)
            );
        }
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn wgpu_fallback_preserves_explicit_residency_and_precision() {
    use crate::builtins::common::test_support;

    let _accel_guard = test_support::accel_test_lock();
    let _guard = reset();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("actual WGPU provider");
    for (parameter, precision) in [
        (
            Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap(),
            ProviderPrecision::F64,
        ),
        (
            Tensor::from_f32(vec![2.0, 3.0], vec![1, 2]).unwrap(),
            ProviderPrecision::F32,
        ),
    ] {
        let handle = gpu_helpers::upload_tensor(provider, &parameter)
            .expect("upload")
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let output = block_on(wblrnd_builtin(vec![
            Value::GpuTensor(handle),
            Value::Num(1.0),
        ]))
        .expect("WGPU wblrnd");
        let Value::GpuTensor(output) = output else {
            panic!("expected resident output");
        };
        assert!(runmat_accelerate_api::handle_is_explicit(&output));
        assert_eq!(output.shape, vec![1, 2]);
        assert_eq!(
            runmat_accelerate_api::handle_precision(&output),
            Some(precision)
        );
    }
}
