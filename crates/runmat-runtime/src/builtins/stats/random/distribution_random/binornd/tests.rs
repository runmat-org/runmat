use futures::executor::block_on;
use runmat_accelerate_api::{GpuTensorHandle, ProviderPrecision};
use runmat_value::{IntegerStorage, NumericDType, Tensor, Value};

use super::*;
use crate::builtins::common::gpu_helpers;

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
    let output = block_on(binornd_builtin(vec![
        Value::Tensor(Tensor::new(vec![5.0, 10.0], vec![1, 2]).unwrap()),
        Value::Num(0.5),
    ]))
    .expect("scalar-expanded binornd");
    let Value::Tensor(output) = output else {
        panic!("expected tensor output");
    };
    assert_eq!(output.shape, vec![1, 2]);
    assert!((0.0..=5.0).contains(&output.materialize_f64()[0]));
    assert!((0.0..=10.0).contains(&output.materialize_f64()[1]));

    let output = block_on(binornd_builtin(vec![
        Value::Num(10.0),
        Value::Num(0.5),
        Value::Num(2.0),
        Value::Num(4.0),
    ]))
    .expect("explicit-size binornd");
    let Value::Tensor(output) = output else {
        panic!("expected tensor output");
    };
    assert_eq!(output.shape, vec![2, 4]);
}

#[test]
fn validates_domains_and_uses_a_bounded_large_trial_sampler() {
    let _guard = reset();
    for arguments in [
        vec![Value::Num(0.0), Value::Num(0.5)],
        vec![Value::Num(2.5), Value::Num(0.5)],
        vec![Value::Num(2.0), Value::Num(-0.1)],
        vec![Value::Num(2.0), Value::Num(1.1)],
    ] {
        let error = block_on(binornd_builtin(arguments)).expect_err("invalid domain");
        assert_eq!(
            error.identifier(),
            BINORND_ERROR_INVALID_ARGUMENT.identifier
        );
    }

    let output = block_on(binornd_builtin(vec![Value::Num(1.0e12), Value::Num(0.5)]))
        .expect("large trial count");
    let Value::Num(output) = output else {
        panic!("expected scalar output");
    };
    assert!(output.is_finite());
    assert!((0.0..=1.0e12).contains(&output));
}

#[test]
fn every_integer_class_works_in_each_supported_role() {
    let _guard = reset();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    for storage in all_integer_storages(1) {
        assert_eq!(
            block_on(binornd_builtin(vec![
                integer_tensor(storage.clone(), vec![1, 1]),
                Value::Num(1.0),
            ]))
            .expect("integer n"),
            Value::Num(1.0)
        );
        assert_eq!(
            block_on(binornd_builtin(vec![
                Value::Num(1.0),
                integer_tensor(storage.clone(), vec![1, 1]),
            ]))
            .expect("integer p"),
            Value::Num(1.0)
        );
        assert_eq!(
            block_on(binornd_builtin(vec![
                Value::Num(1.0),
                Value::Num(1.0),
                integer_tensor(storage, vec![1, 1]),
            ]))
            .expect("integer size"),
            Value::Num(1.0)
        );
    }
}

#[test]
fn compatibility_gates_each_extension_before_gather() {
    let _guard = reset();
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let cases = [
        (
            vec![
                integer_tensor(IntegerStorage::I8(vec![1]), vec![1, 1]),
                Value::Num(0.5),
            ],
            "RunMat:compatibility:BinorndIntegerTrialsExtension",
        ),
        (
            vec![
                Value::Num(1.0),
                integer_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]),
            ],
            "RunMat:compatibility:BinorndIntegerProbabilityExtension",
        ),
        (
            vec![
                Value::Num(1.0),
                Value::Num(0.5),
                integer_tensor(IntegerStorage::U32(vec![2]), vec![1, 1]),
            ],
            "RunMat:compatibility:BinorndIntegerSizeExtension",
        ),
        (
            vec![Value::Bool(true), Value::Num(0.5)],
            "RunMat:compatibility:BinorndLogicalInputExtension",
        ),
    ];
    for (arguments, identifier) in cases {
        let error = block_on(binornd_builtin(arguments)).expect_err("extension gate");
        assert_eq!(error.identifier(), Some(identifier));
    }

    let resident = GpuTensorHandle {
        shape: vec![1, 1],
        device_id: 0,
        buffer_id: 9_306_001,
        descriptor: Default::default(),
    }
    .with_numeric_descriptor(
        runmat_accelerate_api::NumericElementType::I16,
        runmat_accelerate_api::GpuTensorStorage::Real,
    );
    let error = block_on(binornd_builtin(vec![
        Value::GpuTensor(resident),
        Value::Num(0.5),
    ]))
    .expect_err("resident integer gate before gather");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:BinorndIntegerTrialsExtension")
    );
}

#[test]
fn preserves_single_from_either_parameter_and_exact_integer_boundaries() {
    let _guard = reset();
    for arguments in [
        vec![
            Value::Tensor(Tensor::from_f32(vec![2.0, 2.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
        ],
        vec![
            Value::Num(2.0),
            Value::Tensor(Tensor::from_f32(vec![1.0, 1.0], vec![1, 2]).unwrap()),
        ],
    ] {
        let output = block_on(binornd_builtin(arguments)).expect("single binornd");
        let Value::Tensor(output) = output else {
            panic!("expected single tensor");
        };
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
    }

    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let exact_wide = Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 54]), vec![1, 1])
        .expect("integer tensor");
    ensure_exact_integer_boundary(&exact_wide, "test")
        .expect("exact powers of two above 2^53 remain valid");
    let error = block_on(binornd_builtin(vec![
        integer_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]),
        Value::Num(0.5),
    ]))
    .expect_err("inexact integer parameter");
    assert!(error.message().contains("exactly representable as double"));
}

#[test]
fn applies_size_normalization_and_checks_both_parameter_shapes() {
    let _guard = reset();
    let square = block_on(binornd_builtin(vec![
        Value::Num(2.0),
        Value::Num(1.0),
        Value::Num(3.0),
    ]))
    .expect("scalar size");
    let Value::Tensor(square) = square else {
        panic!("expected square tensor");
    };
    assert_eq!(square.shape, vec![3, 3]);

    let empty = block_on(binornd_builtin(vec![
        Value::Num(2.0),
        Value::Num(0.5),
        Value::Num(-2.0),
        Value::Num(4.0),
    ]))
    .expect("nonpositive dimension");
    let Value::Tensor(empty) = empty else {
        panic!("expected empty tensor");
    };
    assert_eq!(empty.shape, vec![0, 4]);
    assert!(empty.is_empty());

    for arguments in [
        vec![
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
            Value::Num(0.5),
            Value::Num(2.0),
            Value::Num(2.0),
        ],
        vec![
            Value::Num(2.0),
            Value::Tensor(Tensor::new(vec![0.5, 0.5], vec![1, 2]).unwrap()),
            Value::Num(2.0),
            Value::Num(2.0),
        ],
    ] {
        let error = block_on(binornd_builtin(arguments)).expect_err("size mismatch");
        assert_eq!(
            error.identifier(),
            BINORND_ERROR_INVALID_ARGUMENT.identifier
        );
    }
}

#[test]
fn rejects_unsupported_representations_and_excess_outputs() {
    let _guard = reset();
    for value in [
        Value::CharArray(runmat_value::CharArray::new(vec!['2'], 1, 1).unwrap()),
        Value::Complex(2.0, 0.0),
        Value::SparseTensor(runmat_value::SparseTensor::zeros(1, 1)),
    ] {
        let error = block_on(binornd_builtin(vec![value, Value::Num(0.5)]))
            .expect_err("unsupported parameter representation");
        assert_eq!(
            error.identifier(),
            BINORND_ERROR_INVALID_ARGUMENT.identifier
        );
    }
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let error = block_on(binornd_builtin(vec![
        Value::Num(2.0),
        Value::Num(0.5),
        Value::Bool(true),
    ]))
    .expect_err("logical size");
    assert_eq!(
        error.identifier(),
        BINORND_ERROR_INVALID_ARGUMENT.identifier
    );

    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = block_on(binornd_builtin(vec![Value::Num(2.0), Value::Num(0.5)]))
        .expect_err("second output");
    assert_eq!(
        error.identifier(),
        BINORND_ERROR_TOO_MANY_OUTPUTS.identifier
    );
}

#[test]
fn reset_random_state_reproduces_samples() {
    let _guard = reset();
    let arguments = || vec![Value::Num(10.0), Value::Num(0.5), Value::Num(8.0)];
    let first = tensor_data(block_on(binornd_builtin(arguments())).expect("first sample"));
    random::reset_rng();
    let second = tensor_data(block_on(binornd_builtin(arguments())).expect("second sample"));
    assert_eq!(first, second);
}

#[test]
fn provider_fallback_preserves_explicit_residency_and_precision() {
    use crate::builtins::common::test_support;

    let _guard = reset();
    test_support::with_test_provider(|provider| {
        for (parameter, precision) in [
            (
                Tensor::new(vec![0.5, 1.0], vec![1, 2]).unwrap(),
                ProviderPrecision::F64,
            ),
            (
                Tensor::from_f32(vec![0.5, 1.0], vec![1, 2]).unwrap(),
                ProviderPrecision::F32,
            ),
        ] {
            let handle = gpu_helpers::upload_tensor(provider, &parameter)
                .expect("upload")
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let output = block_on(binornd_builtin(vec![
                Value::Num(4.0),
                Value::GpuTensor(handle),
            ]))
            .expect("provider binornd");
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
            Tensor::new(vec![0.5, 1.0], vec![1, 2]).unwrap(),
            ProviderPrecision::F64,
        ),
        (
            Tensor::from_f32(vec![0.5, 1.0], vec![1, 2]).unwrap(),
            ProviderPrecision::F32,
        ),
    ] {
        let handle = gpu_helpers::upload_tensor(provider, &parameter)
            .expect("upload")
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let output = block_on(binornd_builtin(vec![
            Value::Num(4.0),
            Value::GpuTensor(handle),
        ]))
        .expect("WGPU binornd");
        let Value::GpuTensor(output) = output else {
            panic!("expected resident output");
        };
        assert!(runmat_accelerate_api::handle_is_explicit(&output));
        assert_eq!(
            output.device_id,
            runmat_accelerate_api::AccelProvider::device_id(provider)
        );
        assert_eq!(output.shape, vec![1, 2]);
        assert_eq!(
            runmat_accelerate_api::handle_precision(&output),
            Some(precision)
        );
    }
}
