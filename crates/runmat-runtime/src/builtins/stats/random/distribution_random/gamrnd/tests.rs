use futures::executor::block_on;
use runmat_accelerate_api::ProviderPrecision;
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

fn tensor_data(value: Value) -> Vec<f64> {
    match value {
        Value::Num(value) => vec![value],
        Value::Tensor(tensor) => tensor.materialize_f64(),
        other => panic!("expected host numeric output, got {other:?}"),
    }
}

#[test]
fn accepts_broadcast_and_explicit_size_forms() {
    let _guard = reset();
    let output = block_on(gamrnd_builtin(vec![
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()),
        Value::Num(2.0),
    ]))
    .expect("broadcast gamrnd");
    let Value::Tensor(output) = output else {
        panic!("expected tensor output");
    };
    assert_eq!(output.shape, vec![1, 3]);
    assert!(output.materialize_f64().iter().all(|value| *value >= 0.0));

    let output = block_on(gamrnd_builtin(vec![
        Value::Num(2.0),
        Value::Num(3.0),
        Value::Num(2.0),
        Value::Num(4.0),
    ]))
    .expect("explicit-size gamrnd");
    let Value::Tensor(output) = output else {
        panic!("expected tensor output");
    };
    assert_eq!(output.shape, vec![2, 4]);
}

#[test]
fn preserves_single_from_either_parameter_and_with_integer_extensions() {
    let _guard = reset();
    for arguments in [
        vec![
            Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::Num(2.0),
        ],
        vec![
            Value::Num(2.0),
            Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        ],
    ] {
        let output = block_on(gamrnd_builtin(arguments)).expect("single gamrnd");
        let Value::Tensor(output) = output else {
            panic!("expected native-single tensor");
        };
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
        assert_eq!(output.shape, vec![1, 2]);
    }

    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let output = block_on(gamrnd_builtin(vec![
        integer_tensor(IntegerStorage::U16(vec![2, 3]), vec![1, 2]),
        Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap()),
    ]))
    .expect("integer and single gamrnd");
    let Value::Tensor(output) = output else {
        panic!("expected native-single tensor");
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
}

#[test]
fn integer_roles_are_independently_compatibility_gated() {
    let _guard = reset();
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let cases = [
        (
            vec![
                integer_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]),
                Value::Num(3.0),
            ],
            "RunMat:compatibility:GamrndIntegerShapeParameterExtension",
        ),
        (
            vec![
                Value::Num(2.0),
                integer_tensor(IntegerStorage::U16(vec![3]), vec![1, 1]),
            ],
            "RunMat:compatibility:GamrndIntegerScaleParameterExtension",
        ),
        (
            vec![
                Value::Num(2.0),
                Value::Num(3.0),
                integer_tensor(IntegerStorage::U16(vec![2, 3]), vec![1, 2]),
            ],
            "RunMat:compatibility:GamrndIntegerSizeExtension",
        ),
    ];
    for (arguments, identifier) in cases {
        let error = block_on(gamrnd_builtin(arguments)).expect_err("integer role gate");
        assert_eq!(error.identifier(), Some(identifier));
    }
}

#[test]
fn integer_sizes_are_exact_and_sampling_boundaries_reject_loss() {
    let _guard = reset();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let output = block_on(gamrnd_builtin(vec![
        integer_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]),
        integer_tensor(IntegerStorage::U16(vec![3]), vec![1, 1]),
        integer_tensor(IntegerStorage::U64(vec![2, 3]), vec![1, 2]),
    ]))
    .expect("checked integer gamrnd");
    let Value::Tensor(output) = output else {
        panic!("expected tensor output");
    };
    assert_eq!(output.shape, vec![2, 3]);

    let error = block_on(gamrnd_builtin(vec![
        integer_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]),
        Value::Num(1.0),
    ]))
    .expect_err("inexact integer parameter");
    assert!(error.message().contains("exactly representable as double"));

    let exact_wide = Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 54]), vec![1, 1])
        .expect("integer tensor");
    ensure_exact_integer_boundary(&exact_wide, "test")
        .expect("exact powers of two above 2^53 remain valid");

    let error = block_on(gamrnd_builtin(vec![
        Value::Num(2.0),
        Value::Num(1.0),
        integer_tensor(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]),
        Value::Num(2.0),
    ]))
    .expect_err("oversized dimensions");
    assert!(error.message().contains("supported array bounds"));
}

#[test]
fn applies_documented_size_normalization_and_shape_constraints() {
    let _guard = reset();
    let square = block_on(gamrnd_builtin(vec![
        Value::Num(2.0),
        Value::Num(1.0),
        Value::Num(3.0),
    ]))
    .expect("scalar size");
    let Value::Tensor(square) = square else {
        panic!("expected square tensor");
    };
    assert_eq!(square.shape, vec![3, 3]);

    let empty = block_on(gamrnd_builtin(vec![
        Value::Num(2.0),
        Value::Num(1.0),
        Value::Num(-2.0),
        Value::Num(4.0),
    ]))
    .expect("nonpositive dimension");
    let Value::Tensor(empty) = empty else {
        panic!("expected empty tensor");
    };
    assert_eq!(empty.shape, vec![0, 4]);
    assert!(empty.is_empty());

    let trailing = block_on(gamrnd_builtin(vec![
        Value::Num(2.0),
        Value::Num(1.0),
        Value::Tensor(Tensor::new(vec![3.0, 1.0, 1.0], vec![1, 3]).unwrap()),
    ]))
    .expect("trailing singleton dimensions");
    let Value::Tensor(trailing) = trailing else {
        panic!("expected tensor output");
    };
    assert_eq!(trailing.shape, vec![3, 1]);

    let column_size = Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap());
    let error = block_on(gamrnd_builtin(vec![
        Value::Num(2.0),
        Value::Num(1.0),
        column_size,
    ]))
    .expect_err("column size vector");
    assert_eq!(
        error.identifier(),
        runmat_builtins::GAMRND_ERROR_INVALID_ARGUMENT.identifier
    );
}

#[test]
fn either_nonscalar_parameter_must_match_an_explicit_size() {
    let _guard = reset();
    for arguments in [
        vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
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
        let error = block_on(gamrnd_builtin(arguments)).expect_err("mismatched explicit size");
        assert_eq!(
            error.identifier(),
            runmat_builtins::GAMRND_ERROR_INVALID_ARGUMENT.identifier
        );
    }
}

#[test]
fn rejects_parameter_domains_representations_and_excess_outputs() {
    let _guard = reset();
    for arguments in [
        vec![Value::Num(-1.0), Value::Num(1.0)],
        vec![Value::Num(1.0), Value::Num(0.0)],
        vec![Value::Num(f64::NAN), Value::Num(1.0)],
    ] {
        let error = block_on(gamrnd_builtin(arguments)).expect_err("invalid parameter domain");
        assert_eq!(
            error.identifier(),
            runmat_builtins::GAMRND_ERROR_INVALID_ARGUMENT.identifier
        );
    }
    for value in [
        Value::Bool(true),
        Value::CharArray(runmat_value::CharArray::new(vec!['2'], 1, 1).unwrap()),
        Value::Complex(2.0, 0.0),
        Value::SparseTensor(runmat_value::SparseTensor::zeros(1, 1)),
    ] {
        let error = block_on(gamrnd_builtin(vec![value, Value::Num(1.0)]))
            .expect_err("unsupported parameter representation");
        assert_eq!(
            error.identifier(),
            runmat_builtins::GAMRND_ERROR_INVALID_ARGUMENT.identifier
        );
    }

    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = block_on(gamrnd_builtin(vec![Value::Num(2.0), Value::Num(1.0)]))
        .expect_err("second output");
    assert_eq!(
        error.identifier(),
        runmat_builtins::GAMRND_ERROR_TOO_MANY_OUTPUTS.identifier
    );
}

#[test]
fn reset_random_state_reproduces_samples() {
    let _guard = reset();
    let arguments = || vec![Value::Num(2.0), Value::Num(3.0), Value::Num(2.0)];
    let first = tensor_data(block_on(gamrnd_builtin(arguments())).expect("first sample"));
    random::reset_rng();
    let second = tensor_data(block_on(gamrnd_builtin(arguments())).expect("second sample"));
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
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let output = block_on(gamrnd_builtin(vec![
                Value::GpuTensor(handle),
                Value::Num(1.0),
            ]))
            .expect("documented floating gpuArray form");
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
        let output = block_on(gamrnd_builtin(vec![
            Value::GpuTensor(handle),
            Value::Num(1.0),
        ]))
        .expect("WGPU gamrnd");
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
