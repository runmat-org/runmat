use super::*;
use futures::executor::block_on;
use runmat_value::{ComplexTensor as HostComplexTensor, IntValue, LogicalArray, NumericDType};

#[path = "tests/provider_route.rs"]
mod provider_route;

const TOL: f64 = 1.0e-12;

fn hilbert_call(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(hilbert_builtin(
        None,
        value,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}

fn as_complex_tensor(value: Value) -> HostComplexTensor {
    match value {
        Value::ComplexTensor(tensor) => tensor,
        Value::Complex(re, im) => HostComplexTensor::new(vec![(re, im)], vec![1, 1]).unwrap(),
        Value::GpuTensor(handle) => block_on(
            crate::builtins::math::fft::common::gather_gpu_complex_tensor(&handle, BUILTIN_NAME),
        )
        .expect("gather complex gpu output"),
        other => panic!("expected complex output, got {other:?}"),
    }
}

fn assert_complex_close(actual: (f64, f64), expected: (f64, f64)) {
    assert!(
        (actual.0 - expected.0).abs() <= TOL,
        "real mismatch: actual={} expected={}",
        actual.0,
        expected.0
    );
    assert!(
        (actual.1 - expected.1).abs() <= TOL,
        "imag mismatch: actual={} expected={}",
        actual.1,
        expected.1
    );
}

#[test]
fn hilbert_row_cosine_returns_quadrature_signal() {
    let input = Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![1, 4]).unwrap();
    let out = as_complex_tensor(hilbert_call(Value::Tensor(input), Vec::new()).unwrap());
    assert_eq!(out.shape(), &[1, 4]);
    let expected = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
    for (actual, expected) in out.materialize_f64().iter().copied().zip(expected) {
        assert_complex_close(actual, expected);
    }
}

#[test]
fn hilbert_single_real_input_preserves_single_complex_class() {
    let real =
        Tensor::from_numeric_storage(NumericStorage::F32(vec![1.0, 0.0, -1.0, 0.0]), vec![1, 4])
            .unwrap();
    let output = as_complex_tensor(hilbert_call(Value::Tensor(real), Vec::new()).unwrap());
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    for (actual, expected) in output.materialize_f64().iter().copied().zip([
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, -1.0),
    ]) {
        assert!((actual.0 - expected.0).abs() < 1.0e-5);
        assert!((actual.1 - expected.1).abs() < 1.0e-5);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn hilbert_low_rank_real_inputs_follow_scalar_and_vector_output_policies() {
    let scalar = Tensor::new(vec![1.0], Vec::new()).expect("rank-zero signal");
    let Value::Complex(real, imaginary) =
        hilbert_call(Value::Tensor(scalar), Vec::new()).expect("scalar Hilbert signal")
    else {
        panic!("double scalar analytic signal must scalarize");
    };
    assert_complex_close((real, imaginary), (1.0, 0.0));

    let vector = Tensor::new(vec![1.0, 0.0], vec![2]).expect("rank-one signal");
    let Value::ComplexTensor(output) =
        hilbert_call(Value::Tensor(vector), Vec::new()).expect("vector Hilbert signal")
    else {
        panic!("rank-one analytic signal must remain complex tensor");
    };
    assert_eq!(output.shape(), &[2]);
    for (actual, expected) in output
        .materialize_f64()
        .iter()
        .copied()
        .zip([(1.0, 0.0), (0.0, 0.0)])
    {
        assert_complex_close(actual, expected);
    }
}

#[test]
fn hilbert_column_cosine_operates_down_columns() {
    let input = Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![4, 1]).unwrap();
    let out = as_complex_tensor(hilbert_call(Value::Tensor(input), Vec::new()).unwrap());
    assert_eq!(out.shape(), &[4, 1]);
    let expected = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
    for (actual, expected) in out.materialize_f64().iter().copied().zip(expected) {
        assert_complex_close(actual, expected);
    }
}

#[test]
fn hilbert_matrix_operates_along_first_nonsingleton_dimension() {
    let input = Tensor::new(vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0], vec![4, 2]).unwrap();
    let out = as_complex_tensor(hilbert_call(Value::Tensor(input), Vec::new()).unwrap());
    assert_eq!(out.shape(), &[4, 2]);
    let expected = [
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, -1.0),
        (0.0, -1.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
    ];
    for (actual, expected) in out.materialize_f64().iter().copied().zip(expected) {
        assert_complex_close(actual, expected);
    }
}

#[test]
fn hilbert_length_argument_pads_or_truncates_transform_axis() {
    let input = Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![1, 4]).unwrap();
    let out = as_complex_tensor(hilbert_call(Value::Tensor(input), vec![Value::Num(6.0)]).unwrap());
    assert_eq!(out.shape(), &[1, 6]);
    assert_eq!(out.materialize_f64().len(), 6);
    assert_complex_close(out.materialize_f64()[0], (1.0, 0.0));
}

#[test]
fn hilbert_zero_length_returns_empty_along_transform_axis() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let input = Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![1, 4]).unwrap();
    let out = as_complex_tensor(
        hilbert_call(Value::Tensor(input), vec![Value::Int(IntValue::I32(0))]).unwrap(),
    );
    assert_eq!(out.shape(), &[1, 0]);
    assert!(out.materialize_f64().is_empty());
}

#[test]
fn hilbert_accepts_logical_input_as_real_signal() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let input = LogicalArray::new(vec![1, 0, 1, 0], vec![1, 4]).unwrap();
    let out = as_complex_tensor(hilbert_call(Value::LogicalArray(input), Vec::new()).unwrap());
    assert_eq!(out.shape(), &[1, 4]);
}

#[test]
fn hilbert_ignores_complex_input_imaginary_part() {
    let input = HostComplexTensor::new(
        vec![(1.0, 9.0), (0.0, -4.0), (-1.0, 2.0), (0.0, 7.0)],
        vec![1, 4],
    )
    .unwrap();
    let output = as_complex_tensor(hilbert_call(Value::ComplexTensor(input), Vec::new()).unwrap());
    let expected = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
    for (actual, expected) in output.materialize_f64().iter().copied().zip(expected) {
        assert_complex_close(actual, expected);
    }
}

#[test]
fn hilbert_runmat_extensions_follow_compatibility_mode() {
    let input = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
    let cases = [
        (
            Value::Int(IntValue::I32(2)),
            "RunMat:compatibility:HilbertTypedIntegerLengthExtension",
        ),
        (
            Value::Bool(true),
            "RunMat:compatibility:HilbertLogicalLengthExtension",
        ),
        (
            Value::Num(0.0),
            "RunMat:compatibility:HilbertZeroLengthExtension",
        ),
    ];
    for (length, identifier) in cases {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = hilbert_call(Value::Tensor(input.clone()), vec![length])
            .expect_err("strict mode rejects extension");
        assert_eq!(err.identifier(), Some(identifier));
    }
}

#[test]
fn hilbert_integer_logical_and_nd_forms_are_extension_gated() {
    let integer = Value::Int(IntValue::I32(1));
    let logical = Value::Bool(true);
    let nd = Value::Tensor(Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![1, 4, 1]).unwrap());
    for (input, identifier) in [
        (integer, "RunMat:compatibility:HilbertIntegerDataExtension"),
        (logical, "RunMat:compatibility:HilbertLogicalDataExtension"),
        (nd, "RunMat:compatibility:HilbertNdInputExtension"),
    ] {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = hilbert_call(input, Vec::new()).expect_err("strict mode rejects extension");
        assert_eq!(error.identifier(), Some(identifier));
    }
}

#[test]
fn hilbert_rejects_fractional_length() {
    let input = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
    let err = hilbert_call(Value::Tensor(input), vec![Value::Num(1.5)]).unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:hilbert:InvalidLength"));
}

#[test]
fn hilbert_rejects_huge_length_before_allocation() {
    let input = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
    let err = hilbert_call(Value::Tensor(input), vec![Value::Num(f64::MAX)]).unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:hilbert:InvalidLength"));
}

#[test]
fn hilbert_gpu_input_returns_owner_resident_analytic_signal() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![1, 4]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &input.materialize_f64(),
            shape: input.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let metadata = hilbert_input_metadata(&handle);
        let out =
            as_complex_tensor(hilbert_call(Value::GpuTensor(handle.clone()), Vec::new()).unwrap());
        assert!(runmat_accelerate_api::provider_for_handle(&handle).is_some());
        assert_eq!(hilbert_input_metadata(&handle), metadata);
        assert_eq!(out.shape(), &[1, 4]);
        let expected = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
        for (actual, expected) in out.materialize_f64().iter().copied().zip(expected) {
            assert_complex_close(actual, expected);
        }
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn hilbert_wgpu_input_returns_resident_complex_gpu_tensor() {
    use crate::builtins::common::test_support;
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    use runmat_accelerate_api::AccelProvider;

    let _guard = test_support::accel_test_lock();
    let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
        return;
    };
    let input = Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![1, 4]).unwrap();
    let view = runmat_accelerate_api::HostTensorView {
        data: &input.materialize_f64(),
        shape: input.shape(),
    };
    let handle = provider.upload(&view).expect("upload");
    let out = hilbert_call(Value::GpuTensor(handle.clone()), Vec::new()).unwrap();
    let Value::GpuTensor(out_handle) = out else {
        panic!("expected resident GPU output");
    };
    assert_eq!(
        runmat_accelerate_api::handle_storage(&out_handle),
        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    );

    let gathered = block_on(
        crate::builtins::math::fft::common::gather_gpu_complex_tensor(&out_handle, BUILTIN_NAME),
    )
    .expect("gather complex output");
    assert_eq!(gathered.shape(), &[1, 4]);
    let expected = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
    for (actual, expected) in gathered.materialize_f64().iter().copied().zip(expected) {
        assert!((actual.0 - expected.0).abs() <= 1.0e-5);
        assert!((actual.1 - expected.1).abs() <= 1.0e-5);
    }
    handle.release().ok();
    out_handle.release().ok();
}

#[cfg(feature = "wgpu")]
#[test]
fn hilbert_wgpu_length_argument_matches_cpu_and_stays_resident() {
    use crate::builtins::common::test_support;
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    use runmat_accelerate_api::AccelProvider;

    let _guard = test_support::accel_test_lock();
    let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
        return;
    };
    let input = Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![1, 4]).unwrap();
    let expected = as_complex_tensor(
        hilbert_call(Value::Tensor(input.clone()), vec![Value::Num(6.0)]).unwrap(),
    );
    let view = runmat_accelerate_api::HostTensorView {
        data: &input.materialize_f64(),
        shape: input.shape(),
    };
    let handle = provider.upload(&view).expect("upload");
    let out = hilbert_call(Value::GpuTensor(handle.clone()), vec![Value::Num(6.0)]).unwrap();
    let Value::GpuTensor(out_handle) = out else {
        panic!("expected resident GPU output");
    };
    assert_eq!(
        runmat_accelerate_api::handle_storage(&out_handle),
        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    );

    let gathered = block_on(
        crate::builtins::math::fft::common::gather_gpu_complex_tensor(&out_handle, BUILTIN_NAME),
    )
    .expect("gather complex output");
    assert_eq!(gathered.shape(), expected.shape());
    for (idx, (actual, expected)) in gathered
        .materialize_f64()
        .iter()
        .copied()
        .zip(expected.materialize_f64())
        .enumerate()
    {
        assert!(
            (actual.0 - expected.0).abs() <= 1.0e-5,
            "real mismatch at {idx}: actual={} expected={}",
            actual.0,
            expected.0
        );
        assert!(
            (actual.1 - expected.1).abs() <= 1.0e-5,
            "imag mismatch at {idx}: actual={} expected={}",
            actual.1,
            expected.1
        );
    }
    handle.release().ok();
    out_handle.release().ok();
}
