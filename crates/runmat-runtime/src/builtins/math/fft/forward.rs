//! MATLAB-compatible `fft` builtin with GPU-aware semantics for RunMat.

use super::common::{
    default_dimension, gather_gpu_complex_tensor, parse_length, transform_complex_tensor,
    TransformDirection, value_to_complex_tensor,
};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{ComplexTensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{shape::normalize_scalar_shape, tensor};
use crate::builtins::math::fft::type_resolvers::fft_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::fft::forward")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fft",
    op_kind: GpuOpKind::Custom("fft"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("fft_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers should implement `fft_dim` to transform along an arbitrary dimension; the runtime gathers to host when unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::fft::forward")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fft",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "FFT participates in fusion plans only as a boundary; no fused kernels are generated today.",
};

const BUILTIN_NAME: &str = "fft";

fn fft_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "fft",
    category = "math/fft",
    summary = "Compute the discrete Fourier transform (DFT) of numeric or complex data.",
    keywords = "fft,fourier transform,complex,gpu",
    type_resolver(fft_type),
    builtin_path = "crate::builtins::math::fft::forward"
)]
async fn fft_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (length, dimension) = parse_arguments(&rest).await?;
    match value {
        Value::GpuTensor(handle) => fft_gpu(handle, length, dimension).await,
        other => fft_host(other, length, dimension),
    }
}

fn fft_host(value: Value, length: Option<usize>, dimension: Option<usize>) -> BuiltinResult<Value> {
    let tensor = value_to_complex_tensor(value, BUILTIN_NAME)?;
    let transformed = fft_complex_tensor(tensor, length, dimension)?;
    Ok(complex_tensor_into_value(transformed))
}

async fn fft_gpu(
    handle: GpuTensorHandle,
    length: Option<usize>,
    dimension: Option<usize>,
) -> BuiltinResult<Value> {
    let mut shape = normalize_scalar_shape(&handle.shape);

    let dim_one_based = match dimension {
        Some(0) => return Err(fft_error("fft: dimension must be >= 1")),
        Some(dim) => dim,
        None => default_dimension(&shape),
    };

    let dim_index = dim_one_based - 1;
    while shape.len() <= dim_index {
        shape.push(1);
    }
    let current_len = shape[dim_index];
    let target_len = length.unwrap_or(current_len);

    if target_len == 0 {
        let complex = gather_gpu_complex_tensor(&handle, BUILTIN_NAME).await?;
        let transformed = fft_complex_tensor(complex, length, dimension)?;
        return Ok(complex_tensor_into_value(transformed));
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(out) = provider.fft_dim(&handle, length, dim_index).await {
            return Ok(Value::GpuTensor(out));
        }
    }

    let complex = gather_gpu_complex_tensor(&handle, BUILTIN_NAME).await?;
    let transformed = fft_complex_tensor(complex, length, dimension)?;
    Ok(complex_tensor_into_value(transformed))
}

async fn parse_dimension_arg(value: &Value) -> BuiltinResult<usize> {
    tensor::dimension_from_value_async(value, BUILTIN_NAME, false)
        .await
        .map_err(fft_error)?
        .ok_or_else(|| fft_error(format!("{BUILTIN_NAME}: dimension must be numeric, got {value:?}")))
}

async fn parse_arguments(args: &[Value]) -> BuiltinResult<(Option<usize>, Option<usize>)> {
    match args.len() {
        0 => Ok((None, None)),
        1 => {
            let len = parse_length(&args[0], BUILTIN_NAME)?;
            Ok((len, None))
        }
        2 => {
            let len = parse_length(&args[0], BUILTIN_NAME)?;
            let dim = Some(parse_dimension_arg(&args[1]).await?);
            Ok((len, dim))
        }
        _ => Err(fft_error(
            "fft: expected fft(X), fft(X, N), or fft(X, N, DIM)",
        )),
    }
}

pub(super) fn fft_complex_tensor(
    tensor: ComplexTensor,
    length: Option<usize>,
    dimension: Option<usize>,
) -> BuiltinResult<ComplexTensor> {
    transform_complex_tensor(tensor, length, dimension, TransformDirection::Forward, BUILTIN_NAME)
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::builtins::math::fft::common;
    use super::*;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use num_complex::Complex;
    use runmat_builtins::{
        ComplexTensor as HostComplexTensor, IntValue, ResolveContext, Tensor, Type,
    };
    use rustfft::FftPlanner;

    fn approx_eq(a: (f64, f64), b: (f64, f64), tol: f64) -> bool {
        (a.0 - b.0).abs() <= tol && (a.1 - b.1).abs() <= tol
    }

    fn error_message(error: crate::RuntimeError) -> String {
        error.message().to_string()
    }

    fn value_as_complex_tensor(value: Value) -> HostComplexTensor {
        match value {
            Value::ComplexTensor(tensor) => tensor,
            Value::Complex(re, im) => HostComplexTensor::new(vec![(re, im)], vec![1, 1]).unwrap(),
            Value::GpuTensor(handle) => {
                let provider = runmat_accelerate_api::provider_for_handle(&handle)
                    .or_else(runmat_accelerate_api::provider)
                    .expect("provider for gpu handle");
                let host = block_on(provider.download(&handle)).expect("download gpu fft output");
                common::host_to_complex_tensor(host, BUILTIN_NAME).expect("decode gpu complex")
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    fn fft_builtin_sync(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::fft_builtin(value, rest))
    }

    #[test]
    fn fft_type_preserves_shape() {
        let out = fft_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft_real_vector() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let result = fft_host(Value::Tensor(tensor), None, None).expect("fft");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![4]);
                let expected = [(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)];
                for (idx, val) in ct.data.iter().enumerate() {
                    assert!(
                        approx_eq(*val, expected[idx], 1e-12),
                        "idx {idx} {:?} ~= {:?}",
                        val,
                        expected[idx]
                    );
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft_matrix_default_dimension() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = fft_host(Value::Tensor(tensor), None, None).expect("fft");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 3]);
                let expected = [
                    (5.0, 0.0),
                    (-3.0, 0.0),
                    (7.0, 0.0),
                    (-3.0, 0.0),
                    (9.0, 0.0),
                    (-3.0, 0.0),
                ];
                for (idx, val) in ct.data.iter().enumerate() {
                    assert!(approx_eq(*val, expected[idx], 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft_zero_padding_with_length_argument() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result =
            fft_host(Value::Tensor(tensor), Some(5), None).expect("fft with explicit length");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![5]);
                assert!(approx_eq(ct.data[0], (6.0, 0.0), 1e-12));
                assert_eq!(ct.data.len(), 5);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft_empty_length_argument_defaults_to_input_length() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let baseline =
            fft_builtin_sync(Value::Tensor(tensor.clone()), Vec::new()).expect("baseline fft");
        let empty = Tensor::new(Vec::<f64>::new(), vec![0]).unwrap();
        let result = fft_builtin_sync(
            Value::Tensor(tensor),
            vec![Value::Tensor(empty), Value::Int(IntValue::I32(1))],
        )
        .expect("fft with empty length");
        let base_ct = value_as_complex_tensor(baseline);
        let result_ct = value_as_complex_tensor(result);
        assert_eq!(base_ct.shape, result_ct.shape);
        assert_eq!(base_ct.data.len(), result_ct.data.len());
        for (idx, (a, b)) in base_ct.data.iter().zip(result_ct.data.iter()).enumerate() {
            assert!(
                approx_eq(*a, *b, 1e-12),
                "mismatch at index {idx}: {:?} vs {:?}",
                a,
                b
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft_truncates_when_length_smaller() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let result =
            fft_host(Value::Tensor(tensor), Some(2), None).expect("fft with truncation length");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2]);
                let expected = [(3.0, 0.0), (-1.0, 0.0)];
                for (idx, val) in ct.data.iter().enumerate() {
                    assert!(approx_eq(*val, expected[idx], 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft_zero_length_returns_empty_tensor() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result = fft_host(Value::Tensor(tensor), Some(0), None).expect("fft with zero length");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![0]);
                assert!(ct.data.is_empty());
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft_complex_input_preserves_imaginary_components() {
        let tensor =
            HostComplexTensor::new(vec![(1.0, 1.0), (0.0, -1.0), (2.0, 0.5)], vec![3]).unwrap();
        let result =
            fft_host(Value::ComplexTensor(tensor.clone()), None, None).expect("fft complex");
        let mut expected = tensor
            .data
            .iter()
            .map(|(re, im)| Complex::new(*re, *im))
            .collect::<Vec<_>>();
        FftPlanner::<f64>::new()
            .plan_fft_forward(expected.len())
            .process(&mut expected);
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![3]);
                assert_eq!(ct.data.len(), 3);
                for (idx, val) in ct.data.iter().enumerate() {
                    let exp = expected[idx];
                    assert!(approx_eq(*val, (exp.re, exp.im), 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft_row_vector_dimension_two() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let result = fft_host(Value::Tensor(tensor), None, Some(2)).expect("fft along dimension 2");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 4]);
                let expected = [(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)];
                for (idx, val) in ct.data.iter().enumerate() {
                    assert!(approx_eq(*val, expected[idx], 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft_dimension_extends_rank() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let original = tensor.clone();
        let result =
            fft_host(Value::Tensor(tensor), None, Some(3)).expect("fft with extra dimension");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 4, 1]);
                assert_eq!(ct.data.len(), original.data.len());
                for (idx, (re, im)) in ct.data.iter().enumerate() {
                    assert!(approx_eq((*re, *im), (original.data[idx], 0.0), 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft_dimension_extends_rank_with_padding() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let original = tensor.clone();
        let result = fft_host(Value::Tensor(tensor), Some(4), Some(3))
            .expect("fft with padded third dimension");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 4, 4]);
                let mut expected = Vec::with_capacity(16);
                for _depth in 0..4 {
                    for &value in &original.data {
                        expected.push((value, 0.0));
                    }
                }
                assert_eq!(ct.data.len(), expected.len());
                for (idx, (actual, expected)) in ct.data.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        approx_eq(*actual, *expected, 1e-12),
                        "idx {idx}: {:?} != {:?}",
                        actual,
                        expected
                    );
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft_rejects_non_numeric_length() {
        assert!(block_on(parse_arguments(&[Value::Bool(true)])).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft_rejects_negative_length() {
        let err = error_message(block_on(parse_arguments(&[Value::Num(-1.0)])).unwrap_err());
        assert!(err.contains("length must be non-negative"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft_rejects_fractional_length() {
        let err = error_message(block_on(parse_arguments(&[Value::Num(1.5)])).unwrap_err());
        assert!(err.contains("length must be an integer"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft_rejects_dimension_zero() {
        let err = error_message(
            block_on(parse_arguments(&[
                Value::Num(4.0),
                Value::Int(IntValue::I32(0)),
            ]))
            .unwrap_err(),
        );
        assert!(err.contains("dimension must be >= 1"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft_accepts_scalar_tensor_dimension_argument() {
        let dim = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let (len, parsed_dim) = block_on(parse_arguments(&[Value::Num(4.0), Value::Tensor(dim)]))
            .expect("parse arguments");
        assert_eq!(len, Some(4));
        assert_eq!(parsed_dim, Some(2));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft_gpu_roundtrip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu = fft_builtin_sync(Value::GpuTensor(handle.clone()), Vec::new()).expect("fft");
            let cpu = fft_builtin_sync(Value::Tensor(tensor), Vec::new()).expect("fft");
            let gpu_host = value_as_complex_tensor(gpu);
            let cpu_host = value_as_complex_tensor(cpu);
            assert_eq!(gpu_host.shape, cpu_host.shape);
            for (a, b) in gpu_host.data.iter().zip(cpu_host.data.iter()) {
                assert!(approx_eq(*a, *b, 1e-12));
            }
            provider.free(&handle).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft_gpu_non_power_of_two_length_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu = fft_builtin_sync(
                Value::GpuTensor(handle.clone()),
                vec![Value::Int(IntValue::I32(7))],
            )
            .expect("fft gpu");
            let cpu = fft_builtin_sync(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(7))])
                .expect("fft cpu");
            let gpu_host = value_as_complex_tensor(gpu);
            let cpu_host = value_as_complex_tensor(cpu);
            assert_eq!(gpu_host.shape, cpu_host.shape);
            for (a, b) in gpu_host.data.iter().zip(cpu_host.data.iter()) {
                assert!(approx_eq(*a, *b, 1e-10));
            }
            provider.free(&handle).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn fft_wgpu_matches_cpu() {
        if let Some(provider) = runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider()
            .expect("wgpu provider")
        {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
            let tensor_cpu = tensor.clone();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu =
                fft_builtin_sync(Value::GpuTensor(handle.clone()), Vec::new()).expect("gpu fft");
            let cpu = fft_builtin_sync(Value::Tensor(tensor_cpu), Vec::new()).expect("cpu fft");
            let gpu_ct = value_as_complex_tensor(gpu);
            let cpu_ct = value_as_complex_tensor(cpu);
            assert_eq!(gpu_ct.shape, cpu_ct.shape);
            for (a, b) in gpu_ct.data.iter().zip(cpu_ct.data.iter()) {
                assert!(approx_eq(*a, *b, 1e-9), "{a:?} vs {b:?}");
            }
            provider.free(&handle).ok();
        }
    }
}
