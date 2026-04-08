//! MATLAB-compatible `ifft` builtin with GPU-aware semantics for RunMat.

use super::common::{
    complex_tensor_to_real_value, default_dimension, download_provider_complex_tensor,
    gather_gpu_complex_tensor, parse_length, parse_symflag, transform_complex_tensor,
    value_to_complex_tensor, TransformDirection,
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
use crate::builtins::math::fft::type_resolvers::ifft_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::fft::ifft")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ifft",
    op_kind: GpuOpKind::Custom("ifft"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("ifft_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers should expose `ifft_dim` (or reuse `fft_dim` with inverse scaling); when absent, the runtime gathers to the host and evaluates the inverse FFT in software.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::fft::ifft")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ifft",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Inverse FFT boundaries are not currently fused; fusion plans terminate before invoking `ifft`.",
};

const BUILTIN_NAME: &str = "ifft";

fn ifft_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "ifft",
    category = "math/fft",
    summary = "Inverse discrete Fourier transform with optional length, dimension, and symmetric flag.",
    keywords = "ifft,inverse fft,inverse fourier transform,symmetric,gpu",
    type_resolver(ifft_type),
    builtin_path = "crate::builtins::math::fft::ifft"
)]
async fn ifft_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (length, dimension, symmetric) = parse_arguments(&rest).await?;
    match value {
        Value::GpuTensor(handle) => ifft_gpu(handle, length, dimension, symmetric).await,
        other => ifft_host(other, length, dimension, symmetric),
    }
}

fn ifft_host(
    value: Value,
    length: Option<usize>,
    dimension: Option<usize>,
    symmetric: bool,
) -> BuiltinResult<Value> {
    let tensor = value_to_complex_tensor(value, BUILTIN_NAME)?;
    let transformed = ifft_complex_tensor(tensor, length, dimension)?;
    finalize_ifft_output(transformed, symmetric)
}

async fn ifft_gpu(
    handle: GpuTensorHandle,
    length: Option<usize>,
    dimension: Option<usize>,
    symmetric: bool,
) -> BuiltinResult<Value> {
    let mut logical_shape = normalize_scalar_shape(&handle.shape);
    if logical_shape.last() == Some(&2) {
        logical_shape.pop();
        logical_shape = normalize_scalar_shape(&logical_shape);
    }

    let dim_one_based = match dimension {
        Some(0) => return Err(ifft_error("ifft: dimension must be >= 1")),
        Some(dim) => dim,
        None => default_dimension(&logical_shape),
    };
    let dim_index = dim_one_based - 1;

    while logical_shape.len() <= dim_index {
        logical_shape.push(1);
    }

    let current_len = logical_shape.get(dim_index).copied().unwrap_or(0);
    let target_len = length.unwrap_or(current_len);

    if let Some(provider) = runmat_accelerate_api::provider() {
        if target_len != 0 {
            if let Ok(out) = provider.ifft_dim(&handle, length, dim_index).await {
                let complex = download_provider_complex_tensor(provider, &out, BUILTIN_NAME, true)
                    .await?;
                return finalize_ifft_output(complex, symmetric);
            }
        }

        let complex = download_provider_complex_tensor(provider, &handle, BUILTIN_NAME, false).await?;
        let transformed = ifft_complex_tensor(complex, length, dimension)?;
        return finalize_ifft_output(transformed, symmetric);
    }

    let complex = gather_gpu_complex_tensor(&handle, BUILTIN_NAME).await?;
    let transformed = ifft_complex_tensor(complex, length, dimension)?;
    finalize_ifft_output(transformed, symmetric)
}

pub(super) fn ifft_complex_tensor(
    tensor: ComplexTensor,
    length: Option<usize>,
    dimension: Option<usize>,
) -> BuiltinResult<ComplexTensor> {
    transform_complex_tensor(tensor, length, dimension, TransformDirection::Inverse, BUILTIN_NAME)
}

fn finalize_ifft_output(tensor: ComplexTensor, symmetric: bool) -> BuiltinResult<Value> {
    if symmetric {
        complex_tensor_to_real_value(tensor, BUILTIN_NAME)
    } else {
        Ok(complex_tensor_into_value(tensor))
    }
}

async fn parse_dimension_arg(value: &Value) -> BuiltinResult<usize> {
    tensor::dimension_from_value_async(value, BUILTIN_NAME, false)
        .await
        .map_err(ifft_error)?
        .ok_or_else(|| ifft_error(format!("{BUILTIN_NAME}: dimension must be numeric, got {value:?}")))
}

async fn parse_arguments(args: &[Value]) -> BuiltinResult<(Option<usize>, Option<usize>, bool)> {
    match args.len() {
        0 => Ok((None, None, false)),
        1 => match parse_symflag(&args[0], BUILTIN_NAME)? {
            Some(flag) => Ok((None, None, flag)),
            None => {
                let len = parse_length(&args[0], BUILTIN_NAME)?;
                Ok((len, None, false))
            }
        },
        2 => {
            let first_flag = parse_symflag(&args[0], BUILTIN_NAME)?;
            let second_flag = parse_symflag(&args[1], BUILTIN_NAME)?;
            if let Some(flag) = second_flag {
                if first_flag.is_some() {
                    return Err(ifft_error(
                        "ifft: symmetry flag must appear as the final argument",
                    ));
                }
                let len = parse_length(&args[0], BUILTIN_NAME)?;
                Ok((len, None, flag))
            } else if first_flag.is_some() {
                Err(ifft_error(
                    "ifft: symmetry flag must appear as the final argument",
                ))
            } else {
                let len = parse_length(&args[0], BUILTIN_NAME)?;
                let dim = Some(parse_dimension_arg(&args[1]).await?);
                Ok((len, dim, false))
            }
        }
        3 => {
            let first_flag = parse_symflag(&args[0], BUILTIN_NAME)?;
            let second_flag = parse_symflag(&args[1], BUILTIN_NAME)?;
            let third_flag = parse_symflag(&args[2], BUILTIN_NAME)?;
            let symmetry = third_flag.ok_or_else(|| {
                ifft_error("ifft: expected 'symmetric' or 'nonsymmetric' as the final argument")
            })?;
            if first_flag.is_some() || second_flag.is_some() {
                return Err(ifft_error(
                    "ifft: symmetry flag may only appear once at the end",
                ));
            }
            let len = parse_length(&args[0], BUILTIN_NAME)?;
            let dim = Some(parse_dimension_arg(&args[1]).await?);
            Ok((len, dim, symmetry))
        }
        _ => Err(ifft_error(
            "ifft: expected ifft(X), ifft(X, N), ifft(X, N, DIM), or ifft(X, N, DIM, symflag)",
        )),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use num_complex::Complex;
    use runmat_builtins::{
        ComplexTensor as HostComplexTensor, IntValue, ResolveContext, Tensor, Type,
    };
    use rustfft::FftPlanner;

    fn approx_eq((a_re, a_im): (f64, f64), (b_re, b_im): (f64, f64), tol: f64) -> bool {
        (a_re - b_re).abs() <= tol && (a_im - b_im).abs() <= tol
    }

    fn error_message(error: crate::RuntimeError) -> String {
        error.message().to_string()
    }

    fn value_as_complex_tensor(value: Value) -> HostComplexTensor {
        match value {
            Value::ComplexTensor(t) => t,
            Value::Tensor(t) => {
                HostComplexTensor::new(t.data.into_iter().map(|re| (re, 0.0)).collect(), t.shape)
                    .unwrap()
            }
            Value::Num(n) => HostComplexTensor::new(vec![(n, 0.0)], vec![1, 1]).unwrap(),
            Value::Int(i) => HostComplexTensor::new(vec![(i.to_f64(), 0.0)], vec![1, 1]).unwrap(),
            other => panic!("unexpected value kind {other:?}"),
        }
    }

    #[test]
    fn ifft_type_preserves_shape() {
        let out = ifft_type(
            &[Type::Tensor {
                shape: Some(vec![Some(4), Some(2)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(4), Some(2)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_inverts_known_fft() {
        let spectrum = HostComplexTensor::new(
            vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)],
            vec![4],
        )
        .unwrap();
        let result = ifft_host(Value::ComplexTensor(spectrum), None, None, false).expect("ifft");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![4]);
                let expected = [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)];
                for (idx, actual) in ct.data.iter().enumerate() {
                    assert!(approx_eq(*actual, expected[idx], 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_symmetric_returns_real_tensor() {
        let spectrum = HostComplexTensor::new(
            vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)],
            vec![4],
        )
        .unwrap();
        let result =
            ifft_host(Value::ComplexTensor(spectrum), None, None, true).expect("ifft symmetric");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![4]);
                assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("expected real tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_zero_length_returns_empty_tensor() {
        let spectrum = HostComplexTensor::new(Vec::new(), vec![0]).unwrap();
        let result = ifft_host(Value::ComplexTensor(spectrum), Some(0), None, false)
            .expect("ifft zero length");
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
    fn ifft_dimension_argument_recovers_matrix() {
        let original = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let mut spectrum = Vec::with_capacity(original.data.len());
        let rows = original.shape[0];
        let cols = original.shape[1];
        for c in 0..cols {
            let mut column = Vec::with_capacity(rows);
            for r in 0..rows {
                column.push(Complex::new(original.data[r + c * rows], 0.0));
            }
            let mut fft = column.clone();
            FftPlanner::<f64>::new()
                .plan_fft_forward(rows)
                .process(&mut fft);
            for value in fft {
                spectrum.push((value.re, value.im));
            }
        }
        let freq = HostComplexTensor::new(spectrum, vec![2, 3]).unwrap();
        let result = ifft_host(Value::ComplexTensor(freq), None, Some(1), false).expect("ifft dim");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 3]);
                for (idx, (re, im)) in ct.data.iter().enumerate() {
                    assert!(approx_eq((*re, *im), (original.data[idx], 0.0), 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_rejects_dimension_zero() {
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
    fn ifft_accepts_scalar_tensor_dimension_argument() {
        let dim = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let (len, parsed_dim, symmetric) =
            block_on(parse_arguments(&[Value::Num(4.0), Value::Tensor(dim)]))
                .expect("parse arguments");
        assert_eq!(len, Some(4));
        assert_eq!(parsed_dim, Some(2));
        assert!(!symmetric);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_rejects_unknown_string_option() {
        let err =
            error_message(block_on(parse_arguments(&[Value::from("invalidflag")])).unwrap_err());
        assert!(err.contains("unrecognized option"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_accepts_nonsymmetric_flag() {
        let (len, dim, symmetric) =
            block_on(parse_arguments(&[Value::from("nonsymmetric")])).expect("parse");
        assert!(len.is_none());
        assert!(dim.is_none());
        assert!(!symmetric);

        let spectrum = HostComplexTensor::new(
            vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)],
            vec![4],
        )
        .unwrap();
        let result =
            ifft_host(Value::ComplexTensor(spectrum), None, None, symmetric).expect("ifft");
        match result {
            Value::ComplexTensor(ct) => assert_eq!(ct.shape, vec![4]),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_symflag_requires_final_position() {
        let err = error_message(
            block_on(parse_arguments(&[
                Value::from("nonsymmetric"),
                Value::Num(4.0),
            ]))
            .unwrap_err(),
        );
        assert!(err.contains("symmetry flag must appear as the final argument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_symflag_accepts_whitespace() {
        let (len, dim, symmetric) =
            block_on(parse_arguments(&[Value::from(" symmetric ")])).expect("parse");
        assert!(len.is_none());
        assert!(dim.is_none());
        assert!(symmetric);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_zero_padding_length_argument() {
        let spectrum = HostComplexTensor::new(vec![(4.0, 0.0)], vec![1]).unwrap();
        let result = ifft_host(Value::ComplexTensor(spectrum), Some(4), None, false).expect("ifft");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![4]);
                for &(re, im) in &ct.data {
                    assert!((re - 1.0).abs() < 1e-12);
                    assert!(im.abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_truncates_when_length_is_smaller() {
        let spectrum = HostComplexTensor::new(
            vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)],
            vec![4],
        )
        .unwrap();
        let result = ifft_host(Value::ComplexTensor(spectrum), Some(2), None, false).expect("ifft");
        let mut expected = vec![Complex::new(10.0, 0.0), Complex::new(-2.0, 2.0)];
        FftPlanner::<f64>::new()
            .plan_fft_inverse(2)
            .process(&mut expected);
        for value in &mut expected {
            *value /= 2.0;
        }
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2]);
                for ((re, im), expected) in ct.data.iter().zip(expected.iter()) {
                    assert!(approx_eq((*re, *im), (expected.re, expected.im), 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_empty_length_with_symmetric_flag() {
        let empty = Tensor::new(Vec::new(), vec![0]).unwrap();
        let (len, dim, symmetric) = block_on(parse_arguments(&[
            Value::Tensor(empty),
            Value::from("symmetric"),
        ]))
        .expect("parse");
        assert!(len.is_none());
        assert!(dim.is_none());
        assert!(symmetric);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_gpu_roundtrip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let spectrum = vec![10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0];
            let shape = vec![4, 2];
            let view = runmat_accelerate_api::HostTensorView {
                data: &spectrum,
                shape: &shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu = ifft_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("ifft");
            let cpu_spectrum = HostComplexTensor::new(
                vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)],
                vec![4],
            )
            .unwrap();
            let cpu = ifft_builtin(Value::ComplexTensor(cpu_spectrum), Vec::new()).expect("ifft");
            let gpu_ct = value_as_complex_tensor(gpu);
            let cpu_ct = value_as_complex_tensor(cpu);
            assert_eq!(gpu_ct.shape, cpu_ct.shape);
            for (a, b) in gpu_ct.data.iter().zip(cpu_ct.data.iter()) {
                assert!(approx_eq(*a, *b, 1e-12));
            }
            provider.free(&handle).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ifft_wgpu_matches_cpu() {
        if let Some(provider) = runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider()
            .expect("wgpu provider")
        {
            let spectrum = vec![10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0];
            let shape = vec![4, 2];
            let view = runmat_accelerate_api::HostTensorView {
                data: &spectrum,
                shape: &shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu = ifft_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("gpu ifft");
            let cpu_spectrum = HostComplexTensor::new(
                vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)],
                vec![4],
            )
            .unwrap();
            let cpu =
                ifft_builtin(Value::ComplexTensor(cpu_spectrum), Vec::new()).expect("cpu ifft");
            let gpu_ct = value_as_complex_tensor(gpu);
            let cpu_ct = value_as_complex_tensor(cpu);
            assert_eq!(gpu_ct.shape, cpu_ct.shape);
            for (a, b) in gpu_ct.data.iter().zip(cpu_ct.data.iter()) {
                assert!(approx_eq(*a, *b, 1e-9));
            }
            provider.free(&handle).ok();
        }
    }

    fn ifft_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::ifft_builtin(value, rest))
    }
}
