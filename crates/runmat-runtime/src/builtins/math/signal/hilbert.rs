//! MATLAB-compatible `hilbert` builtin for analytic signal construction.

use std::mem::size_of;

use num_complex::Complex;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::fft::common::{
    default_dimension, parse_length, tensor_to_complex_tensor, transform_complex_tensor,
    TransformDirection,
};
use crate::builtins::math::fft::type_resolvers::fft_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "hilbert";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::hilbert")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Custom("analytic-signal"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Computes the analytic signal using FFT-domain one-sided spectrum weighting. GPU inputs gather through the active provider until a dedicated analytic-signal provider hook lands.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::hilbert")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Hilbert transforms are FFT-domain operations and terminate fusion plans.",
};

const HILBERT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Analytic signal with real part equal to the input signal.",
}];

const HILBERT_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real-valued signal vector, matrix, or N-D array.",
}];

const HILBERT_INPUTS_WITH_N: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real-valued signal vector, matrix, or N-D array.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "FFT length along the first non-singleton dimension.",
    },
];

const HILBERT_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "z = hilbert(x)",
        inputs: &HILBERT_INPUTS_CORE,
        outputs: &HILBERT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "z = hilbert(x, N)",
        inputs: &HILBERT_INPUTS_WITH_N,
        outputs: &HILBERT_OUTPUT,
    },
];

const HILBERT_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HILBERT.ARG_COUNT",
    identifier: Some("RunMat:hilbert:ArgCount"),
    when: "More than two input arguments are supplied.",
    message: "hilbert: expected hilbert(X) or hilbert(X, N)",
};

const HILBERT_ERROR_INVALID_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HILBERT.INVALID_LENGTH",
    identifier: Some("RunMat:hilbert:InvalidLength"),
    when: "Length argument N is non-scalar, negative, non-finite, or fractional.",
    message: "hilbert: invalid length argument",
};

const HILBERT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HILBERT.INVALID_INPUT",
    identifier: Some("RunMat:hilbert:InvalidInput"),
    when: "Input cannot be converted to a real numeric/logical signal.",
    message: "hilbert: expected real numeric input",
};

const HILBERT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HILBERT.INTERNAL",
    identifier: Some("RunMat:hilbert:Internal"),
    when: "FFT execution or tensor shaping fails internally.",
    message: "hilbert: internal error",
};

const HILBERT_ERRORS: [BuiltinErrorDescriptor; 4] = [
    HILBERT_ERROR_ARG_COUNT,
    HILBERT_ERROR_INVALID_LENGTH,
    HILBERT_ERROR_INVALID_INPUT,
    HILBERT_ERROR_INTERNAL,
];

pub const HILBERT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HILBERT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &HILBERT_ERRORS,
};

fn hilbert_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    hilbert_error_with_message(error.message, error)
}

fn hilbert_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    hilbert_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn hilbert_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: RuntimeError,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn hilbert_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "hilbert",
    category = "math/signal",
    summary = "Construct the analytic signal with the Hilbert transform.",
    keywords = "hilbert,analytic signal,signal processing,fft,complex",
    type_resolver(fft_type),
    descriptor(crate::builtins::math::signal::hilbert::HILBERT_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::hilbert"
)]
async fn hilbert_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let length = parse_arguments(&rest)?;
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|source| {
                    hilbert_error_with_source(
                        &HILBERT_ERROR_INVALID_INPUT,
                        "gpu gather failed",
                        source,
                    )
                })?;
            hilbert_tensor(tensor, length)
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(hilbert_error_with_detail(
            &HILBERT_ERROR_INVALID_INPUT,
            "input must be real-valued",
        )),
        other => {
            let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, other).map_err(|detail| {
                hilbert_error_with_detail(&HILBERT_ERROR_INVALID_INPUT, detail)
            })?;
            hilbert_tensor(tensor, length)
        }
    }
}

fn parse_arguments(args: &[Value]) -> BuiltinResult<Option<usize>> {
    match args.len() {
        0 => Ok(None),
        1 => parse_length(&args[0], BUILTIN_NAME).map_err(|source| {
            hilbert_error_with_source(&HILBERT_ERROR_INVALID_LENGTH, "length parse failed", source)
        }),
        _ => Err(hilbert_error(&HILBERT_ERROR_ARG_COUNT)),
    }
}

fn hilbert_tensor(tensor: Tensor, length: Option<usize>) -> BuiltinResult<Value> {
    let complex = tensor_to_complex_tensor(tensor, BUILTIN_NAME).map_err(|source| {
        hilbert_error_with_source(&HILBERT_ERROR_INTERNAL, "input promotion failed", source)
    })?;
    let analytic = analytic_signal(complex, length)?;
    Ok(complex_tensor_into_value(analytic))
}

fn analytic_signal(tensor: ComplexTensor, length: Option<usize>) -> BuiltinResult<ComplexTensor> {
    let mut shape = tensor.shape.clone();
    if crate::builtins::common::shape::is_scalar_shape(&shape) {
        shape = crate::builtins::common::shape::normalize_scalar_shape(&shape);
    }
    let dim_one_based = default_dimension(&shape);
    let dim_index = dim_one_based - 1;
    validate_transform_allocation(&shape, dim_index, length)?;

    let spectrum = transform_complex_tensor(
        tensor,
        length,
        Some(dim_one_based),
        TransformDirection::Forward,
        BUILTIN_NAME,
    )
    .map_err(|source| hilbert_error_with_source(&HILBERT_ERROR_INTERNAL, "fft failed", source))?;
    let filtered = apply_analytic_signal_mask(spectrum, dim_index)?;
    transform_complex_tensor(
        filtered,
        None,
        Some(dim_one_based),
        TransformDirection::Inverse,
        BUILTIN_NAME,
    )
    .map_err(|source| hilbert_error_with_source(&HILBERT_ERROR_INTERNAL, "ifft failed", source))
}

fn validate_transform_allocation(
    shape: &[usize],
    dim_index: usize,
    length: Option<usize>,
) -> BuiltinResult<()> {
    let mut logical_shape = shape.to_vec();
    while logical_shape.len() <= dim_index {
        logical_shape.push(1);
    }
    let current_len = logical_shape[dim_index];
    let target_len = length.unwrap_or(current_len);
    if target_len == 0 {
        return Ok(());
    }

    let inner_stride = checked_product(&logical_shape[..dim_index])?;
    let outer_stride = checked_product(&logical_shape[dim_index + 1..])?;
    let num_slices = inner_stride.checked_mul(outer_stride).ok_or_else(|| {
        hilbert_error_with_detail(&HILBERT_ERROR_INVALID_LENGTH, "shape is too large")
    })?;
    let output_len = target_len.checked_mul(num_slices).ok_or_else(|| {
        hilbert_error_with_detail(
            &HILBERT_ERROR_INVALID_LENGTH,
            "requested length is too large",
        )
    })?;
    let max_complex_vec_len = isize::MAX as usize / size_of::<Complex<f64>>();
    if target_len > max_complex_vec_len || output_len > max_complex_vec_len {
        return Err(hilbert_error_with_detail(
            &HILBERT_ERROR_INVALID_LENGTH,
            "requested length is too large",
        ));
    }
    Ok(())
}

fn checked_product(dims: &[usize]) -> BuiltinResult<usize> {
    dims.iter().copied().try_fold(1usize, |acc, dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            hilbert_error_with_detail(&HILBERT_ERROR_INVALID_LENGTH, "shape is too large")
        })
    })
}

fn apply_analytic_signal_mask(
    mut spectrum: ComplexTensor,
    dim_index: usize,
) -> BuiltinResult<ComplexTensor> {
    let mut shape = spectrum.shape.clone();
    while shape.len() <= dim_index {
        shape.push(1);
    }

    let len = shape[dim_index];
    if len == 0 || spectrum.data.is_empty() {
        return Ok(spectrum);
    }

    let inner_stride = shape[..dim_index]
        .iter()
        .copied()
        .fold(1usize, |acc, dim| acc.saturating_mul(dim));
    let outer_stride = shape[dim_index + 1..]
        .iter()
        .copied()
        .fold(1usize, |acc, dim| acc.saturating_mul(dim));

    for outer in 0..outer_stride {
        let base = outer.saturating_mul(len.saturating_mul(inner_stride));
        for inner in 0..inner_stride {
            for freq in 0..len {
                let idx = base + inner + freq * inner_stride;
                let Some(slot) = spectrum.data.get_mut(idx) else {
                    return Err(hilbert_error_with_detail(
                        &HILBERT_ERROR_INTERNAL,
                        "frequency mask index out of bounds",
                    ));
                };
                let scale = analytic_signal_multiplier(freq, len);
                let value = Complex::new(slot.0, slot.1) * scale;
                *slot = (value.re, value.im);
            }
        }
    }

    Ok(spectrum)
}

fn analytic_signal_multiplier(freq: usize, len: usize) -> f64 {
    if len == 0 {
        return 0.0;
    }
    if freq == 0 {
        return 1.0;
    }
    if len.is_multiple_of(2) {
        if freq < len / 2 {
            2.0
        } else if freq == len / 2 {
            1.0
        } else {
            0.0
        }
    } else if freq <= len / 2 {
        2.0
    } else {
        0.0
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{ComplexTensor as HostComplexTensor, IntValue, LogicalArray, Type};

    const TOL: f64 = 1.0e-12;

    fn hilbert_call(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(hilbert_builtin(value, rest))
    }

    fn as_complex_tensor(value: Value) -> HostComplexTensor {
        match value {
            Value::ComplexTensor(tensor) => tensor,
            Value::Complex(re, im) => HostComplexTensor::new(vec![(re, im)], vec![1, 1]).unwrap(),
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
    fn hilbert_type_preserves_numeric_shape() {
        let out = fft_type(
            &[Type::Tensor {
                shape: Some(vec![Some(1), Some(4)]),
            }],
            &runmat_builtins::ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(4)])
            }
        );
    }

    #[test]
    fn hilbert_row_cosine_returns_quadrature_signal() {
        let input = Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![1, 4]).unwrap();
        let out = as_complex_tensor(hilbert_call(Value::Tensor(input), Vec::new()).unwrap());
        assert_eq!(out.shape, vec![1, 4]);
        let expected = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
        for (actual, expected) in out.data.iter().copied().zip(expected) {
            assert_complex_close(actual, expected);
        }
    }

    #[test]
    fn hilbert_column_cosine_operates_down_columns() {
        let input = Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![4, 1]).unwrap();
        let out = as_complex_tensor(hilbert_call(Value::Tensor(input), Vec::new()).unwrap());
        assert_eq!(out.shape, vec![4, 1]);
        let expected = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
        for (actual, expected) in out.data.iter().copied().zip(expected) {
            assert_complex_close(actual, expected);
        }
    }

    #[test]
    fn hilbert_matrix_operates_along_first_nonsingleton_dimension() {
        let input =
            Tensor::new(vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0], vec![4, 2]).unwrap();
        let out = as_complex_tensor(hilbert_call(Value::Tensor(input), Vec::new()).unwrap());
        assert_eq!(out.shape, vec![4, 2]);
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
        for (actual, expected) in out.data.iter().copied().zip(expected) {
            assert_complex_close(actual, expected);
        }
    }

    #[test]
    fn hilbert_length_argument_pads_or_truncates_transform_axis() {
        let input = Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![1, 4]).unwrap();
        let out =
            as_complex_tensor(hilbert_call(Value::Tensor(input), vec![Value::Num(6.0)]).unwrap());
        assert_eq!(out.shape, vec![1, 6]);
        assert_eq!(out.data.len(), 6);
        assert_complex_close(out.data[0], (1.0, 0.0));
    }

    #[test]
    fn hilbert_zero_length_returns_empty_along_transform_axis() {
        let input = Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![1, 4]).unwrap();
        let out = as_complex_tensor(
            hilbert_call(Value::Tensor(input), vec![Value::Int(IntValue::I32(0))]).unwrap(),
        );
        assert_eq!(out.shape, vec![1, 0]);
        assert!(out.data.is_empty());
    }

    #[test]
    fn hilbert_accepts_logical_input_as_real_signal() {
        let input = LogicalArray::new(vec![1, 0, 1, 0], vec![1, 4]).unwrap();
        let out = as_complex_tensor(hilbert_call(Value::LogicalArray(input), Vec::new()).unwrap());
        assert_eq!(out.shape, vec![1, 4]);
    }

    #[test]
    fn hilbert_rejects_complex_input() {
        let input = HostComplexTensor::new(vec![(1.0, 1.0)], vec![1, 1]).unwrap();
        let err = hilbert_call(Value::ComplexTensor(input), Vec::new()).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:hilbert:InvalidInput"));
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
    fn hilbert_gpu_input_gathers_and_computes_analytic_signal() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![1, 4]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &input.data,
                shape: &input.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let out =
                as_complex_tensor(hilbert_call(Value::GpuTensor(handle), Vec::new()).unwrap());
            assert_eq!(out.shape, vec![1, 4]);
            let expected = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
            for (actual, expected) in out.data.iter().copied().zip(expected) {
                assert_complex_close(actual, expected);
            }
        });
    }
}
