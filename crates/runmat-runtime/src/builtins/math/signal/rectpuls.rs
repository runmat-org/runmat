//! MATLAB-compatible `rectpuls` builtin for rectangular pulse samples.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor::{scalar_f64_from_value_async, tensor_into_value};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::signal::type_resolvers::numeric_unary_shape_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "rectpuls";

const RECTPULS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Rectangular pulse samples.",
}];

const RECTPULS_INPUTS_T: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample times relative to the pulse center.",
}];

const RECTPULS_INPUTS_T_W: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample times relative to the pulse center.",
    },
    BuiltinParamDescriptor {
        name: "W",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Pulse width.",
    },
];

const RECTPULS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = rectpuls(T)",
        inputs: &RECTPULS_INPUTS_T,
        outputs: &RECTPULS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = rectpuls(T, W)",
        inputs: &RECTPULS_INPUTS_T_W,
        outputs: &RECTPULS_OUTPUT,
    },
];

const RECTPULS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RECTPULS.INVALID_INPUT",
    identifier: Some("RunMat:rectpuls:InvalidInput"),
    when: "Input times cannot be interpreted as real numeric samples.",
    message: "rectpuls: expected real numeric input",
};

const RECTPULS_ERROR_WIDTH_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RECTPULS.WIDTH_INVALID",
    identifier: Some("RunMat:rectpuls:WidthInvalid"),
    when: "Width is not a positive finite scalar.",
    message: "rectpuls: width must be a positive finite scalar",
};

const RECTPULS_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RECTPULS.ARG_COUNT",
    identifier: Some("RunMat:rectpuls:ArgCount"),
    when: "Too many input arguments are provided.",
    message: "rectpuls: expected 1 or 2 arguments",
};

const RECTPULS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RECTPULS.INTERNAL",
    identifier: Some("RunMat:rectpuls:InternalError"),
    when: "Internal tensor construction or GPU gather fails.",
    message: "rectpuls: internal error",
};

const RECTPULS_ERRORS: [BuiltinErrorDescriptor; 4] = [
    RECTPULS_ERROR_INVALID_INPUT,
    RECTPULS_ERROR_WIDTH_INVALID,
    RECTPULS_ERROR_ARG_COUNT,
    RECTPULS_ERROR_INTERNAL,
];

pub const RECTPULS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RECTPULS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RECTPULS_ERRORS,
};

fn rectpuls_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    rectpuls_error_with_message(error.message, error)
}

fn rectpuls_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    rectpuls_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn rectpuls_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn rectpuls_error_with_source(
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

pub(crate) fn rectpuls_scalar(t: f64, width: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    let distance = t.abs();
    let half_width = width / 2.0;
    if distance < half_width {
        1.0
    } else if distance == half_width {
        0.5
    } else {
        0.0
    }
}

pub(crate) fn rectpuls_tensor(tensor: Tensor, width: f64) -> Result<Tensor, String> {
    let shape = tensor.shape.clone();
    let data = tensor
        .data
        .iter()
        .map(|&value| rectpuls_scalar(value, width))
        .collect::<Vec<_>>();
    Tensor::new(data, shape).map_err(|err| err.to_string())
}

pub(crate) fn validate_width(width: f64) -> Result<f64, String> {
    if !width.is_finite() || width <= 0.0 {
        Err(format!("got {width}"))
    } else {
        Ok(width)
    }
}

#[runtime_builtin(
    name = "rectpuls",
    category = "math/signal",
    summary = "Generate sampled rectangular pulses.",
    keywords = "rectpuls,rectangular pulse,pulse train,signal processing",
    type_resolver(numeric_unary_shape_type),
    descriptor(crate::builtins::math::signal::rectpuls::RECTPULS_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::rectpuls"
)]
async fn rectpuls_builtin(t: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let width = parse_width(&rest).await?;
    match t {
        Value::GpuTensor(handle) => rectpuls_gpu(handle, width).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(rectpuls_error(&RECTPULS_ERROR_INVALID_INPUT))
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            Err(rectpuls_error(&RECTPULS_ERROR_INVALID_INPUT))
        }
        other => rectpuls_real(other, width),
    }
}

async fn parse_width(rest: &[Value]) -> BuiltinResult<f64> {
    match rest.len() {
        0 => Ok(1.0),
        1 => {
            let raw = scalar_f64_from_value_async(&rest[0])
                .await
                .map_err(|err| rectpuls_error_with_detail(&RECTPULS_ERROR_WIDTH_INVALID, err))?
                .ok_or_else(|| rectpuls_error(&RECTPULS_ERROR_WIDTH_INVALID))?;
            validate_width(raw).map_err(|err| {
                rectpuls_error_with_detail(&RECTPULS_ERROR_WIDTH_INVALID, err.as_str())
            })
        }
        _ => Err(rectpuls_error_with_detail(
            &RECTPULS_ERROR_ARG_COUNT,
            format!("got {}", rest.len() + 1),
        )),
    }
}

async fn rectpuls_gpu(handle: GpuTensorHandle, width: f64) -> BuiltinResult<Value> {
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|source| {
            rectpuls_error_with_source(&RECTPULS_ERROR_INTERNAL, "gpu gather failed", source)
        })?;
    rectpuls_tensor(tensor, width)
        .map(tensor_into_value)
        .map_err(|err| rectpuls_error_with_detail(&RECTPULS_ERROR_INTERNAL, err))
}

fn rectpuls_real(value: Value, width: f64) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|err| rectpuls_error_with_detail(&RECTPULS_ERROR_INVALID_INPUT, err))?;
    rectpuls_tensor(tensor, width)
        .map(tensor_into_value)
        .map_err(|err| rectpuls_error_with_detail(&RECTPULS_ERROR_INTERNAL, err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{builtin_function_by_name, CharArray, ResolveContext, Type};

    fn call(t: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(rectpuls_builtin(t, rest))
    }

    fn expect_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn rectpuls_type_preserves_input_shape() {
        let out = numeric_unary_shape_type(
            &[Type::Tensor {
                shape: Some(vec![Some(1), Some(4)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(4)])
            }
        );
    }

    #[test]
    fn rectpuls_default_width_samples_rectangle() {
        let input = Tensor::new(vec![-0.5, -0.25, 0.0, 0.25, 0.5, 0.75], vec![1, 6]).unwrap();
        let out = expect_tensor(call(Value::Tensor(input), Vec::new()).expect("rectpuls"));
        assert_eq!(out.shape, vec![1, 6]);
        assert_eq!(out.data, vec![0.5, 1.0, 1.0, 1.0, 0.5, 0.0]);
    }

    #[test]
    fn rectpuls_custom_width_and_scalar_output() {
        let value = call(Value::Num(0.75), vec![Value::Num(2.0)]).expect("rectpuls");
        assert!(matches!(value, Value::Num(n) if n == 1.0));
    }

    #[test]
    fn rectpuls_rejects_bad_width_and_text_input() {
        let err = call(Value::Num(0.0), vec![Value::Num(0.0)]).expect_err("width");
        assert_eq!(err.identifier(), RECTPULS_ERROR_WIDTH_INVALID.identifier);

        let err =
            call(Value::CharArray(CharArray::new_row("abc")), Vec::new()).expect_err("text input");
        assert_eq!(err.identifier(), RECTPULS_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn rectpuls_is_registered() {
        assert!(builtin_function_by_name("rectpuls").is_some());
    }
}
