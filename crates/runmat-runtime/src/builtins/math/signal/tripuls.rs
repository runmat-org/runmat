//! MATLAB-compatible `tripuls` builtin for triangular pulse samples.

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

const BUILTIN_NAME: &str = "tripuls";

const TRIPULS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Triangular pulse samples.",
}];

const TRIPULS_INPUTS_T: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample times relative to the pulse center.",
}];

const TRIPULS_INPUTS_T_W: [BuiltinParamDescriptor; 2] = [
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

const TRIPULS_INPUTS_T_W_S: [BuiltinParamDescriptor; 3] = [
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
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Skew in [-1, 1].",
    },
];

const TRIPULS_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "Y = tripuls(T)",
        inputs: &TRIPULS_INPUTS_T,
        outputs: &TRIPULS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = tripuls(T, W)",
        inputs: &TRIPULS_INPUTS_T_W,
        outputs: &TRIPULS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = tripuls(T, W, S)",
        inputs: &TRIPULS_INPUTS_T_W_S,
        outputs: &TRIPULS_OUTPUT,
    },
];

const TRIPULS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIPULS.INVALID_INPUT",
    identifier: Some("RunMat:tripuls:InvalidInput"),
    when: "Input times cannot be interpreted as real numeric samples.",
    message: "tripuls: expected real numeric input",
};

const TRIPULS_ERROR_WIDTH_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIPULS.WIDTH_INVALID",
    identifier: Some("RunMat:tripuls:WidthInvalid"),
    when: "Width is not a positive finite scalar.",
    message: "tripuls: width must be a positive finite scalar",
};

const TRIPULS_ERROR_SKEW_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIPULS.SKEW_INVALID",
    identifier: Some("RunMat:tripuls:SkewInvalid"),
    when: "Skew is not a finite scalar in [-1, 1].",
    message: "tripuls: skew must be a finite scalar in [-1, 1]",
};

const TRIPULS_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIPULS.ARG_COUNT",
    identifier: Some("RunMat:tripuls:ArgCount"),
    when: "Too many input arguments are provided.",
    message: "tripuls: expected 1, 2, or 3 arguments",
};

const TRIPULS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIPULS.INTERNAL",
    identifier: Some("RunMat:tripuls:InternalError"),
    when: "Internal tensor construction or GPU gather fails.",
    message: "tripuls: internal error",
};

const TRIPULS_ERRORS: [BuiltinErrorDescriptor; 5] = [
    TRIPULS_ERROR_INVALID_INPUT,
    TRIPULS_ERROR_WIDTH_INVALID,
    TRIPULS_ERROR_SKEW_INVALID,
    TRIPULS_ERROR_ARG_COUNT,
    TRIPULS_ERROR_INTERNAL,
];

pub const TRIPULS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRIPULS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TRIPULS_ERRORS,
};

fn tripuls_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    tripuls_error_with_message(error.message, error)
}

fn tripuls_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    tripuls_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn tripuls_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn tripuls_error_with_source(
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

pub(crate) fn tripuls_scalar(t: f64, width: f64, skew: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    let left = -width / 2.0;
    let right = width / 2.0;
    if t < left || t > right {
        return 0.0;
    }
    let peak = skew * width / 2.0;
    if t == peak {
        return 1.0;
    }
    if t < peak {
        let span = peak - left;
        if span <= 0.0 {
            1.0
        } else {
            (t - left) / span
        }
    } else {
        let span = right - peak;
        if span <= 0.0 {
            1.0
        } else {
            (right - t) / span
        }
    }
}

pub(crate) fn tripuls_tensor(tensor: Tensor, width: f64, skew: f64) -> Result<Tensor, String> {
    let shape = tensor.shape.clone();
    let data = tensor
        .data
        .iter()
        .map(|&value| tripuls_scalar(value, width, skew))
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

pub(crate) fn validate_skew(skew: f64) -> Result<f64, String> {
    if !skew.is_finite() || !(-1.0..=1.0).contains(&skew) {
        Err(format!("got {skew}"))
    } else {
        Ok(skew)
    }
}

#[runtime_builtin(
    name = "tripuls",
    category = "math/signal",
    summary = "Generate sampled triangular pulses.",
    keywords = "tripuls,triangular pulse,pulse train,signal processing,skew",
    type_resolver(numeric_unary_shape_type),
    descriptor(crate::builtins::math::signal::tripuls::TRIPULS_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::tripuls"
)]
async fn tripuls_builtin(t: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let (width, skew) = parse_options(&rest).await?;
    match t {
        Value::GpuTensor(handle) => tripuls_gpu(handle, width, skew).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(tripuls_error(&TRIPULS_ERROR_INVALID_INPUT))
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            Err(tripuls_error(&TRIPULS_ERROR_INVALID_INPUT))
        }
        other => tripuls_real(other, width, skew),
    }
}

async fn parse_options(rest: &[Value]) -> BuiltinResult<(f64, f64)> {
    if rest.len() > 2 {
        return Err(tripuls_error_with_detail(
            &TRIPULS_ERROR_ARG_COUNT,
            format!("got {}", rest.len() + 1),
        ));
    }
    let width = match rest.first() {
        Some(value) => {
            let raw = scalar_f64_from_value_async(value)
                .await
                .map_err(|err| tripuls_error_with_detail(&TRIPULS_ERROR_WIDTH_INVALID, err))?
                .ok_or_else(|| tripuls_error(&TRIPULS_ERROR_WIDTH_INVALID))?;
            validate_width(raw).map_err(|err| {
                tripuls_error_with_detail(&TRIPULS_ERROR_WIDTH_INVALID, err.as_str())
            })?
        }
        None => 1.0,
    };
    let skew = match rest.get(1) {
        Some(value) => {
            let raw = scalar_f64_from_value_async(value)
                .await
                .map_err(|err| tripuls_error_with_detail(&TRIPULS_ERROR_SKEW_INVALID, err))?
                .ok_or_else(|| tripuls_error(&TRIPULS_ERROR_SKEW_INVALID))?;
            validate_skew(raw).map_err(|err| {
                tripuls_error_with_detail(&TRIPULS_ERROR_SKEW_INVALID, err.as_str())
            })?
        }
        None => 0.0,
    };
    Ok((width, skew))
}

async fn tripuls_gpu(handle: GpuTensorHandle, width: f64, skew: f64) -> BuiltinResult<Value> {
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|source| {
            tripuls_error_with_source(&TRIPULS_ERROR_INTERNAL, "gpu gather failed", source)
        })?;
    tripuls_tensor(tensor, width, skew)
        .map(tensor_into_value)
        .map_err(|err| tripuls_error_with_detail(&TRIPULS_ERROR_INTERNAL, err))
}

fn tripuls_real(value: Value, width: f64, skew: f64) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|err| tripuls_error_with_detail(&TRIPULS_ERROR_INVALID_INPUT, err))?;
    tripuls_tensor(tensor, width, skew)
        .map(tensor_into_value)
        .map_err(|err| tripuls_error_with_detail(&TRIPULS_ERROR_INTERNAL, err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{builtin_function_by_name, CharArray};

    fn call(t: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(tripuls_builtin(t, rest))
    }

    fn expect_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn tripuls_default_samples_triangle() {
        let input = Tensor::new(vec![-0.5, -0.25, 0.0, 0.25, 0.5], vec![1, 5]).unwrap();
        let out = expect_tensor(call(Value::Tensor(input), Vec::new()).expect("tripuls"));
        assert_eq!(out.shape, vec![1, 5]);
        assert_eq!(out.data, vec![0.0, 0.5, 1.0, 0.5, 0.0]);
    }

    #[test]
    fn tripuls_skew_moves_peak() {
        let input = Tensor::new(vec![-1.0, 0.0, 1.0], vec![1, 3]).unwrap();
        let out = expect_tensor(
            call(Value::Tensor(input), vec![Value::Num(2.0), Value::Num(1.0)]).expect("tripuls"),
        );
        assert_eq!(out.data, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn tripuls_rejects_bad_options_and_text_input() {
        let err = call(Value::Num(0.0), vec![Value::Num(-1.0)]).expect_err("width");
        assert_eq!(err.identifier(), TRIPULS_ERROR_WIDTH_INVALID.identifier);

        let err = call(Value::Num(0.0), vec![Value::Num(1.0), Value::Num(2.0)]).expect_err("skew");
        assert_eq!(err.identifier(), TRIPULS_ERROR_SKEW_INVALID.identifier);

        let err =
            call(Value::CharArray(CharArray::new_row("abc")), Vec::new()).expect_err("text input");
        assert_eq!(err.identifier(), TRIPULS_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn tripuls_is_registered() {
        assert!(builtin_function_by_name("tripuls").is_some());
    }
}
