//! MATLAB-compatible `bounds` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::math::reduction::type_resolvers::reduce_numeric_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

fn bounds_type(args: &[Type], ctx: &ResolveContext) -> Type {
    reduce_numeric_type(args, ctx)
}

const OUTPUT_MIN: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Smallest values.",
};

const OUTPUT_MAX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "L",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Largest values.",
};

const OUTPUTS: [BuiltinParamDescriptor; 2] = [OUTPUT_MIN, OUTPUT_MAX];

const INPUT_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
};

const INPUT_DIM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Dimension selector, vector of dimensions, or \"all\".",
};

const INPUT_NANFLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nanflag",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"omitnan\""),
    description: "NaN handling mode: \"includenan\" or \"omitnan\".",
};

const INPUTS_A: [BuiltinParamDescriptor; 1] = [INPUT_A];
const INPUTS_A_DIM: [BuiltinParamDescriptor; 2] = [INPUT_A, INPUT_DIM];
const INPUTS_A_NANFLAG: [BuiltinParamDescriptor; 2] = [INPUT_A, INPUT_NANFLAG];
const INPUTS_A_DIM_NANFLAG: [BuiltinParamDescriptor; 3] = [INPUT_A, INPUT_DIM, INPUT_NANFLAG];

const BOUNDS_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "[S, L] = bounds(A)",
        inputs: &INPUTS_A,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[S, L] = bounds(A, dim)",
        inputs: &INPUTS_A_DIM,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[S, L] = bounds(A, vecdim)",
        inputs: &INPUTS_A_DIM,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[S, L] = bounds(A, nanflag)",
        inputs: &INPUTS_A_NANFLAG,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[S, L] = bounds(A, dim, nanflag)",
        inputs: &INPUTS_A_DIM_NANFLAG,
        outputs: &OUTPUTS,
    },
];

const BOUNDS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BOUNDS.INVALID_ARGUMENT",
    identifier: Some("RunMat:bounds:InvalidArgument"),
    when: "Dimension selectors, nanflags, or argument ordering are invalid.",
    message: "bounds: invalid argument",
};

const BOUNDS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BOUNDS.INTERNAL",
    identifier: Some("RunMat:bounds:Internal"),
    when: "Min/max evaluation fails internally.",
    message: "bounds: internal failure",
};

const BOUNDS_ERRORS: [BuiltinErrorDescriptor; 2] =
    [BOUNDS_ERROR_INVALID_ARGUMENT, BOUNDS_ERROR_INTERNAL];

pub const BOUNDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BOUNDS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BOUNDS_ERRORS,
};

#[runtime_builtin(
    name = "bounds",
    category = "math/reduction",
    summary = "Return smallest and largest elements along dimensions.",
    keywords = "bounds,min,max,reduction,omitnan",
    type_resolver(bounds_type),
    descriptor(crate::builtins::math::reduction::bounds::BOUNDS_DESCRIPTOR),
    builtin_path = "crate::builtins::math::reduction::bounds"
)]
async fn bounds_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    evaluate_bounds(value, &rest).await?.into_value()
}

#[derive(Debug)]
struct BoundsEvaluation {
    smallest: Value,
    largest: Value,
}

impl BoundsEvaluation {
    fn into_value(self) -> BuiltinResult<Value> {
        if let Some(out_count) = crate::output_count::current_output_count() {
            if out_count == 0 {
                return Ok(Value::OutputList(Vec::new()));
            }
            return Ok(crate::output_count::output_list_with_padding(
                out_count,
                vec![self.smallest, self.largest],
            ));
        }
        Ok(Value::OutputList(vec![self.smallest, self.largest]))
    }
}

async fn evaluate_bounds(value: Value, rest: &[Value]) -> BuiltinResult<BoundsEvaluation> {
    let rest = normalize_bounds_args(rest)?;
    let value = match value {
        Value::GpuTensor(handle) => Value::Tensor(
            gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| bounds_internal(err.message()))?,
        ),
        other => other,
    };
    let min_eval = super::min::evaluate(value.clone(), &rest)
        .await
        .map_err(|err| map_bounds_reduction_error("min", err))?;
    let max_eval = super::max::evaluate(value, &rest)
        .await
        .map_err(|err| map_bounds_reduction_error("max", err))?;
    Ok(BoundsEvaluation {
        smallest: min_eval.into_value(),
        largest: max_eval.into_value(),
    })
}

fn normalize_bounds_args(rest: &[Value]) -> BuiltinResult<Vec<Value>> {
    if rest.len() > 2 {
        return Err(bounds_invalid_argument(
            "bounds accepts at most a dimension selector and nanflag",
        ));
    }
    let has_nanflag = rest.iter().any(is_nanflag);
    let mut out = Vec::with_capacity(rest.len() + 2);
    out.push(empty_placeholder());
    out.extend(rest.iter().cloned());
    if !has_nanflag {
        out.push(Value::from("omitnan"));
    }
    Ok(out)
}

fn is_nanflag(value: &Value) -> bool {
    matches!(
        keyword_of(value).as_deref(),
        Some("omitnan" | "includenan" | "omitmissing" | "includemissing")
    )
}

fn map_bounds_reduction_error(label: &str, err: RuntimeError) -> RuntimeError {
    let detail = format!("{label} evaluation failed: {}", err.message());
    match err.identifier() {
        Some(identifier)
            if identifier.contains("InvalidArgument")
                || identifier.contains("InvalidInput")
                || identifier.contains("SizeMismatch") =>
        {
            bounds_invalid_argument(detail)
        }
        _ => bounds_internal(detail),
    }
}

fn empty_placeholder() -> Value {
    Value::Tensor(runmat_builtins::Tensor::new(Vec::new(), vec![0, 0]).unwrap())
}

fn bounds_error(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", descriptor.message, detail.as_ref()))
        .with_builtin("bounds");
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn bounds_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    bounds_error(&BOUNDS_ERROR_INVALID_ARGUMENT, detail)
}

fn bounds_internal(detail: impl AsRef<str>) -> RuntimeError {
    bounds_error(&BOUNDS_ERROR_INTERNAL, detail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{IntValue, IntegerStorage, Tensor};

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    #[test]
    fn bounds_descriptor_declares_runtime_nanflag_default() {
        assert_eq!(INPUT_NANFLAG.default, Some("\"omitnan\""));
    }

    #[tokio::test]
    async fn bounds_default_returns_min_and_max() {
        let input = tensor(vec![3.0, 1.0, 4.0, 2.0], vec![2, 2]);
        let result = evaluate_bounds(input, &[]).await.unwrap();
        let Value::Tensor(mins) = result.smallest else {
            panic!("expected min tensor");
        };
        assert_eq!(mins.materialize_f64(), vec![1.0, 2.0]);
        let Value::Tensor(maxes) = result.largest else {
            panic!("expected max tensor");
        };
        assert_eq!(maxes.materialize_f64(), vec![3.0, 4.0]);
    }

    #[tokio::test]
    async fn bounds_supports_all_and_omitnan() {
        let input = tensor(vec![f64::NAN, 3.0, -1.0, 5.0], vec![2, 2]);
        let result = evaluate_bounds(input, &[Value::from("all"), Value::from("omitnan")])
            .await
            .unwrap();
        assert_eq!(result.smallest, Value::Num(-1.0));
        assert_eq!(result.largest, Value::Num(5.0));
    }

    #[tokio::test]
    async fn bounds_omits_nan_by_default() {
        let input = tensor(vec![f64::NAN, 3.0, -1.0, 5.0], vec![2, 2]);
        let result = evaluate_bounds(input, &[Value::from("all")]).await.unwrap();
        assert_eq!(result.smallest, Value::Num(-1.0));
        assert_eq!(result.largest, Value::Num(5.0));
    }

    #[tokio::test]
    async fn bounds_preserves_exact_integer_extrema_for_dimensions_and_all() {
        let large = 9_007_199_254_740_992_u64;
        let input = Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![large + 1, large, u64::MAX, 7]),
                vec![2, 2],
            )
            .unwrap(),
        );

        let reduced = evaluate_bounds(input.clone(), &[]).await.unwrap();
        let Value::Tensor(minima) = reduced.smallest else {
            panic!("expected integer minima tensor");
        };
        let Value::Tensor(maxima) = reduced.largest else {
            panic!("expected integer maxima tensor");
        };
        assert_eq!(
            minima.integer_storage(),
            Some(&IntegerStorage::U64(vec![large, 7]))
        );
        assert_eq!(
            maxima.integer_storage(),
            Some(&IntegerStorage::U64(vec![large + 1, u64::MAX]))
        );

        let all = evaluate_bounds(input, &[Value::from("all")]).await.unwrap();
        assert_eq!(all.smallest, Value::Int(IntValue::U64(7)));
        assert_eq!(all.largest, Value::Int(IntValue::U64(u64::MAX)));
    }

    #[tokio::test]
    async fn bounds_invalid_reduction_arguments_use_bounds_identifier() {
        let input = tensor(vec![1.0, 2.0], vec![2, 1]);
        let err = evaluate_bounds(input, &[Value::Num(0.0)])
            .await
            .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:bounds:InvalidArgument"));
    }

    #[tokio::test]
    async fn bounds_builtin_pads_requested_outputs() {
        let input = tensor(vec![1.0, 4.0], vec![2, 1]);
        let _guard = crate::output_count::push_output_count(Some(3));
        let result = bounds_builtin(input, Vec::new()).await.unwrap();
        let Value::OutputList(values) = result else {
            panic!("expected output list");
        };
        assert_eq!(values.len(), 3);
        assert_eq!(values[0], Value::Num(1.0));
        assert_eq!(values[1], Value::Num(4.0));
    }
}
