//! MATLAB-compatible legacy `interp1q` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::Tensor;

use super::pp::{
    evaluate_linear_or_nearest, query_points, series_from_values, Extrapolation, InterpMethod,
};

const NAME: &str = "interp1q";

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Vq",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Linearly interpolated query values.",
}];

const INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Strictly increasing sample locations.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query locations.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Vq = interp1q(X, V, Xq)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERP1Q.INVALID_INPUT",
    identifier: Some("RunMat:interp1q:InvalidInput"),
    when:
        "Arguments cannot be converted to the legacy one-dimensional linear interpolation domain.",
    message: "interp1q: invalid input",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERP1Q.INTERNAL",
    identifier: Some("RunMat:interp1q:Internal"),
    when: "Interpolation evaluation or output construction fails unexpectedly.",
    message: "interp1q: internal interpolation failure",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_INTERNAL];

pub const INTERP1Q_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::interpolation::interp1q")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("interpolation-1d-linear"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Legacy interp1q gathers GPU inputs and evaluates through the CPU linear interpolation path.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::interpolation::interp1q"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Interpolation is a runtime sink.",
};

fn interp1q_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args.get(2) {
        Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
        Some(Type::Tensor { shape }) | Some(Type::Logical { shape }) => Type::Tensor {
            shape: shape.clone(),
        },
        _ => Type::tensor(),
    }
}

#[runtime_builtin(
    name = "interp1q",
    category = "math/interpolation",
    summary = "Legacy quick one-dimensional linear interpolation.",
    keywords = "interp1q,interpolation,linear,legacy",
    accel = "sink",
    sink = true,
    type_resolver(interp1q_type),
    descriptor(crate::builtins::math::interpolation::interp1q::INTERP1Q_DESCRIPTOR),
    builtin_path = "crate::builtins::math::interpolation::interp1q"
)]
async fn interp1q_builtin(x: Value, v: Value, xq: Value) -> BuiltinResult<Value> {
    let series = series_from_values(x, v, NAME)
        .await
        .map_err(|err| wrap_error(err, &ERROR_INVALID_INPUT))?;
    let query = query_points(xq, NAME)
        .await
        .map_err(|err| wrap_error(err, &ERROR_INVALID_INPUT))?;
    let result = evaluate_linear_or_nearest(
        &series,
        &query,
        InterpMethod::Linear,
        &Extrapolation::Nan,
        NAME,
    )
    .map_err(|err| wrap_error(err, &ERROR_INTERNAL))?;
    reshape_matrix_output(result, &series, &query)
}

fn wrap_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(err.message().to_string()).with_builtin(NAME);
    if let Some(identifier) = err.identifier().or(fallback.identifier) {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn reshape_matrix_output(
    result: Value,
    series: &super::pp::NumericSeries,
    query: &super::pp::QueryPoints,
) -> BuiltinResult<Value> {
    if series.series <= 1 {
        return Ok(result);
    }
    let Value::Tensor(tensor) = result else {
        return Ok(result);
    };
    let mut shape = vec![query.values.len()];
    if series.trailing_shape.is_empty() {
        shape.push(series.series);
    } else {
        shape.extend(series.trailing_shape.iter().copied());
    }
    Tensor::new(tensor.data, shape)
        .map(Value::Tensor)
        .map_err(|err| {
            build_runtime_error(format!("interp1q: {err}"))
                .with_builtin(NAME)
                .build()
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::Tensor;

    fn row(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![1, values.len()]).expect("tensor"))
    }

    #[test]
    fn interp1q_linearly_interpolates() {
        let out = block_on(interp1q_builtin(
            row(&[1.0, 2.0, 3.0]),
            row(&[10.0, 20.0, 40.0]),
            row(&[1.5, 2.5]),
        ))
        .expect("interp1q");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![1, 2]);
        assert_eq!(tensor.data, vec![15.0, 30.0]);
    }

    #[test]
    fn interp1q_out_of_range_returns_nan() {
        let out = block_on(interp1q_builtin(
            row(&[1.0, 2.0]),
            row(&[10.0, 20.0]),
            Value::Num(3.0),
        ))
        .expect("interp1q");
        let Value::Num(value) = out else {
            panic!("expected scalar");
        };
        assert!(value.is_nan());
    }

    #[test]
    fn interp1q_matrix_values_return_query_length_by_series_shape() {
        let x = row(&[1.0, 2.0, 3.0]);
        let v = Value::Tensor(
            Tensor::new(
                vec![
                    10.0, 20.0, 40.0, //
                    100.0, 200.0, 400.0,
                ],
                vec![3, 2],
            )
            .expect("matrix values"),
        );
        let xq = Value::Tensor(Tensor::new(vec![1.5, 2.5], vec![2, 1]).expect("query"));
        let out = block_on(interp1q_builtin(x, v, xq)).expect("interp1q");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.data, vec![15.0, 30.0, 150.0, 300.0]);
    }

    #[test]
    fn interp1q_rejects_unsorted_x() {
        let err = block_on(interp1q_builtin(
            row(&[2.0, 1.0]),
            row(&[20.0, 10.0]),
            Value::Num(1.5),
        ))
        .expect_err("unsorted x should fail");
        assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
    }
}
