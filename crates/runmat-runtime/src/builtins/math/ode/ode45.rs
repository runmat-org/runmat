//! MATLAB-compatible `ode45` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::ode::common::{
    build_ode_output, ode_options_from_struct, parse_ode_input, parse_options, solve_ode, OdeMethod,
};
use crate::builtins::math::ode::type_resolvers::ode_solution_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "ode45";

const ODE45_OUTPUT_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Solution states evaluated over tspan.",
}];

const ODE45_OUTPUT_TY: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "t",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Time points selected by solver.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Solution states at each returned time point.",
    },
];

const ODE45_INPUTS_CORE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "odefun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "ODE right-hand-side callback f(t,y).",
    },
    BuiltinParamDescriptor {
        name: "tspan",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Time interval or monotonic time vector.",
    },
    BuiltinParamDescriptor {
        name: "y0",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Initial state vector/value.",
    },
];

const ODE45_INPUTS_WITH_OPTIONS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "odefun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "ODE right-hand-side callback f(t,y).",
    },
    BuiltinParamDescriptor {
        name: "tspan",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Time interval or monotonic time vector.",
    },
    BuiltinParamDescriptor {
        name: "y0",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Initial state vector/value.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional struct with tolerances and step controls.",
    },
];

const ODE45_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "y = ode45(odefun, tspan, y0)",
        inputs: &ODE45_INPUTS_CORE,
        outputs: &ODE45_OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = ode45(odefun, tspan, y0, options)",
        inputs: &ODE45_INPUTS_WITH_OPTIONS,
        outputs: &ODE45_OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "[t, y] = ode45(odefun, tspan, y0)",
        inputs: &ODE45_INPUTS_CORE,
        outputs: &ODE45_OUTPUT_TY,
    },
    BuiltinSignatureDescriptor {
        label: "[t, y] = ode45(odefun, tspan, y0, options)",
        inputs: &ODE45_INPUTS_WITH_OPTIONS,
        outputs: &ODE45_OUTPUT_TY,
    },
];

const ODE45_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ODE45.INVALID_ARGUMENT",
    identifier: Some("RunMat:ode45:InvalidArgument"),
    when: "Input argument count/options struct grammar is invalid.",
    message: "ode45: invalid argument",
};

const ODE45_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ODE45.INVALID_INPUT",
    identifier: Some("RunMat:ode45:InvalidInput"),
    when: "ODE input/state/callback semantics are invalid for integration.",
    message: "ode45: invalid input",
};

const ODE45_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ODE45.INTERNAL",
    identifier: Some("RunMat:ode45:Internal"),
    when: "Internal output materialization fails.",
    message: "ode45: internal runtime failure",
};

const ODE45_ERRORS: [BuiltinErrorDescriptor; 3] = [
    ODE45_ERROR_INVALID_ARGUMENT,
    ODE45_ERROR_INVALID_INPUT,
    ODE45_ERROR_INTERNAL,
];

pub const ODE45_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ODE45_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ODE45_ERRORS,
};

fn ode45_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("ode45:") {
        detail.to_string()
    } else {
        format!("{}: {}", error.message, detail)
    };
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn ode45_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        ode45_error_with_detail(fallback, err.message())
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::ode::ode45")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ode45",
    op_kind: GpuOpKind::Custom("ode-solve"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Adaptive ODE integration runs on the host. RHS callbacks may call GPU-aware builtins.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::ode::ode45")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ode45",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "ODE integration repeatedly invokes user callbacks and terminates fusion planning.",
};

#[runtime_builtin(
    name = "ode45",
    category = "math/ode",
    summary = "Solve nonstiff ODE systems using adaptive Dormand-Prince 5(4) integration.",
    keywords = "ode45,ode,nonstiff,dormand-prince,adaptive step",
    accel = "sink",
    type_resolver(ode_solution_type),
    descriptor(crate::builtins::math::ode::ode45::ODE45_DESCRIPTOR),
    builtin_path = "crate::builtins::math::ode::ode45"
)]
async fn ode45_builtin(
    function: Value,
    tspan: Value,
    y0: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(ode45_error_with_detail(
            &ODE45_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        ));
    }
    let options = parse_options(NAME, rest.first())
        .map_err(|err| ode45_map_error(err, &ODE45_ERROR_INVALID_ARGUMENT))?;
    let opts = ode_options_from_struct(NAME, options.as_ref())
        .map_err(|err| ode45_map_error(err, &ODE45_ERROR_INVALID_ARGUMENT))?;
    let input = parse_ode_input(NAME, tspan, y0)
        .await
        .map_err(|err| ode45_map_error(err, &ODE45_ERROR_INVALID_INPUT))?;
    let result = solve_ode(NAME, OdeMethod::Ode45, &function, &input, &opts)
        .await
        .map_err(|err| ode45_map_error(err, &ODE45_ERROR_INVALID_INPUT))?;
    build_ode_output(NAME, result).map_err(|err| ode45_map_error(err, &ODE45_ERROR_INTERNAL))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::Tensor;
    use std::sync::Arc;

    #[test]
    fn ode45_scalar_decay_returns_reasonable_final_value() {
        let _resolver =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|_name| {
                Some(0)
            })));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |_function, args, _requested_outputs| {
                let y = match &args[1] {
                    Value::Num(n) => *n,
                    other => panic!("expected scalar state, got {other:?}"),
                };
                Box::pin(async move { Ok(Value::Num(-y)) })
            },
        )));

        let out = block_on(ode45_builtin(
            Value::FunctionHandle("decay".into()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            Vec::new(),
        ))
        .unwrap();

        match out {
            Value::Tensor(t) => {
                assert_eq!(t.cols(), 1);
                let last = t.data[t.rows() - 1];
                assert!((last - (-1.0_f64).exp()).abs() < 5.0e-3);
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn ode45_rejects_nan_rhs() {
        let _resolver =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|_name| {
                Some(0)
            })));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |_function, _args, _requested_outputs| {
                Box::pin(async move { Ok(Value::Num(f64::NAN)) })
            },
        )));

        let err = block_on(ode45_builtin(
            Value::FunctionHandle("nan_rhs".into()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            Vec::new(),
        ))
        .expect_err("ode45 should reject NaN derivative values");

        assert!(err.to_string().contains("function value must be finite"));
    }

    #[test]
    fn ode45_accepts_external_function_handle_rhs() {
        let _resolver =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.decay").then_some(56)
            })));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |function, args, _requested_outputs| {
                assert_eq!(function, 56);
                let y = match &args[1] {
                    Value::Num(n) => *n,
                    other => panic!("expected scalar state, got {other:?}"),
                };
                Box::pin(async move { Ok(Value::Num(-y)) })
            },
        )));

        let out = block_on(ode45_builtin(
            Value::ExternalFunctionHandle("pkg.decay".to_string()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            Vec::new(),
        ))
        .unwrap();

        match out {
            Value::Tensor(t) => {
                assert_eq!(t.cols(), 1);
                let last = t.data[t.rows() - 1];
                assert!(last.is_finite());
                assert!(last > 0.0);
                assert!(last < 1.0);
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn ode45_too_many_inputs_uses_stable_identifier() {
        let err = block_on(ode45_builtin(
            Value::FunctionHandle("decay".into()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            vec![Value::Num(1.0), Value::Num(2.0)],
        ))
        .expect_err("expected too many inputs error");
        assert_eq!(err.identifier(), ODE45_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn ode45_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = ODE45_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(
            labels,
            vec![
                "y = ode45(odefun, tspan, y0)",
                "y = ode45(odefun, tspan, y0, options)",
                "[t, y] = ode45(odefun, tspan, y0)",
                "[t, y] = ode45(odefun, tspan, y0, options)",
            ]
        );
    }

    #[test]
    fn ode45_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = ODE45_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert_eq!(
            codes,
            vec![
                "RM.ODE45.INVALID_ARGUMENT",
                "RM.ODE45.INVALID_INPUT",
                "RM.ODE45.INTERNAL",
            ]
        );
    }
}
