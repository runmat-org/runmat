//! MATLAB-compatible `ode15s` builtin.

#[cfg(test)]
use runmat_builtins::ODE15S_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, ODE15S_ERROR_INTERNAL, ODE15S_ERROR_INVALID_ARGUMENT,
    ODE15S_ERROR_INVALID_INPUT, ODE15S_INTEGER_CALLBACK_EXTENSION as INTEGER_CALLBACK_EXTENSION,
    ODE15S_INTEGER_OPTION_EXTENSION as INTEGER_OPTION_EXTENSION,
    ODE15S_INTEGER_TSPAN_EXTENSION as INTEGER_TSPAN_EXTENSION,
    ODE15S_INTEGER_Y0_EXTENSION as INTEGER_Y0_EXTENSION,
    ODE15S_LOGICAL_NUMERIC_EXTENSION as LOGICAL_NUMERIC_EXTENSION,
    ODE15S_RESIDENT_INPUT_EXTENSION as RESIDENT_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::ode::common::{
    build_ode_output, check_cancellation, parse_ode_input, OdeCompatibilityExtensions, OdeMethod,
};
use crate::builtins::math::ode::options;
use crate::builtins::math::ode::solver::solve_ode;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "ode15s";

pub(crate) const ODE_COMPATIBILITY_EXTENSIONS: OdeCompatibilityExtensions =
    OdeCompatibilityExtensions {
        integer_tspan: &INTEGER_TSPAN_EXTENSION,
        integer_y0: &INTEGER_Y0_EXTENSION,
        integer_option: &INTEGER_OPTION_EXTENSION,
        integer_callback: &INTEGER_CALLBACK_EXTENSION,
        logical_numeric: &LOGICAL_NUMERIC_EXTENSION,
        resident_input: &RESIDENT_INPUT_EXTENSION,
    };

fn ode15s_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("ode15s:") {
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

fn ode15s_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        ode15s_error_with_detail(fallback, err.message())
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::ode::ode15s")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ode15s",
    op_kind: GpuOpKind::Custom("ode-solve-stiff"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Stiff ODE integration runs on the host. RHS callbacks may call GPU-aware builtins.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::ode::ode15s")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ode15s",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "ODE integration repeatedly invokes user callbacks and terminates fusion planning.",
};

#[runtime_builtin(
    name = "ode15s",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::ode::ode15s"
)]
async fn ode15s_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    function: Value,
    tspan: Value,
    y0: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ode15s_in_runtime(runtime.as_ref(), function, tspan, y0, rest).await
}

#[cfg(test)]
async fn ode15s_builtin(
    function: Value,
    tspan: Value,
    y0: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ode15s_in_runtime(None, function, tspan, y0, rest).await
}

async fn ode15s_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    function: Value,
    tspan: Value,
    y0: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::resource_ingress::require_unbounded(NAME, runtime)?;
    check_cancellation(NAME, runtime)?;
    if rest.len() > 1 {
        return Err(ode15s_error_with_detail(
            &ODE15S_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        ));
    }
    let options = options::parse(NAME, rest.into_iter().next())
        .map_err(|err| ode15s_map_error(err, &ODE15S_ERROR_INVALID_ARGUMENT))?;
    let opts = options::admit(NAME, options, ODE_COMPATIBILITY_EXTENSIONS)
        .await
        .map_err(|err| ode15s_map_error(err, &ODE15S_ERROR_INVALID_ARGUMENT))?;
    check_cancellation(NAME, runtime)?;
    let input = parse_ode_input(NAME, tspan, y0, ODE_COMPATIBILITY_EXTENSIONS, runtime)
        .await
        .map_err(|err| ode15s_map_error(err, &ODE15S_ERROR_INVALID_INPUT))?;
    let result = solve_ode(NAME, OdeMethod::Ode15s, &function, &input, &opts)
        .await
        .map_err(|err| ode15s_map_error(err, &ODE15S_ERROR_INVALID_INPUT))?;
    check_cancellation(NAME, runtime)?;
    build_ode_output(NAME, result).map_err(|err| ode15s_map_error(err, &ODE15S_ERROR_INTERNAL))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::Tensor;
    use std::sync::Arc;

    #[test]
    fn ode15s_handles_linear_stiff_decay() {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _fixtures = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("stiff_decay", 571)],
        )
        .unwrap();
        let entry = runtime.enter();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |function, args, _requested_outputs| {
                assert_eq!(function, 571);
                let y = match &args[1] {
                    Value::Num(n) => *n,
                    other => panic!("expected scalar state, got {other:?}"),
                };
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        -15.0 * y,
                    ))
                })
            },
        )))
        .expect("call-port scope");
        drop(entry);

        let out = block_on(runtime.scope(ode15s_builtin(
            Value::FunctionHandle("stiff_decay".into()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )))
        .unwrap();

        match out {
            Value::Tensor(t) => {
                assert_eq!(t.cols(), 1);
                let last = t.materialize_f64()[t.rows() - 1];
                assert!(last.is_finite());
                assert!(last > 0.0);
                assert!(last < 0.1);
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn ode15s_accepts_picard_unstable_stiff_step_with_newton() {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _fixtures = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("very_stiff_decay", 572)],
        )
        .unwrap();
        let entry = runtime.enter();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |function, args, _requested_outputs| {
                assert_eq!(function, 572);
                let y = match &args[1] {
                    Value::Num(n) => *n,
                    other => panic!("expected scalar state, got {other:?}"),
                };
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        -1000.0 * y,
                    ))
                })
            },
        )))
        .expect("call-port scope");
        drop(entry);
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.insert("RelTol", Value::Num(1.0e6));
        options.insert("AbsTol", Value::Num(1.0e6));
        options.insert("InitialStep", Value::Num(0.1));
        options.insert("MaxStep", Value::Num(0.1));
        options.insert("MaxSteps", Value::Num(2.0));

        let out = block_on(runtime.scope(ode15s_builtin(
            Value::FunctionHandle("very_stiff_decay".into()),
            Value::Tensor(Tensor::new(vec![0.0, 0.1], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Struct(
                options.finish(),
            )]),
        )))
        .unwrap();

        match out {
            Value::Tensor(t) => {
                assert_eq!(t.cols(), 1);
                let last = t.materialize_f64()[t.rows() - 1];
                assert!(last.is_finite());
                assert!(last > 0.0);
                assert!(last < 0.02);
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn ode15s_accepts_semantic_function_handle_rhs() {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 57);
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |function, args, _requested_outputs| {
                assert_eq!(function, 57);
                let y = match &args[1] {
                    Value::Num(n) => *n,
                    other => panic!("expected scalar state, got {other:?}"),
                };
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        -15.0 * y,
                    ))
                })
            },
        )))
        .expect("call-port scope");

        let out = block_on(ode15s_builtin(
            crate::user_functions::test_support::exact_local_value("ode_stiff_decay", binding),
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap();

        match out {
            Value::Tensor(t) => {
                assert_eq!(t.cols(), 1);
                let last = t.materialize_f64()[t.rows() - 1];
                assert!(last.is_finite());
                assert!(last > 0.0);
                assert!(last < 0.1);
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn ode15s_too_many_inputs_uses_stable_identifier() {
        let err = block_on(ode15s_builtin(
            Value::FunctionHandle("stiff_decay".into()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1.0),
                Value::Num(2.0),
            ]),
        ))
        .expect_err("expected too many inputs error");
        assert_eq!(err.identifier(), ODE15S_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn ode15s_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = ODE15S_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(
            labels,
            vec![
                "y = ode15s(odefun, tspan, y0)",
                "y = ode15s(odefun, tspan, y0, options)",
                "[t, y] = ode15s(odefun, tspan, y0)",
                "[t, y] = ode15s(odefun, tspan, y0, options)",
            ]
        );
    }

    #[test]
    fn ode15s_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = ODE15S_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert_eq!(
            codes,
            vec![
                "RM.ODE15S.INVALID_ARGUMENT",
                "RM.ODE15S.INVALID_INPUT",
                "RM.ODE15S.INTERNAL",
            ]
        );
    }
}
