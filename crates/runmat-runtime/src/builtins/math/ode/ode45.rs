//! MATLAB-compatible `ode45` builtin.

#[cfg(test)]
use runmat_builtins::ODE45_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, ODE45_ERROR_INTERNAL, ODE45_ERROR_INVALID_ARGUMENT,
    ODE45_ERROR_INVALID_INPUT, ODE45_INTEGER_CALLBACK_EXTENSION as INTEGER_CALLBACK_EXTENSION,
    ODE45_INTEGER_OPTION_EXTENSION as INTEGER_OPTION_EXTENSION,
    ODE45_INTEGER_TSPAN_EXTENSION as INTEGER_TSPAN_EXTENSION,
    ODE45_INTEGER_Y0_EXTENSION as INTEGER_Y0_EXTENSION,
    ODE45_LOGICAL_NUMERIC_EXTENSION as LOGICAL_NUMERIC_EXTENSION,
    ODE45_RESIDENT_INPUT_EXTENSION as RESIDENT_INPUT_EXTENSION,
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

const NAME: &str = "ode45";

pub(crate) const ODE_COMPATIBILITY_EXTENSIONS: OdeCompatibilityExtensions =
    OdeCompatibilityExtensions {
        integer_tspan: &INTEGER_TSPAN_EXTENSION,
        integer_y0: &INTEGER_Y0_EXTENSION,
        integer_option: &INTEGER_OPTION_EXTENSION,
        integer_callback: &INTEGER_CALLBACK_EXTENSION,
        logical_numeric: &LOGICAL_NUMERIC_EXTENSION,
        resident_input: &RESIDENT_INPUT_EXTENSION,
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
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::ode::ode45"
)]
async fn ode45_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    function: Value,
    tspan: Value,
    y0: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ode45_in_runtime(runtime.as_ref(), function, tspan, y0, rest).await
}

#[cfg(test)]
async fn ode45_builtin(
    function: Value,
    tspan: Value,
    y0: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ode45_in_runtime(None, function, tspan, y0, rest).await
}

async fn ode45_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    function: Value,
    tspan: Value,
    y0: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::resource_ingress::require_unbounded(NAME, runtime)?;
    check_cancellation(NAME, runtime)?;
    if rest.len() > 1 {
        return Err(ode45_error_with_detail(
            &ODE45_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        ));
    }
    let options = options::parse(NAME, rest.into_iter().next())
        .map_err(|err| ode45_map_error(err, &ODE45_ERROR_INVALID_ARGUMENT))?;
    let opts = options::admit(NAME, options, ODE_COMPATIBILITY_EXTENSIONS)
        .await
        .map_err(|err| ode45_map_error(err, &ODE45_ERROR_INVALID_ARGUMENT))?;
    check_cancellation(NAME, runtime)?;
    let input = parse_ode_input(NAME, tspan, y0, ODE_COMPATIBILITY_EXTENSIONS, runtime)
        .await
        .map_err(|err| ode45_map_error(err, &ODE45_ERROR_INVALID_INPUT))?;
    let result = solve_ode(NAME, OdeMethod::Ode45, &function, &input, &opts)
        .await
        .map_err(|err| ode45_map_error(err, &ODE45_ERROR_INVALID_INPUT))?;
    check_cancellation(NAME, runtime)?;
    build_ode_output(NAME, result).map_err(|err| ode45_map_error(err, &ODE45_ERROR_INTERNAL))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use runmat_value::Tensor;
    use runmat_value::{IntValue, IntegerStorage};
    use std::rc::Rc;
    use std::sync::Arc;

    #[test]
    fn cancelled_invocation_stops_before_derivative_or_provider_work() {
        let source = CancellationSource::new();
        let runtime = crate::context::RuntimeContext::with_cancellation(
            Rc::new(crate::execution::RuntimeExecutionService::new()),
            source.token(),
        );
        source.cancel(CancellationReason::User);
        let error = block_on(ode45_in_runtime(
            Some(&runtime),
            Value::FunctionHandle("unused".into()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("cancelled invocation cannot reach derivative");
        assert!(error
            .identifier()
            .is_some_and(|identifier| identifier.ends_with(":Cancelled")));
    }

    #[test]
    fn ode45_scalar_decay_returns_reasonable_final_value() {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _fixtures = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("decay", 901)],
        )
        .unwrap();
        let entry = runtime.enter();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |function, args, _requested_outputs| {
                assert_eq!(function, 901);
                let y = match &args[1] {
                    Value::Num(n) => *n,
                    other => panic!("expected scalar state, got {other:?}"),
                };
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        -y,
                    ))
                })
            },
        )))
        .expect("call-port scope");
        drop(entry);

        let out = block_on(runtime.scope(ode45_builtin(
            Value::FunctionHandle("decay".into()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )))
        .unwrap();

        match out {
            Value::Tensor(t) => {
                assert_eq!(t.cols(), 1);
                let last = t.materialize_f64()[t.rows() - 1];
                assert!((last - (-1.0_f64).exp()).abs() < 5.0e-3);
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn ode45_strict_mode_rejects_integer_tspan_before_rhs() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let tspan = Tensor::new_integer(IntegerStorage::U16(vec![0, 1]), vec![1, 2]).unwrap();

        let error = block_on(ode45_builtin(
            Value::FunctionHandle("unused".into()),
            Value::Tensor(tspan),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("integer tspan is a RunMat-only extension");

        assert_eq!(error.identifier(), INTEGER_TSPAN_EXTENSION.error_identifier);
    }

    #[test]
    fn ode45_runmat_mode_rejects_wide_integer_tspan() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let tspan =
            Tensor::new_integer(IntegerStorage::U64(vec![0, (1_u64 << 53) + 1]), vec![1, 2])
                .unwrap();

        let error = block_on(ode45_builtin(
            Value::FunctionHandle("unused".into()),
            Value::Tensor(tspan),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("wide integer tspan cannot cross exactly");

        assert!(error.message().contains("exactly representable"));
    }

    #[test]
    fn ode45_strict_mode_rejects_integer_derivative_result() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _fixtures = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("integer_rhs", 903)],
        )
        .unwrap();
        let entry = runtime.enter();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, _args, _requested_outputs| {
                assert_eq!(function, 903);
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Int(
                        IntValue::I32(-1),
                    ))
                })
            },
        )))
        .expect("call-port scope");
        drop(entry);

        let error = block_on(runtime.scope(ode45_builtin(
            Value::FunctionHandle("integer_rhs".into()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )))
        .expect_err("integer derivative is a RunMat-only extension");

        assert_eq!(
            error.identifier(),
            INTEGER_CALLBACK_EXTENSION.error_identifier
        );
    }

    #[test]
    fn ode45_automatic_resident_input_gathers_but_explicit_input_is_gated() {
        test_support::with_test_provider(|provider| {
            let runtime = crate::user_functions::test_support::exact_fixture_runtime();
            let _entry = runtime.enter();
            let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 905);
            let _bound =
                crate::user_functions::test_support::install_bound_forwarder(binding.clone());
            let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
                Arc::new(|_function, _args, _requested_outputs| {
                    Box::pin(async move {
                        crate::user_functions::test_support::rooted_single_value_sequence(
                            Value::Num(-1.0),
                        )
                    })
                }),
            ))
            .expect("call-port scope");
            let times = [0.0, 0.1];
            let shape = [1, 2];
            let automatic = provider
                .upload(&HostTensorView {
                    data: &times,
                    shape: &shape,
                })
                .expect("automatic upload");
            let automatic = automatic
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .expect("admit provenance view");
            let result = block_on(ode45_builtin(
                crate::user_functions::test_support::exact_local_value("constant_rhs", binding),
                Value::GpuTensor(automatic),
                Value::Num(1.0),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect("automatic resident tspan gathers");
            assert!(matches!(result, Value::Tensor(_)));

            let explicit = provider
                .upload(&HostTensorView {
                    data: &times,
                    shape: &shape,
                })
                .expect("explicit upload");
            let explicit = explicit
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("admit provenance view");
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(ode45_builtin(
                Value::FunctionHandle("unused".into()),
                Value::GpuTensor(explicit),
                Value::Num(1.0),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect_err("explicit resident tspan is gated before fallback");
            assert_eq!(
                error.identifier(),
                RESIDENT_INPUT_EXTENSION.error_identifier
            );
        });
    }

    #[test]
    fn ode45_rejects_nan_rhs() {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _fixtures = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("nan_rhs", 904)],
        )
        .unwrap();
        let entry = runtime.enter();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |function, _args, _requested_outputs| {
                assert_eq!(function, 904);
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        f64::NAN,
                    ))
                })
            },
        )))
        .expect("call-port scope");
        drop(entry);

        let err = block_on(runtime.scope(ode45_builtin(
            Value::FunctionHandle("nan_rhs".into()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )))
        .expect_err("ode45 should reject NaN derivative values");

        assert!(err.to_string().contains("function value must be finite"));
    }

    #[test]
    fn ode45_accepts_external_function_handle_rhs() {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _fixtures = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("pkg.decay", 56)],
        )
        .unwrap();
        let entry = runtime.enter();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |function, args, _requested_outputs| {
                assert_eq!(function, 56);
                let y = match &args[1] {
                    Value::Num(n) => *n,
                    other => panic!("expected scalar state, got {other:?}"),
                };
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        -y,
                    ))
                })
            },
        )))
        .expect("call-port scope");
        drop(entry);

        let out = block_on(runtime.scope(ode45_builtin(
            Value::ExternalFunctionHandle("pkg.decay".into()),
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
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1.0),
                Value::Num(2.0),
            ]),
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
