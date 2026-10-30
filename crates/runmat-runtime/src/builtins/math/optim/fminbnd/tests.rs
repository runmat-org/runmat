use super::*;
use crate::builtins::math::optim::brent::brent_min_tolerance;
use futures::executor::block_on;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, FMINBND_DESCRIPTOR,
};
use runmat_value::IntegerStorage;
use runmat_value::Value as V;

fn objective_result_type(
    _args: &[runmat_builtins::Type],
    _context: &runmat_builtins::ResolveContext,
) -> runmat_builtins::Type {
    runmat_builtins::Type::Num
}

#[test]
fn registered_fminbnd_preserves_invocation_cancellation_before_solver_effects() {
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::rc::Rc;

    let source = CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    source.cancel(CancellationReason::User);
    let error = block_on(fminbnd_builtin_registered(
        Some(runtime),
        Value::FunctionHandle("sin".into()),
        Value::Num(0.0),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("cancelled invocation must not call objective");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(matches!(
        error
            .source
            .as_deref()
            .and_then(|source| source.downcast_ref::<crate::execution::ExecutionServiceError>()),
        Some(crate::execution::ExecutionServiceError::Cancelled)
    ));
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
}

#[test]
fn excessive_requested_outputs_fail_before_callback_evaluation() {
    let _outputs = crate::output_count::push_output_count(Some(5)).expect("output-count scope");
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let observed = calls.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
        std::sync::Arc::new(move |_function, _args, _requested_outputs| {
            observed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Box::pin(async {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(0.0))
            })
        }),
    ))
    .expect("call-port scope");
    let error = block_on(fminbnd_builtin(
        Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
            runmat_value::BoundFunctionHandleData {
                name: "unobserved".into(),
                function: 7_710,
                authority: runmat_value::BoundFunctionAuthority::unavailable(7_710),
                source_role: runmat_types::FunctionSourceRole::Unknown,
                source_file: None,
            },
        )),
        Value::Num(0.0),
        Value::Num(5.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("five outputs are not part of the public contract");
    assert_eq!(error.identifier(), Some("RunMat:fminbnd:TooManyOutputs"));
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
}

const FMINBND_HELPER_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fx",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Objective scalar value.",
}];

const FMINBND_HELPER_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar objective input.",
}];

const FMINBND_HELPER_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "fx = __fminbnd_helper(x)",
    inputs: &FMINBND_HELPER_INPUTS,
    outputs: &FMINBND_HELPER_OUTPUT,
}];

const FMINBND_HELPER_ERRORS: [BuiltinErrorDescriptor; 0] = [];

pub const FMINBND_TEST_HELPER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FMINBND_HELPER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &FMINBND_HELPER_ERRORS,
};

fn run_default(handle: &str, lo: f64, hi: f64) -> Value {
    block_on(fminbnd_builtin(
        V::FunctionHandle(handle.into()),
        V::Num(lo),
        V::Num(hi),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("fminbnd")
}

fn run_with(handle: &str, lo: f64, hi: f64, extra: Vec<Value>) -> Value {
    block_on(fminbnd_builtin(
        V::FunctionHandle(handle.into()),
        V::Num(lo),
        V::Num(hi),
        crate::builtins::common::test_support::builtin_arguments(extra),
    ))
    .expect("fminbnd")
}

#[test]
fn fminbnd_test_helper_descriptor_is_attached_shape() {
    assert_eq!(
        FMINBND_TEST_HELPER_DESCRIPTOR.signatures[0].label,
        "fx = __fminbnd_helper(x)"
    );
}

#[runtime_builtin(
    name = "__fminbnd_quad_minus_two",
    type_resolver(crate::builtins::math::optim::fminbnd::tests::objective_result_type),
    descriptor(crate::builtins::math::optim::fminbnd::tests::FMINBND_TEST_HELPER_DESCRIPTOR),
    builtin_path = "crate::builtins::math::optim::fminbnd::tests"
)]
async fn quad_minus_two(x: Value) -> crate::BuiltinResult<Value> {
    let x = scalar_bound("x", x).await?;
    let diff = x - 2.0;
    Ok(Value::Num(diff * diff))
}

#[runtime_builtin(
    name = "__fminbnd_quad_minus_three",
    type_resolver(crate::builtins::math::optim::fminbnd::tests::objective_result_type),
    descriptor(crate::builtins::math::optim::fminbnd::tests::FMINBND_TEST_HELPER_DESCRIPTOR),
    builtin_path = "crate::builtins::math::optim::fminbnd::tests"
)]
async fn quad_minus_three(x: Value) -> crate::BuiltinResult<Value> {
    let x = scalar_bound("x", x).await?;
    let diff = x - 3.0;
    Ok(Value::Num(diff * diff))
}

#[runtime_builtin(
    name = "__fminbnd_multi_modal",
    type_resolver(crate::builtins::math::optim::fminbnd::tests::objective_result_type),
    descriptor(crate::builtins::math::optim::fminbnd::tests::FMINBND_TEST_HELPER_DESCRIPTOR),
    builtin_path = "crate::builtins::math::optim::fminbnd::tests"
)]
async fn multi_modal(x: Value) -> crate::BuiltinResult<Value> {
    // 1 + sin(3x) on [0, 2π] — local minima near x ≈ π/2 + 2π/3 (etc.).
    let x = scalar_bound("x", x).await?;
    Ok(Value::Num(1.0 + (3.0 * x).sin()))
}

#[test]
fn locates_smooth_quadratic_minimum() {
    let result = run_default("__fminbnd_quad_minus_two", 0.0, 5.0);
    match result {
        V::Num(x) => assert!((x - 2.0).abs() < 1.0e-3, "x = {x}"),
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn locates_quadratic_minimum_offset_three() {
    let result = run_default("__fminbnd_quad_minus_three", 0.0, 5.0);
    match result {
        V::Num(x) => assert!((x - 3.0).abs() < 1.0e-3, "x = {x}"),
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn locates_cosine_minimum_at_right_endpoint() {
    // cos(x) is monotonically decreasing on [0, π]; minimum is at x = π.
    let result = run_default("cos", 0.0, std::f64::consts::PI);
    match result {
        V::Num(x) => assert!((x - std::f64::consts::PI).abs() < 1.0e-3, "x = {x}"),
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn returns_lone_endpoint_when_bounds_collapse() {
    let result = run_default("__fminbnd_quad_minus_two", 1.5, 1.5);
    match result {
        V::Num(x) => assert!((x - 1.5).abs() < 1.0e-12, "x = {x}"),
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn reports_inconsistent_reversed_bounds() {
    let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let result = block_on(fminbnd_builtin(
        V::FunctionHandle("__fminbnd_quad_minus_two".into()),
        V::Num(5.0),
        V::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("fminbnd");
    match result {
        V::OutputList(outputs) => {
            assert_eq!(outputs.len(), 4);
            assert!(matches!(&outputs[0], V::Tensor(t) if t.materialize_f64().is_empty()));
            assert!(matches!(&outputs[1], V::Tensor(t) if t.materialize_f64().is_empty()));
            assert!(matches!(&outputs[2], V::Num(flag) if *flag == -2.0));
            match &outputs[3] {
                V::Struct(s) => {
                    assert!(matches!(s.fields.get("iterations"), Some(V::Num(0.0))));
                    assert!(matches!(s.fields.get("funcCount"), Some(V::Num(0.0))));
                    match s.fields.get("message") {
                        Some(V::String(runmat_value::StringElement::Text(text))) => {
                            assert!(text.contains("bounds"))
                        }
                        other => panic!("unexpected message field {other:?}"),
                    }
                }
                other => panic!("unexpected output struct {other:?}"),
            }
        }
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn tolerance_is_additive_not_scaled_by_x() {
    let params = BrentParams {
        tol_x: 1.0e-4,
        max_iter: 500,
        max_fun_evals: 500,
    };
    let small = brent_min_tolerance(2.0, params);
    let large = brent_min_tolerance(1.0e9, params);
    assert!(small > params.tol_x);
    assert!(
        large < params.tol_x * 1.0e9,
        "large-scale tolerance was {large}"
    );
}

#[test]
fn finds_local_minimum_in_multi_modal_function() {
    // 1 + sin(3x) on [1.5, 3.5] has its local minimum near x = π/2 ≈ 1.571 (sin(3π/2) = -1).
    let result = run_default("__fminbnd_multi_modal", 1.5, 3.5);
    match result {
        V::Num(x) => {
            let target = std::f64::consts::PI / 2.0;
            assert!((x - target).abs() < 5.0e-3, "x = {x}, target = {target}");
        }
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn options_struct_overrides_default_tolerance() {
    let mut opts = runmat_value::UnboundedStructValueBuilder::new();
    opts.insert("TolX", Value::Num(1.0e-12));
    let result = run_with(
        "__fminbnd_quad_minus_two",
        0.0,
        5.0,
        vec![Value::Struct(opts.finish())],
    );
    match result {
        V::Num(x) => assert!((x - 2.0).abs() < 1.0e-6, "x = {x}"),
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn logical_numeric_options_are_not_accepted_as_floating_controls() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    for (field, value) in [
        ("TolX", Value::Bool(true)),
        (
            "MaxIter",
            Value::LogicalArray(LogicalArray {
                data: vec![1].into(),
                shape: vec![1, 1].into(),
            }),
        ),
    ] {
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.insert(field, value);
        let options = options.finish();
        let error = FminbndOptions::from_struct(Some(&options), None)
            .expect_err("logical numeric option must reject");
        assert_eq!(error.identifier(), Some("RunMat:fminbnd:InvalidArgument"));
    }
}

#[test]
fn bounds_and_options_read_typed_integer_tensor_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let lower = Tensor::new_integer(IntegerStorage::I16(vec![0]), vec![1, 1]).expect("lower bound");
    let upper = Tensor::new_integer(IntegerStorage::U16(vec![5]), vec![1, 1]).expect("upper bound");
    let max_iter =
        Tensor::new_integer(IntegerStorage::U16(vec![1000]), vec![1, 1]).expect("MaxIter");

    let mut opts = runmat_value::UnboundedStructValueBuilder::new();
    opts.insert("MaxIter", Value::Tensor(max_iter));
    let result = block_on(fminbnd_builtin(
        V::FunctionHandle("__fminbnd_quad_minus_two".into()),
        V::Tensor(lower),
        V::Tensor(upper),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Struct(
            opts.finish(),
        )]),
    ))
    .expect("fminbnd");
    match result {
        V::Num(x) => assert!((x - 2.0).abs() < 1.0e-3, "x = {x}"),
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn options_reject_nonpositive_typed_integer_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let max_iter = Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("MaxIter");
    let mut opts = runmat_value::UnboundedStructValueBuilder::new();
    opts.insert("MaxIter", Value::Tensor(max_iter));
    let opts = opts.finish();

    assert!(FminbndOptions::from_struct(Some(&opts), None).is_err());
}

#[test]
fn options_reject_unrepresentable_double_integer_boundary() {
    let boundary = if usize::BITS == 64 {
        usize::MAX as f64
    } else {
        (usize::MAX as f64) + 1.0
    };
    let mut opts = runmat_value::UnboundedStructValueBuilder::new();
    opts.insert("MaxIter", Value::Num(boundary));
    let opts = opts.finish();

    assert!(FminbndOptions::from_struct(Some(&opts), None).is_err());
}

#[test]
fn max_fun_evals_default_is_independent_of_max_iter() {
    let mut opts = runmat_value::UnboundedStructValueBuilder::new();
    opts.insert("MaxIter", Value::Num(1000.0));
    let opts = opts.finish();
    let parsed = FminbndOptions::from_struct(Some(&opts), None).unwrap();
    assert_eq!(parsed.max_iter, 1000);
    assert_eq!(parsed.max_fun_evals, DEFAULT_MAX_FUN_EVALS);
}

#[test]
fn rejects_nonfinite_bounds() {
    let err = block_on(fminbnd_builtin(
        V::FunctionHandle("__fminbnd_quad_minus_two".into()),
        V::Num(f64::NAN),
        V::Num(5.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(err.message().to_ascii_lowercase().contains("finite"));
}

#[test]
fn rejects_invalid_options_type() {
    let err = block_on(fminbnd_builtin(
        V::FunctionHandle("__fminbnd_quad_minus_two".into()),
        V::Num(0.0),
        V::Num(5.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .unwrap_err();
    assert!(err.message().to_ascii_lowercase().contains("options"));
}

#[test]
fn rejects_nonpositive_tol_x() {
    let mut opts = runmat_value::UnboundedStructValueBuilder::new();
    opts.insert("TolX", Value::Num(0.0));
    let err = block_on(fminbnd_builtin(
        V::FunctionHandle("__fminbnd_quad_minus_two".into()),
        V::Num(0.0),
        V::Num(5.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Struct(
            opts.finish(),
        )]),
    ))
    .unwrap_err();
    assert!(err.message().to_lowercase().contains("tolx"));
}

#[test]
fn rejects_unknown_display_value() {
    let mut opts = runmat_value::UnboundedStructValueBuilder::new();
    opts.insert("Display", Value::from("loud"));
    let err = block_on(fminbnd_builtin(
        V::FunctionHandle("__fminbnd_quad_minus_two".into()),
        V::Num(0.0),
        V::Num(5.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Struct(
            opts.finish(),
        )]),
    ))
    .unwrap_err();
    assert!(err.message().to_lowercase().contains("display"));
}

#[test]
fn multi_output_two_returns_x_and_fval() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let result = block_on(fminbnd_builtin(
        V::FunctionHandle("__fminbnd_quad_minus_two".into()),
        V::Num(0.0),
        V::Num(5.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("fminbnd");
    match result {
        V::OutputList(outputs) => {
            assert_eq!(outputs.len(), 2);
            match (&outputs[0], &outputs[1]) {
                (V::Num(x), V::Num(fval)) => {
                    assert!((x - 2.0).abs() < 1.0e-3);
                    assert!(fval.abs() < 1.0e-5);
                }
                other => panic!("unexpected outputs {other:?}"),
            }
        }
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn multi_output_three_includes_exitflag() {
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let result = block_on(fminbnd_builtin(
        V::FunctionHandle("__fminbnd_quad_minus_two".into()),
        V::Num(0.0),
        V::Num(5.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("fminbnd");
    match result {
        V::OutputList(outputs) => {
            assert_eq!(outputs.len(), 3);
            match &outputs[2] {
                V::Num(flag) => assert!((*flag - 1.0).abs() < 1.0e-12),
                other => panic!("unexpected exitflag {other:?}"),
            }
        }
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn multi_output_four_includes_output_struct() {
    let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let result = block_on(fminbnd_builtin(
        V::FunctionHandle("__fminbnd_quad_minus_two".into()),
        V::Num(0.0),
        V::Num(5.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("fminbnd");
    match result {
        V::OutputList(outputs) => {
            assert_eq!(outputs.len(), 4);
            match &outputs[3] {
                V::Struct(s) => {
                    assert!(matches!(s.fields.get("iterations"), Some(V::Num(_))));
                    assert!(matches!(s.fields.get("funcCount"), Some(V::Num(_))));
                    match s.fields.get("algorithm") {
                        Some(V::String(runmat_value::StringElement::Text(text))) => {
                            assert!(text.contains("golden"))
                        }
                        other => panic!("unexpected algorithm field {other:?}"),
                    }
                    assert!(s.fields.get("message").is_some());
                }
                other => panic!("unexpected output struct {other:?}"),
            }
        }
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn reports_zero_exitflag_when_max_iter_exhausted() {
    let mut opts = runmat_value::UnboundedStructValueBuilder::new();
    opts.insert("MaxIter", Value::Num(1.0));
    opts.insert("MaxFunEvals", Value::Num(2.0));
    opts.insert("Display", Value::from("off"));
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let result = block_on(fminbnd_builtin(
        V::FunctionHandle("__fminbnd_quad_minus_two".into()),
        V::Num(0.0),
        V::Num(5.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Struct(
            opts.finish(),
        )]),
    ))
    .expect("fminbnd");
    match result {
        V::OutputList(outputs) => match &outputs[2] {
            V::Num(flag) => assert_eq!(*flag, 0.0),
            other => panic!("unexpected exitflag {other:?}"),
        },
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn fminbnd_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = FMINBND_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    assert_eq!(
        labels,
        vec![
            "fminbnd(fun, x1, x2)",
            "fminbnd(fun, x1, x2, options)",
            "x = fminbnd(fun, x1, x2)",
            "x = fminbnd(fun, x1, x2, options)",
            "[x, fval] = fminbnd(fun, x1, x2)",
            "[x, fval] = fminbnd(fun, x1, x2, options)",
            "[x, fval, exitflag] = fminbnd(fun, x1, x2)",
            "[x, fval, exitflag] = fminbnd(fun, x1, x2, options)",
            "[x, fval, exitflag, output] = fminbnd(fun, x1, x2)",
            "[x, fval, exitflag, output] = fminbnd(fun, x1, x2, options)",
        ]
    );

    let codes: Vec<&str> = FMINBND_DESCRIPTOR
        .errors
        .iter()
        .map(|error| error.code)
        .collect();
    assert_eq!(
        codes,
        vec![
            "RM.FMINBND.INVALID_ARGUMENT",
            "RM.FMINBND.INVALID_INPUT",
            "RM.FMINBND.TOO_MANY_OUTPUTS",
        ]
    );
}

#[test]
fn fminbnd_too_many_args_uses_stable_identifier() {
    let err = block_on(fminbnd_builtin(
        V::FunctionHandle("__fminbnd_quad_minus_two".into()),
        V::Num(0.0),
        V::Num(5.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Struct(StructValue::new()),
            Value::Struct(StructValue::new()),
        ]),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:fminbnd:InvalidArgument"));
}
