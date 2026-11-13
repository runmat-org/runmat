use super::*;
use runmat_value::{IntegerStorage, Tensor};
fn options(fields: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    let mut out = runmat_value::UnboundedStructValueBuilder::new();
    for (name, value) in fields {
        out.insert(name, value);
    }
    Value::Struct(out.finish())
}
fn get(args: Vec<Value>) -> crate::execution::RootedValueSequence {
    query(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    )
    .unwrap()
}
fn only(value: &crate::execution::RootedValueSequence) -> &Value {
    assert_eq!(value.value().len(), 1);
    &value.value().as_slice()[0]
}
fn empty() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap())
}

#[test]
fn names_unset_default_and_stored_false_zero_text_are_distinct() {
    let no_settings = get(vec![empty(), Value::from("Display"), Value::from("final")]);
    assert_eq!(only(&no_settings), &Value::from("final"));
    let result = get(vec![
        options([("Display", Value::from("iter"))]),
        Value::from("dIsP"),
    ]);
    assert_eq!(only(&result), &Value::from("iter"));
    let result = get(vec![
        options([("TolX", empty())]),
        Value::from("StepTolerance"),
        Value::Num(7.0),
    ]);
    assert_eq!(only(&result), &Value::Num(7.0));
    let result = get(vec![options([]), Value::from("MaxIterations")]);
    assert!(matches!(only(&result),Value::Tensor(value)if value.is_empty()));
    for stored in [Value::Num(0.0), Value::Bool(false), Value::from("")] {
        let result = get(vec![
            options([("Display", stored.clone())]),
            Value::from("Display"),
            Value::from("fallback"),
        ]);
        assert_eq!(only(&result), &stored);
    }
    for name in ["Max", "Tol", "not_an_option", ""] {
        assert!(query(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                options([]),
                Value::from(name)
            ])
        )
        .is_err());
    }
    let duplicate = options([
        ("Display", Value::from("iter")),
        ("display", Value::from("off")),
    ]);
    assert!(query(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            duplicate,
            Value::from("Display")
        ])
    )
    .is_err());
}
#[test]
fn explicit_profile_prefixes_preserve_projected_values_and_plain_ambiguity() {
    for (solver, name, prefix) in [
        ("ga", "MaxGenerations", "maxgener"),
        ("ga", "FitnessLimit", "fitnessl"),
        ("particleswarm", "MaxIterations", "maxitera"),
        ("patternsearch", "MaxFunctionEvaluations", "maxfunct"),
        ("patternsearch", "StepTolerance", "stept"),
        ("simulannealbnd", "MaxFunctionEvaluations", "maxfunct"),
        ("simulannealbnd", "MaxIterations", "maxitera"),
    ] {
        assert!(runmat_builtins::OptimSolver::parse(solver).is_some());
        let result = get(vec![
            options([("Solver", Value::from(solver)), (name, Value::Bool(false))]),
            Value::from(prefix),
        ]);
        assert_eq!(only(&result), &Value::Bool(false));
    }
    let display = get(vec![
        options([("Display", Value::from("iter"))]),
        Value::from("disp"),
    ]);
    assert_eq!(only(&display), &Value::from("iter"));
    let plot = get(vec![
        options([("PlotFcn", Value::Bool(false))]),
        Value::from("plotf"),
    ]);
    assert_eq!(only(&plot), &Value::Bool(false));
    let tolerance = get(vec![
        options([
            ("Solver", Value::from("patternsearch")),
            ("FunctionTolerance", Value::Num(7.0)),
        ]),
        Value::from("TolFun"),
    ]);
    assert_eq!(only(&tolerance), &Value::Num(7.0));
    // A modern-looking field is not a profile. The unprofiled global grammar
    // does not infer patternsearch/particleswarm from stored names or values.
    for solver in [None, Some("not_a_solver")] {
        let mut fields = vec![("MaxIterations", Value::Num(37.0))];
        if let Some(solver) = solver {
            fields.push(("Solver", Value::from(solver)));
        }
        assert!(query(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                options(fields),
                Value::from("maxitera")
            ])
        )
        .is_err());
    }
    let duplicate = options([
        ("Solver", Value::from("particleswarm")),
        ("solver", Value::from("patternsearch")),
        ("MaxIterations", Value::Num(37.0)),
    ]);
    assert!(query(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            duplicate,
            Value::from("maxitera")
        ])
    )
    .is_err());
}
#[test]
fn native_integer_and_unreadable_resident_callback_default_keep_identity() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(
        crate::context::physical_test_support::OwnedPhysicalProvider::new(),
    ));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let allocation = provider.allocate_f64(&[1, 1]);
    let handle = allocation
        .handle
        .clone()
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        .unwrap();
    let callback = Value::Closure(runmat_value::Closure::new(
        "opaque_option".into(),
        None,
        runmat_types::FunctionSourceRole::Anonymous,
        None,
        vec![
            Value::GpuTensor(handle.clone()),
            Value::Int(runmat_value::IntValue::U64(u64::MAX)),
        ],
        None,
    ));
    let instance = callback.callable_instance_key();
    let opaque_profile = get(vec![
        options([
            ("Solver", callback.clone()),
            ("OutputFcn", callback.clone()),
        ]),
        Value::from("outp"),
    ]);
    assert_eq!(only(&opaque_profile).callable_instance_key(), instance);
    let result = get(vec![
        options([("OutputFcn", callback.clone())]),
        Value::from("outp"),
    ]);
    assert_eq!(only(&result).callable_instance_key(), instance);
    let result = get(vec![options([]), Value::from("OutputFcn"), callback]);
    assert_eq!(only(&result).callable_instance_key(), instance);
    let Value::Closure(stored) = only(&result) else {
        panic!("closure")
    };
    assert!(
        matches!(&stored.captures[0],Value::GpuTensor(value)if value.view_identity()==handle.view_identity())
    );
    let result = get(vec![
        options([("TolX", Value::GpuTensor(handle.clone()))]),
        Value::from("TolX"),
    ]);
    assert!(
        matches!(only(&result),Value::GpuTensor(value)if value.view_identity()==handle.view_identity())
    );
    assert!(handle.allocation_lease().validate_live().is_ok());
    let payload = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap(),
    );
    let result = get(vec![options([]), Value::from("TolX"), payload]);
    let Value::Tensor(value) = only(&result) else {
        panic!("native tensor")
    };
    assert!(
        matches!(value.numeric_value_at(0),Some(runmat_value::NumericScalar::U64(value))if value==u64::MAX)
    );
}
#[test]
fn original_policy_and_cancellation_precede_query_normalization() {
    use crate::context::{
        host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
        RuntimeParallelService, RuntimeServicePorts,
    };
    use std::rc::Rc;
    struct Managed(ManagedHostEffectLedger);
    impl RuntimeParallelService for Managed {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let s = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(s.free_bytes()),
                epoch: s.epoch,
                ..Default::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    let ledger = ManagedHostEffectLedger::new(1_000_000);
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Managed(ledger.clone()))),
    );
    let before = ledger.snapshot();
    let error = query(
        Some(&runtime),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(
        ledger.snapshot().live_claimed_bytes,
        before.live_claimed_bytes
    );
    let source = runmat_execution::cancellation::CancellationSource::new();
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_cancellation_token(source.token());
    runtime.set_error_namespace("OriginalOptionQuery");
    source.cancel(runmat_execution::CancellationReason::User);
    let error = query(
        Some(&runtime),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .unwrap_err();
    assert_eq!(error.identifier(), Some("OriginalOptionQuery:Cancelled"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
}
