use super::*;
use futures::executor::block_on;
use runmat_value::{IntValue, Tensor, UnboundedStructValueBuilder};

fn call(args: Vec<Value>) -> BuiltinResult<crate::execution::RootedValue> {
    block_on(odeset_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
}
fn fields(value: &crate::execution::RootedValue) -> &runmat_value::StructValue {
    let Value::Struct(fields) = value.value() else {
        panic!("options structure expected")
    };
    fields
}
fn empty() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

#[test]
fn defaults_aliases_repeated_pairs_and_value_classes_are_preserved() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let defaults = call(vec![]).unwrap();
    assert_eq!(fields(&defaults).fields.len(), OdeOptionField::ALL.len());
    assert!(fields(&defaults)
        .fields
        .values()
        .all(crate::builtins::common::validation::value_is_empty));
    let single = Value::Tensor(Tensor::from_f32(vec![0.125], vec![1, 1]).unwrap());
    let options = call(vec![
        Value::from("relt"),
        Value::Num(0.01),
        Value::from("RELTOL"),
        Value::Num(0.02),
        Value::from("Abs"),
        single.clone(),
        Value::from("Refine"),
        Value::Int(IntValue::U64(3)),
    ])
    .unwrap();
    assert_eq!(fields(&options).fields["RelTol"], Value::Num(0.02));
    assert_eq!(fields(&options).fields["AbsTol"], single);
    assert_eq!(
        fields(&options).fields["Refine"],
        Value::Int(IntValue::U64(3))
    );
    assert!(decode_name("MaxS").is_err());
    assert_eq!(decode_name("MaxStep").unwrap(), OdeOptionField::MaxStep);
    assert_eq!(decode_name("Initial").unwrap(), OdeOptionField::InitialStep);
}

#[test]
fn merging_skips_empty_new_values_but_explicit_empty_resets_and_does_not_mutate_old() {
    let old = call(vec![
        Value::from("RelTol"),
        Value::Num(0.01),
        Value::from("AbsTol"),
        Value::Num(0.001),
    ])
    .unwrap();
    let new = call(vec![
        Value::from("RelTol"),
        empty(),
        Value::from("MaxStep"),
        Value::Num(0.1),
    ])
    .unwrap();
    let merged = call(vec![old.value().clone(), new.value().clone()]).unwrap();
    assert_eq!(fields(&merged).fields["RelTol"], Value::Num(0.01));
    assert_eq!(fields(&merged).fields["AbsTol"], Value::Num(0.001));
    assert_eq!(fields(&merged).fields["MaxStep"], Value::Num(0.1));
    let reset = call(vec![merged.value().clone(), Value::from("RelTol"), empty()]).unwrap();
    assert!(crate::builtins::common::validation::value_is_empty(
        &fields(&reset).fields["RelTol"]
    ));
    assert_eq!(fields(&old).fields["RelTol"], Value::Num(0.01));
    assert!(call(vec![empty(), Value::from("AbsTol"), Value::Num(0.01)]).is_ok());
}

#[test]
fn invalid_grammar_unsupported_options_and_outputs_are_not_silently_accepted() {
    for args in [
        vec![Value::from("RelTol")],
        vec![Value::Num(2.0), Value::Num(1.0)],
        vec![empty(), empty(), empty()],
    ] {
        assert!(call(args).is_err());
    }
    for name in [
        "",
        "MaxS",
        "Events",
        "OutputFcn",
        "Jacobian",
        "Mass",
        "Unknown",
    ] {
        let error = call(vec![Value::from(name), Value::Num(1.0)]).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:odeset:UnknownOption"));
    }
    let mut unsupported = UnboundedStructValueBuilder::new();
    unsupported.insert("Events", empty());
    assert!(call(vec![Value::Struct(unsupported.finish())]).is_err());
    let _scope = crate::output_count::push_output_count(Some(2)).unwrap();
    assert!(call(vec![]).is_err());
}

#[test]
fn options_feed_existing_solver_admission_and_empty_defaults_are_method_owned() {
    let output = call(vec![
        Value::from("RelTol"),
        Value::Num(1e-5),
        Value::from("MaxStep"),
        Value::Num(0.25),
        Value::from("Refine"),
        Value::Num(2.0),
    ])
    .unwrap();
    let parsed = super::super::options::parse("ode45", Some(output.value().clone())).unwrap();
    let options = block_on(super::super::options::admit(
        "ode45",
        parsed,
        super::super::ode45::ODE_COMPATIBILITY_EXTENSIONS,
    ))
    .unwrap();
    assert_eq!(options.rel_tol, 1e-5);
    assert_eq!(options.abs_tol, 1e-6);
    assert_eq!(options.max_step, Some(0.25));
    assert_eq!(options.refine, Some(2));
}

#[test]
fn strict_mode_preserves_empty_defaults_and_rejects_runmat_only_payloads() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    assert!(call(vec![]).is_ok());
    let integer = call(vec![Value::from("Refine"), Value::Int(IntValue::U64(3))]).unwrap_err();
    assert_eq!(
        integer.identifier(),
        Some("RunMat:compatibility:OdesetIntegerOptionExtension")
    );
    let count = call(vec![Value::from("MaxSteps"), Value::Num(20.0)]).unwrap_err();
    assert_eq!(
        count.identifier(),
        Some("RunMat:compatibility:OdesetMaxStepsExtension")
    );
}

#[test]
fn zero_input_no_output_lists_supported_controls() {
    let capture = crate::console::begin_capture().unwrap();
    let _scope = crate::output_count::push_output_count(Some(0)).unwrap();
    call(vec![]).unwrap();
    let text = capture.finish().unwrap();
    assert!(text.contains("RelTol:"));
    assert!(text.contains("Refine:"));
    assert!(text.contains("RunMat extension"));
    assert!(!text.contains("Events:"));
}

#[test]
fn contextual_defaults_use_managed_backing_and_retire_with_the_context() {
    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use std::rc::Rc;
    struct Managed(ManagedHostEffectLedger);
    impl RuntimeParallelService for Managed {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..RuntimeParallelResources::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    let ledger = ManagedHostEffectLedger::new(262_144);
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Managed(ledger.clone()))),
    );
    let args = crate::call::BuiltinArguments::from_borrowed(&[], Some(&runtime)).unwrap();
    let output = block_on(odeset_builtin(Some(runtime.clone()), args)).unwrap();
    assert_eq!(fields(&output).fields.len(), OdeOptionField::ALL.len());
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(output);
    drop(runtime);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}
