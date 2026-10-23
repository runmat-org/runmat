use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        literals: LiteralContext::new(vec![LiteralValue::Unknown; arguments.len()]),
        arguments,
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn retained_callback_lifecycle_declares_suspension() {
    assert_eq!(
        ON_CLEANUP_CATALOG_ENTRY.contract.async_behavior,
        BuiltinAsyncBehavior::MaySuspend
    );
    let inferred = inference::infer(
        &request(vec![ValueFact::unknown(DynamicReason::RuntimeValue)], 1),
        &ON_CLEANUP_CATALOG_ENTRY,
    );
    assert!(inferred.effects.0.contains(&EffectKind::HostCallback));
    assert!(inferred.effects.0.contains(&EffectKind::MaySuspend));
    assert!(inferred.effects.0.contains(&EffectKind::Unknown));
}

#[test]
fn callable_unknown_invalid_and_discarded_output() {
    let mut call = request(vec![ValueFact::unknown(DynamicReason::RuntimeValue)], 1);
    let unknown = inference::infer(&call, &ON_CLEANUP_CATALOG_ENTRY);
    assert!(unknown.diagnostics.is_empty());
    assert_eq!(unknown.outputs.len(), 1);
    assert_eq!(unknown.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(unknown.outputs[0].storage, StorageFact::Opaque);
    assert!(matches!(
        &unknown.outputs[0].kind,
        ValueKindFact::Object(ObjectFact {
            runtime_class: Some(class),
            handle_semantics: Some(true),
            ..
        }) if class.display_name() == "onCleanup"
    ));
    call.arguments[0] = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: CapabilitySet::default(),
        parameters: vec![],
        parameters_complete: true,
        outputs: vec![],
        outputs_complete: true,
        variadic_inputs: false,
        variadic_outputs: false,
        captures: vec![],
        captures_complete: true,
    }));
    assert!(inference::infer(&call, &ON_CLEANUP_CATALOG_ENTRY)
        .diagnostics
        .is_empty());

    call.outputs = OutputSelection::new(RequestedOutputCount::Exactly(0));
    assert!(inference::infer(&call, &ON_CLEANUP_CATALOG_ENTRY)
        .outputs
        .is_empty());
    call.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
    assert!(!inference::infer(&call, &ON_CLEANUP_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    call.outputs = OutputSelection::new(RequestedOutputCount::Exactly(1));
    call.arguments[0] = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    assert!(!inference::infer(&call, &ON_CLEANUP_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    call.arguments.clear();
    assert!(!inference::infer(&call, &ON_CLEANUP_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

#[test]
fn lifecycle_operations_have_canonical_executable_contracts() {
    for entry in [&CANCEL_CATALOG_ENTRY, &ON_CLEANUP_DELETE_CATALOG_ENTRY] {
        assert_eq!(entry.bindings.len(), 1);
        assert_eq!(entry.bindings[0].variant, "default");
        assert_eq!(
            entry.bindings[0].availability,
            BuiltinBindingAvailability::Required
        );
        assert_eq!(
            entry.link.execution_stack,
            ExecutionStackRequirement::Process
        );
        assert_eq!(
            entry.placement.portability,
            BuiltinPortability::NativeAndWasm
        );
        assert_eq!(
            entry.placement.preparation,
            BuiltinOperandPreparation::RuntimeOwned
        );
        assert_eq!(
            entry.placement.accelerator,
            BuiltinAcceleratorPolicy::Forbidden
        );
        assert_eq!(entry.placement.fusion, BuiltinFusionPolicy::Boundary);
        assert_eq!(entry.placement.residency, BuiltinResidencyPolicy::Host);
        assert_eq!(
            entry.placement.distributed,
            BuiltinDistributedPolicy::Unsupported
        );
        assert_eq!(entry.contract.purity, BuiltinPurity::Impure);
        assert!(entry.suppress_auto_output);
        assert!(ENTRIES
            .iter()
            .any(|candidate| candidate.identity.name == entry.identity.name));
        let inferred = crate::catalog::inference::infer_catalog_call(
            entry,
            &request(vec![ValueFact::unknown(DynamicReason::RuntimeValue)], 1),
        );
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(
            inferred.outputs[0],
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }))
        );
        assert!(inferred.effects.0.contains(&EffectKind::HostCallback));
        assert!(inferred.effects.0.contains(&EffectKind::MayThrow));
        for outputs in [0, 2] {
            let result = crate::catalog::inference::infer_catalog_call(
                entry,
                &request(
                    vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
                    outputs,
                ),
            );
            if outputs == 0 {
                assert!(result.outputs.is_empty());
                assert!(result.diagnostics.is_empty());
            } else {
                assert!(!result.diagnostics.is_empty());
            }
        }
        assert!(
            !crate::catalog::inference::infer_catalog_call(entry, &request(vec![], 1))
                .diagnostics
                .is_empty()
        );
        let integer = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }));
        assert!(
            !crate::catalog::inference::infer_catalog_call(entry, &request(vec![integer], 1))
                .diagnostics
                .is_empty()
        );
    }
    assert_eq!(
        CANCEL_DESCRIPTOR.completion_policy,
        BuiltinCompletionPolicy::Public
    );
    assert_eq!(
        ON_CLEANUP_DELETE_DESCRIPTOR.completion_policy,
        BuiltinCompletionPolicy::HiddenInternal
    );
    assert_eq!(
        CANCEL_CATALOG_ENTRY.contract.async_behavior,
        BuiltinAsyncBehavior::NeverSuspends
    );
    assert!(!CANCEL_CATALOG_ENTRY.documentation.examples.is_empty());
    assert!(CANCEL_CATALOG_ENTRY
        .documentation
        .example_exemption
        .is_none());
    assert!(ON_CLEANUP_DELETE_CATALOG_ENTRY
        .documentation
        .example_exemption
        .is_some());
    assert!(ON_CLEANUP_DELETE_CATALOG_ENTRY
        .contract
        .effect_set()
        .0
        .contains(&EffectKind::Unknown));
    assert_eq!(
        ON_CLEANUP_DELETE_CATALOG_ENTRY.contract.async_behavior,
        BuiltinAsyncBehavior::MaySuspend
    );
}

#[test]
fn cancel_facts_accept_typed_execution_and_wrapped_task_identities() {
    let output = ValueSequenceFact::dynamic();
    for kind in [
        ValueKindFact::Execution(ExecutionFact::Future {
            output: output.clone(),
            state: FutureStateFact::Unknown,
        }),
        ValueKindFact::Execution(ExecutionFact::Task {
            output: output.clone(),
            spawn_safety: SpawnSafetyFact::SpawnSafe,
        }),
        ValueKindFact::Execution(ExecutionFact::Job { output }),
    ] {
        let argument = ValueFact::scalar(kind);
        assert!(crate::catalog::inference::infer_catalog_call(
            &CANCEL_CATALOG_ENTRY,
            &request(vec![argument.clone()], 1)
        )
        .diagnostics
        .is_empty());
        assert!(!crate::catalog::inference::infer_catalog_call(
            &ON_CLEANUP_DELETE_CATALOG_ENTRY,
            &request(vec![argument], 1)
        )
        .diagnostics
        .is_empty());
    }
    for (class, handle, shape) in [
        ("onCleanup", Some(true), ShapeFact::Scalar),
        (
            "parallel.FevalFuture",
            Some(false),
            ShapeFact::from(vec![Some(1), Some(3)]),
        ),
        ("parallel.FevalOnAllFuture", Some(false), ShapeFact::Scalar),
    ] {
        let argument = ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(StaticClassIdentity::new(class).owned()),
                properties: Default::default(),
                properties_complete: false,
                handle_semantics: handle,
            }),
            shape,
            StorageFact::Opaque,
        );
        assert!(crate::catalog::inference::infer_catalog_call(
            &CANCEL_CATALOG_ENTRY,
            &request(vec![argument], 1)
        )
        .diagnostics
        .is_empty());
    }
    let pool = ValueFact::scalar(ValueKindFact::Execution(ExecutionFact::Pool));
    assert!(!crate::catalog::inference::infer_catalog_call(
        &CANCEL_CATALOG_ENTRY,
        &request(vec![pool], 1)
    )
    .diagnostics
    .is_empty());
}
