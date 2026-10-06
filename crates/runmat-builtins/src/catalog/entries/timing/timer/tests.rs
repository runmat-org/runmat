use super::constructors::TIMER;
use super::discovery::{TIMERFIND, TIMERFINDALL};
use super::methods::{START, STARTAT};
use super::setters::*;
use super::*;
use crate::REQUIRED_DEFAULT_BINDING;
use runmat_types::{
    BuiltinId, CallRequest, CallableFact, CallableIdentity, CapabilityRequirement, CapabilitySet,
    CellFact, DynamicReason, EffectKind, LiteralContext, LiteralValue, OutputSelection,
    RequestedOutputCount, ValueFact, ValueKindFact,
};

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn timer_effect_declarations_are_sorted_and_unique() {
    let mut entries = Vec::new();
    extend_entries(&mut entries);
    for entry in entries {
        assert!(
            entry
                .contract
                .effects
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "{} has noncanonical effect ordering",
            entry.identity.name
        );
    }
}

#[test]
fn all_runtime_timer_bindings_have_one_catalog_identity() {
    let mut entries = Vec::new();
    extend_entries(&mut entries);
    assert_eq!(entries.len(), 21);
    for (index, entry) in entries.iter().enumerate() {
        assert_eq!(entry.bindings.len(), REQUIRED_DEFAULT_BINDING.len());
        assert!(entries[..index]
            .iter()
            .all(|previous| previous.identity.name != entry.identity.name));
    }
    assert_eq!(TIMER.documentation.examples.len(), 2);
    assert_eq!(TIMERFIND.documentation.examples.len(), 1);
    assert_eq!(TIMERFINDALL.documentation.examples.len(), 1);
}

#[test]
fn timer_inference_owns_handle_fact_and_rejects_late_invalid_property() {
    let constructor = inference::constructor(&request(Vec::new(), Vec::new()), &TIMER);
    assert!(matches!(
        constructor.outputs[0].kind,
        ValueKindFact::Object(_)
    ));
    let invalid = inference::constructor(
        &request(
            vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::unknown(DynamicReason::RuntimeValue),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ],
            vec![
                LiteralValue::String("Period".into()),
                LiteralValue::Number(1.0),
                LiteralValue::String("Running".into()),
            ],
        ),
        &TIMER,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-TIMER-READONLY"));
}

#[test]
fn startat_accepts_optional_delay_other_methods_do_not() {
    assert_eq!(
        STARTAT.contract.compatibility,
        crate::BuiltinCompatibility::Matlab
    );
    assert_eq!(STARTAT.descriptor.signatures[0].inputs.len(), 2);
    assert_eq!(START.descriptor.signatures[0].inputs.len(), 1);
    let receiver = ValueFact::unknown(DynamicReason::RuntimeValue);
    let delay = ValueFact::unknown(DynamicReason::RuntimeValue);
    let call = request(vec![receiver, delay], Vec::new());
    assert!(inference::startat(&call, &STARTAT).diagnostics.is_empty());
    for count in [4, 7] {
        let call = request(
            vec![ValueFact::unknown(DynamicReason::RuntimeValue); count],
            Vec::new(),
        );
        assert!(inference::startat(&call, &STARTAT).diagnostics.is_empty());
    }
    for count in [3, 5, 6, 8] {
        let call = request(
            vec![ValueFact::unknown(DynamicReason::RuntimeValue); count],
            Vec::new(),
        );
        assert!(inference::startat(&call, &STARTAT)
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-TIMER-METHOD-ARITY"));
    }
    assert!(inference::start(&call, &START)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-TIMER-METHOD-ARITY"));
}

fn callback(identity: Option<CallableIdentity>, capabilities: CapabilitySet) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity,
        capabilities,
        parameters: Vec::new(),
        parameters_complete: false,
        outputs: Vec::new(),
        outputs_complete: false,
        variadic_inputs: true,
        variadic_outputs: true,
        captures: Vec::new(),
        captures_complete: true,
    }))
}

#[test]
fn constructor_retains_callback_fact_and_start_propagates_capability() {
    let filesystem = CapabilitySet([CapabilityRequirement::Filesystem].into_iter().collect());
    let constructed = inference::constructor(
        &request(
            vec![
                ValueFact::scalar(ValueKindFact::String),
                callback(None, filesystem.clone()),
            ],
            vec![LiteralValue::String("TimerFcn".into())],
        ),
        &TIMER,
    );
    let ValueKindFact::Object(object) = &constructed.outputs[0].kind else {
        panic!("timer constructor must retain an object fact")
    };
    assert!(matches!(
        object.properties.get("TimerFcn").map(|value| &value.kind),
        Some(ValueKindFact::Callable(_))
    ));
    let started = inference::start(
        &request(vec![constructed.outputs[0].clone()], Vec::new()),
        &START,
    );
    assert!(started
        .capabilities
        .0
        .contains(&CapabilityRequirement::HostRuntime));
    assert!(started
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    assert!(started.effects.0.contains(&EffectKind::Unknown));
    assert!(started.effects.0.contains(&EffectKind::HostCallback));
    let replaced = inference::set_timer_fcn(
        &request(
            vec![constructed.outputs[0].clone(), callback(None, filesystem)],
            Vec::new(),
        ),
        &SET_TIMER_FCN,
    );
    let ValueKindFact::Object(object) = &replaced.outputs[0].kind else {
        panic!("timer setter must retain an object fact")
    };
    assert!(matches!(
        object.properties.get("TimerFcn").map(|value| &value.kind),
        Some(ValueKindFact::Callable(_))
    ));
}

#[test]
fn known_builtin_callback_adds_transitive_effects_and_find_stays_noncallback() {
    let constructed = inference::constructor(
        &request(
            vec![
                ValueFact::scalar(ValueKindFact::String),
                callback(
                    Some(CallableIdentity::Builtin(BuiltinId("imread".into()))),
                    CapabilitySet::default(),
                ),
            ],
            vec![LiteralValue::String("TimerFcn".into())],
        ),
        &TIMER,
    );
    let started = inference::start(
        &request(vec![constructed.outputs[0].clone()], Vec::new()),
        &START,
    );
    assert!(started.effects.0.contains(&EffectKind::FilesystemRead));
    assert!(started
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    let found = inference::find(&request(Vec::new(), Vec::new()), &TIMERFIND);
    assert!(!found.effects.0.contains(&EffectKind::HostCallback));
    assert!(!found.effects.0.contains(&EffectKind::Clock));
}

#[test]
fn lifecycle_callback_cycles_and_cell_extras_remain_bounded_and_dynamic() {
    for identity in ["__runmat_timer_start", "__runmat_timer_wait"] {
        let callback = callback(
            Some(CallableIdentity::Builtin(BuiltinId(identity.into()))),
            CapabilitySet::default(),
        );
        let callback_cell = ValueFact::scalar(ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            elements: vec![callback, ValueFact::scalar(ValueKindFact::String)],
            elements_complete: true,
        }));
        let constructed = inference::constructor(
            &request(
                vec![ValueFact::scalar(ValueKindFact::String), callback_cell],
                vec![LiteralValue::String("TimerFcn".into())],
            ),
            &TIMER,
        );
        let started = inference::start(
            &request(vec![constructed.outputs[0].clone()], Vec::new()),
            &START,
        );
        assert!(started.effects.0.contains(&EffectKind::Unknown));
        assert!(started.effects.0.contains(&EffectKind::HostCallback));
        assert!(started
            .capabilities
            .0
            .contains(&CapabilityRequirement::HostRuntime));
    }
}
