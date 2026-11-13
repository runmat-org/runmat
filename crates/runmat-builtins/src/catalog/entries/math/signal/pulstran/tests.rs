use super::*;
use runmat_types::*;

fn numeric(class: NumericClass, domain: NumericDomain, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        shape,
        StorageFact::Dense,
    )
}

fn call(args: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments: args,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &PULSTRAN_CATALOG_ENTRY,
    )
}

#[test]
fn known_times_preserve_nd_shape_and_publish_double_with_unknown_residency() {
    let shape = ShapeFact::from(vec![Some(2), Some(0), Some(3)]);
    let out = call(
        vec![
            numeric(NumericClass::Single, NumericDomain::Real, shape.clone()),
            numeric(
                NumericClass::Double,
                NumericDomain::Real,
                ShapeFact::from(vec![Some(1), Some(2)]),
            ),
            ValueFact::scalar(ValueKindFact::String),
        ],
        vec![LiteralValue::Unknown; 3],
        1,
    );
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    assert_eq!(out.outputs[0].shape, shape);
    assert_eq!(
        out.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(out.outputs[0].residency, ResidencyFact::Unknown);
}

#[test]
fn invalid_delay_prototype_and_rate_do_not_mint_output_facts() {
    let t = numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar);
    let delay = numeric(
        NumericClass::Double,
        NumericDomain::Real,
        ShapeFact::from(vec![Some(2), Some(3)]),
    );
    let pulse = numeric(
        NumericClass::Double,
        NumericDomain::Real,
        ShapeFact::from(vec![Some(2), Some(2)]),
    );
    let out = call(
        vec![t.clone(), delay, pulse],
        vec![LiteralValue::Unknown; 3],
        1,
    );
    assert!(!out.diagnostics.is_empty());
    assert_eq!(out.outputs[0].kind, ValueKindFact::Unknown);
    let out = call(
        vec![
            t,
            numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar),
            numeric(
                NumericClass::Double,
                NumericDomain::Real,
                ShapeFact::from(vec![Some(1), Some(2)]),
            ),
            numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Number(0.0),
        ],
        1,
    );
    assert!(!out.diagnostics.is_empty());
    assert_eq!(out.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn unknown_time_and_discarded_output_are_conservative() {
    let args = vec![
        ValueFact::unknown(DynamicReason::RuntimeValue),
        numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar),
        ValueFact::scalar(ValueKindFact::String),
    ];
    assert_eq!(
        call(args.clone(), vec![LiteralValue::Unknown; 3], 1).outputs[0].kind,
        ValueKindFact::Unknown
    );
    assert!(call(args, vec![LiteralValue::Unknown; 3], 0)
        .outputs
        .is_empty());
}

#[test]
fn inexact_integer_time_is_diagnosed() {
    let out = call(
        vec![
            numeric(NumericClass::UInt64, NumericDomain::Real, ShapeFact::Scalar),
            numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar),
            ValueFact::scalar(ValueKindFact::String),
        ],
        vec![
            LiteralValue::Integer {
                text: "9007199254740993".into(),
                class: NumericClass::UInt64,
            },
            LiteralValue::Unknown,
            LiteralValue::Unknown,
        ],
        1,
    );
    assert!(!out.diagnostics.is_empty());
    assert_eq!(out.outputs[0].kind, ValueKindFact::Unknown);
}

fn pulse_call(pulse: ValueFact, literal: LiteralValue) -> CallInference {
    call(
        vec![
            numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar),
            numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar),
            pulse,
        ],
        vec![LiteralValue::Unknown, LiteralValue::Unknown, literal],
        1,
    )
}

fn callable(identity: Option<CallableIdentity>) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity,
        capabilities: CapabilitySet::default(),
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
fn sampled_and_statically_known_direct_pulses_do_not_infer_callback_effects() {
    for (pulse, literal) in [
        (
            numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar),
            LiteralValue::Unknown,
        ),
        (
            ValueFact::scalar(ValueKindFact::String),
            LiteralValue::String(" @rectpuls ".into()),
        ),
        (
            callable(Some(CallableIdentity::Builtin(BuiltinId("tripuls".into())))),
            LiteralValue::Unknown,
        ),
    ] {
        let inferred = pulse_call(pulse, literal);
        assert!(!inferred.effects.0.contains(&EffectKind::HostCallback));
        assert!(!inferred.effects.0.contains(&EffectKind::Unknown));
        assert!(inferred.effects.0.contains(&EffectKind::MaySuspend));
        assert!(inferred.effects.0.contains(&EffectKind::MayThrow));
    }
}

#[test]
fn typed_builtin_callback_unions_catalog_effects_and_capabilities() {
    for (name, effect, capability) in [
        (
            "csvread",
            EffectKind::FilesystemRead,
            CapabilityRequirement::Filesystem,
        ),
        (
            "sendmail",
            EffectKind::Network,
            CapabilityRequirement::Network,
        ),
    ] {
        let inferred = pulse_call(
            callable(Some(CallableIdentity::Builtin(BuiltinId(name.into())))),
            LiteralValue::Unknown,
        );
        assert!(inferred.effects.0.contains(&EffectKind::HostCallback));
        assert!(inferred.effects.0.contains(&effect), "{name}");
        assert!(inferred.capabilities.0.contains(&capability), "{name}");
        assert!(!inferred.effects.0.contains(&EffectKind::Unknown), "{name}");
    }
}

#[test]
fn unresolved_callable_and_text_names_retain_unknown_effect_and_declared_capabilities() {
    let mut dynamic = callable(Some(CallableIdentity::DynamicName(SymbolName(
        "rectpuls".into(),
    ))));
    if let ValueKindFact::Callable(ref mut callback) = dynamic.kind {
        callback
            .capabilities
            .0
            .insert(CapabilityRequirement::Network);
    }
    let inferred = pulse_call(dynamic, LiteralValue::Unknown);
    assert!(inferred.effects.0.contains(&EffectKind::HostCallback));
    assert!(inferred.effects.0.contains(&EffectKind::Unknown));
    assert!(inferred
        .capabilities
        .0
        .contains(&CapabilityRequirement::Network));

    let named = pulse_call(
        ValueFact::scalar(ValueKindFact::Character),
        LiteralValue::Character("shadowed_function".into()),
    );
    assert!(named.effects.0.contains(&EffectKind::HostCallback));
    assert!(named.effects.0.contains(&EffectKind::Unknown));
}
