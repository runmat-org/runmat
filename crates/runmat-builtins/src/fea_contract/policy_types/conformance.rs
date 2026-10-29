//! Shared conformance checks applied to every typed FEA policy identity by the
//! owning class family's tests: a valid call, wrong arity, a wrong argument
//! kind per position, a device-resident argument, and unknown inputs.

use super::inference::{PolicyArg, PolicyCall, ARITY, LOGICAL, NUMERIC, RESIDENT, TEXT, TYPED};
use crate::{BuiltinCatalogEntry, BuiltinInferenceRule};
use runmat_types::{
    CallInference, CallRequest, CellFact, DynamicReason, LiteralContext, NumericClass,
    NumericDomain, NumericFact, ObjectFact, OutputSelection, RequestedOutputCount, ResidencyFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(crate) fn run(entry: &BuiltinCatalogEntry, arguments: Vec<ValueFact>) -> CallInference {
    let BuiltinInferenceRule::Identity(rule) = entry.contract.inference_rule else {
        panic!(
            "{} must own an identity inference rule",
            entry.identity.name
        )
    };
    rule.infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
        entry,
    )
}

fn double(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

pub(crate) fn object(class: runmat_types::StaticClassIdentity) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn cell_of(element: ValueFact, count: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element.clone()),
            elements: vec![element; count],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(1), Some(count)]),
        StorageFact::Dense,
    )
}

pub(crate) fn valid(rule: PolicyArg) -> ValueFact {
    match rule.supplied() {
        PolicyArg::Optional(_) => unreachable!("optional rules wrap a supplied rule"),
        PolicyArg::Text => ValueFact::scalar(ValueKindFact::String),
        PolicyArg::RealScalar | PolicyArg::Count | PolicyArg::ScalarOrVector3 => {
            double(ShapeFact::Scalar)
        }
        PolicyArg::RealVector => double(ShapeFact::from(vec![Some(1), Some(4)])),
        PolicyArg::RealVector3 => double(ShapeFact::from(vec![Some(1), Some(3)])),
        PolicyArg::Logical => ValueFact::scalar(ValueKindFact::Logical),
        PolicyArg::Typed(classes) | PolicyArg::TypedOrCell(classes) => object(classes[0]),
        PolicyArg::TypedCell { classes, minimum } => cell_of(object(classes[0]), minimum.max(1)),
    }
}

/// A proven-incompatible fact for the rule and the diagnostic code it must raise.
pub(crate) fn wrong(rule: PolicyArg) -> (ValueFact, &'static str) {
    match rule.supplied() {
        PolicyArg::Optional(_) => unreachable!("optional rules wrap a supplied rule"),
        PolicyArg::Text => (ValueFact::scalar(ValueKindFact::Logical), TEXT),
        PolicyArg::RealScalar
        | PolicyArg::Count
        | PolicyArg::RealVector
        | PolicyArg::RealVector3
        | PolicyArg::ScalarOrVector3 => (ValueFact::scalar(ValueKindFact::String), NUMERIC),
        PolicyArg::Logical => (double(ShapeFact::Scalar), LOGICAL),
        PolicyArg::Typed(_) | PolicyArg::TypedOrCell(_) | PolicyArg::TypedCell { .. } => {
            (double(ShapeFact::Scalar), TYPED)
        }
    }
}

fn codes(inference: &CallInference) -> Vec<&str> {
    inference
        .diagnostics
        .iter()
        .map(|diagnostic| diagnostic.code.as_str())
        .collect()
}

pub(crate) fn assert_conforms(entry: &BuiltinCatalogEntry, call: &PolicyCall) {
    let name = entry.identity.name;
    let class = call.result.display_name();
    assert!(
        name.strip_prefix(class)
            .is_some_and(|method| method.starts_with('.')),
        "{name} must be a static method of its result class {class}"
    );
    assert_eq!(
        entry.descriptor.signatures[0].inputs.len(),
        call.arguments.len(),
        "{name} signature and static argument rules disagree"
    );
    assert!(
        entry
            .documentation
            .examples
            .iter()
            .all(|example| example.program.contains(name)),
        "{name} examples must call the documented identity"
    );

    let arguments: Vec<_> = call.arguments.iter().copied().map(valid).collect();
    let accepted = run(entry, arguments.clone());
    assert!(
        accepted.diagnostics.is_empty(),
        "{name}: {:?}",
        accepted.diagnostics
    );
    let ValueKindFact::Object(object) = &accepted.outputs[0].kind else {
        panic!("{name} must return an object fact")
    };
    assert_eq!(
        object.runtime_class.as_ref(),
        Some(&call.result.owned()),
        "{name}"
    );
    assert_eq!(accepted.outputs[0].residency, ResidencyFact::Host, "{name}");

    let mut extra = arguments.clone();
    extra.push(double(ShapeFact::Scalar));
    assert!(
        codes(&run(entry, extra)).contains(&ARITY),
        "{name} accepted an extra argument"
    );

    let unknown = vec![ValueFact::unknown(DynamicReason::RuntimeValue); call.arguments.len()];
    let unresolved = run(entry, unknown);
    assert!(
        unresolved.diagnostics.is_empty(),
        "{name}: {:?}",
        unresolved.diagnostics
    );

    for (index, rule) in call.arguments.iter().copied().enumerate() {
        let (fact, code) = wrong(rule);
        let mut invalid = arguments.clone();
        invalid[index] = fact;
        let inference = run(entry, invalid);
        assert!(
            inference
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == code && diagnostic.argument == Some(index)),
            "{name} argument {index} must raise {code}: {:?}",
            inference.diagnostics
        );
        let mut resident = arguments.clone();
        resident[index].residency = ResidencyFact::Device { provider: None };
        assert!(
            codes(&run(entry, resident)).contains(&RESIDENT),
            "{name} argument {index}"
        );
    }
    let required = call.required();
    assert!(
        call.arguments[required..]
            .iter()
            .all(|rule| rule.is_optional()),
        "{name}: only trailing arguments may be optional"
    );
    for (index, rule) in call.arguments.iter().enumerate() {
        assert_eq!(
            entry.descriptor.signatures[0].inputs[index].arity,
            if rule.is_optional() {
                crate::BuiltinParamArity::Optional
            } else {
                crate::BuiltinParamArity::Required
            },
            "{name} argument {index} arity disagrees with its static rule"
        );
    }
    for supplied in required..arguments.len() {
        let inference = run(entry, arguments[..supplied].to_vec());
        assert!(
            inference.diagnostics.is_empty(),
            "{name} must accept {supplied} argument(s): {:?}",
            inference.diagnostics
        );
    }
    if let Some(missing) = required.checked_sub(1) {
        assert!(
            codes(&run(entry, arguments[..missing].to_vec())).contains(&ARITY),
            "{name} accepted a missing argument"
        );
    }
}

/// Applies the shared checks to every identity of one class family.
pub(crate) fn assert_family(entries: &[(&BuiltinCatalogEntry, &PolicyCall)]) {
    for (entry, call) in entries {
        assert_conforms(entry, call);
    }
}
