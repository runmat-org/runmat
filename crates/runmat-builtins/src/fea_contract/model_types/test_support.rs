//! Table-driven contract checks shared by every model-object family's tests.

use super::inference::{ModelArg, ModelObjectCall, ARITY, RESIDENT};
use crate::fea_contract::constructor_inference::object_fact;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
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

fn valid(rule: ModelArg) -> ValueFact {
    match rule {
        ModelArg::Text | ModelArg::TextList => ValueFact::scalar(ValueKindFact::String),
        ModelArg::RealScalar => double(ShapeFact::Scalar),
        ModelArg::RealVector3 => double(ShapeFact::from(vec![Some(1), Some(3)])),
        ModelArg::Typed(classes) | ModelArg::TypedOrCell(classes) => object_fact(classes[0]),
        ModelArg::ObjectOrCellOf { object, .. } => object_fact(object[0]),
    }
}

fn run(entry: &BuiltinCatalogEntry, arguments: Vec<ValueFact>) -> runmat_types::CallInference {
    match entry.contract.inference_rule {
        crate::BuiltinInferenceRule::Identity(rule) => rule.infer(&request(arguments), entry),
        _ => panic!("model-object entries use identity inference"),
    }
}

fn codes(entry: &BuiltinCatalogEntry, arguments: Vec<ValueFact>) -> Vec<String> {
    run(entry, arguments)
        .diagnostics
        .into_iter()
        .map(|diagnostic| diagnostic.code)
        .collect()
}

/// Valid, unknown, wrong-arity, wrong-kind, and resident cases for one entry.
pub(crate) fn assert_model_object_contract(entry: &BuiltinCatalogEntry, call: &ModelObjectCall) {
    let name = entry.identity.name;
    let proven: Vec<_> = call.arguments.iter().map(|rule| valid(*rule)).collect();
    let inference = run(entry, proven.clone());
    assert!(
        inference.diagnostics.is_empty(),
        "{name}: {:?}",
        inference.diagnostics
    );
    let ValueKindFact::Object(object) = &inference.outputs[0].kind else {
        panic!("{name}: expected an object result")
    };
    assert_eq!(
        object.runtime_class.as_ref(),
        Some(&call.result.owned()),
        "{name}"
    );
    assert_eq!(object.handle_semantics, Some(false), "{name}");
    assert_eq!(
        inference.outputs[0].residency,
        ResidencyFact::Host,
        "{name}"
    );

    let unknown = vec![ValueFact::unknown(DynamicReason::RuntimeValue); call.arguments.len()];
    assert!(
        codes(entry, unknown).is_empty(),
        "{name}: unknown inputs must not error"
    );

    let mut extra = proven.clone();
    extra.push(ValueFact::scalar(ValueKindFact::String));
    assert!(
        codes(entry, extra).iter().any(|code| code == ARITY),
        "{name}: arity"
    );

    for index in 0..call.arguments.len() {
        let mut wrong = proven.clone();
        wrong[index] = ValueFact::scalar(ValueKindFact::Logical);
        assert!(
            !codes(entry, wrong).is_empty(),
            "{name}: argument {index} accepted a logical scalar"
        );
        let mut device = proven.clone();
        device[index].residency = ResidencyFact::Device { provider: None };
        assert!(
            codes(entry, device).iter().any(|code| code == RESIDENT),
            "{name}: argument {index} accepted a resident value"
        );
    }
}
