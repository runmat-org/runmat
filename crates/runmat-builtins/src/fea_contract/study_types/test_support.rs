//! Table-driven contract checks shared by the study-type family tests.

use super::inference::{
    StudyArgument, StudyConstructorSignature, StudyResult, ARITY_CODE, NUMERIC_CODE, OBJECT_CODE,
    POSITIONS_CODE, RESIDENT_CODE, TEXT_CODE,
};
use crate::{BuiltinCatalogEntry, BuiltinInferenceRule};
use runmat_types::{
    CallInference, CallRequest, CellFact, CertaintyFact, DynamicReason, LiteralContext,
    NumericClass, NumericDomain, NumericFact, ObjectFact, OutputSelection, RequestedOutputCount,
    ResidencyFact, ShapeFact, StaticClassIdentity, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub fn infer(entry: &BuiltinCatalogEntry, arguments: Vec<ValueFact>) -> CallInference {
    let request = CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    match entry.contract.inference_rule {
        BuiltinInferenceRule::Identity(rule) => rule.infer(&request, entry),
        other => panic!("expected identity inference, got {other:?}"),
    }
}

pub fn real(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

pub fn object(class: Option<StaticClassIdentity>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: class.map(StaticClassIdentity::owned),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

pub fn cell_of(element: ValueFact, count: usize) -> ValueFact {
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

/// A statically valid fact for one argument kind.
pub fn valid_fact(kind: StudyArgument) -> ValueFact {
    match kind {
        StudyArgument::Text => ValueFact::scalar(ValueKindFact::String),
        StudyArgument::TextList => cell_of(ValueFact::scalar(ValueKindFact::String), 2),
        StudyArgument::RealScalar | StudyArgument::Count => real(ShapeFact::Scalar),
        StudyArgument::RealValues => real(ShapeFact::from(vec![Some(1), Some(3)])),
        StudyArgument::Positions => real(ShapeFact::from(vec![Some(3), Some(3)])),
        StudyArgument::Object(classes) => object(classes.first().copied()),
        StudyArgument::ObjectList(classes) => cell_of(object(classes.first().copied()), 2),
    }
}

pub fn expected_code(kind: StudyArgument) -> &'static str {
    match kind {
        StudyArgument::Text | StudyArgument::TextList => TEXT_CODE,
        StudyArgument::RealScalar | StudyArgument::Count | StudyArgument::RealValues => {
            NUMERIC_CODE
        }
        StudyArgument::Positions => POSITIONS_CODE,
        StudyArgument::Object(_) | StudyArgument::ObjectList(_) => OBJECT_CODE,
    }
}

fn result_class(signature: &StudyConstructorSignature, arguments: &[ValueFact]) -> Option<String> {
    match signature.result {
        StudyResult::Class(class) => Some(class.display_name().to_string()),
        StudyResult::SameClassAs { argument, .. } => match &arguments[argument].kind {
            ValueKindFact::Object(object) => object
                .runtime_class
                .as_ref()
                .map(|class| class.display_name().to_string()),
            _ => None,
        },
    }
}

fn output_class(inference: &CallInference) -> Option<String> {
    match &inference.outputs[0].kind {
        ValueKindFact::Object(object) => object
            .runtime_class
            .as_ref()
            .map(|class| class.display_name().to_string()),
        other => panic!("expected an object result, got {other:?}"),
    }
}

fn has_code(inference: &CallInference, code: &str, argument: Option<usize>) -> bool {
    inference.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == code && (argument.is_none() || diagnostic.argument == argument)
    })
}

/// Valid, wrong-arity, wrong-kind, device-resident, and unknown-input checks.
pub fn assert_constructor_contract(
    entry: &BuiltinCatalogEntry,
    signature: &StudyConstructorSignature,
) {
    let name = entry.identity.name;
    let valid: Vec<_> = signature
        .arguments
        .iter()
        .copied()
        .map(valid_fact)
        .collect();
    let accepted = infer(entry, valid.clone());
    assert!(
        accepted.diagnostics.is_empty(),
        "{name}: valid facts produced {:?}",
        accepted.diagnostics
    );
    assert_eq!(accepted.outputs.len(), 1, "{name}: one output");
    assert_eq!(
        output_class(&accepted),
        result_class(signature, &valid),
        "{name}: result class"
    );
    assert_eq!(
        accepted.outputs[0].shape,
        ShapeFact::Scalar,
        "{name}: scalar object"
    );

    let mut extra = valid.clone();
    extra.push(ValueFact::scalar(ValueKindFact::String));
    assert!(
        has_code(&infer(entry, extra), ARITY_CODE, None),
        "{name}: extra argument"
    );
    if !valid.is_empty() {
        let mut missing = valid.clone();
        missing.pop();
        assert!(
            has_code(&infer(entry, missing), ARITY_CODE, None),
            "{name}: missing argument"
        );
    }

    for (index, kind) in signature.arguments.iter().copied().enumerate() {
        let mut wrong = valid.clone();
        wrong[index] = ValueFact::scalar(ValueKindFact::Logical);
        assert!(
            has_code(&infer(entry, wrong), expected_code(kind), Some(index)),
            "{name}: wrong kind at argument {index}"
        );
        let mut resident = valid.clone();
        resident[index].residency = ResidencyFact::Device { provider: None };
        assert!(
            has_code(&infer(entry, resident), RESIDENT_CODE, Some(index)),
            "{name}: device-resident argument {index}"
        );
    }

    let unknown = vec![ValueFact::unknown(DynamicReason::RuntimeValue); signature.arguments.len()];
    let dynamic = infer(entry, unknown);
    assert!(
        dynamic.diagnostics.is_empty(),
        "{name}: unknown inputs must not error: {:?}",
        dynamic.diagnostics
    );
    if let StudyResult::SameClassAs { .. } = signature.result {
        assert!(matches!(
            dynamic.outputs[0].certainty,
            CertaintyFact::Dynamic(_)
        ));
    } else {
        assert_eq!(output_class(&dynamic), result_class(signature, &valid));
    }
}
