use super::*;
use runmat_types::*;

fn scalar() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn descriptor_retains_legacy_public_forms() {
    let labels: Vec<_> = QUAD_DESCRIPTOR
        .signatures
        .iter()
        .map(|form| form.label)
        .collect();
    assert_eq!(
        labels,
        [
            "q = quad(fun, a, b)",
            "q = quad(fun, a, b, tol, trace, p1, p2, ...)",
            "[q, fcnt] = quad(fun, a, b)",
            "[q, fcnt] = quad(fun, a, b, tol, trace, p1, p2, ...)",
        ]
    );
    assert_eq!(QUAD_EXTENSIONS.len(), 5);
    assert_eq!(QUAD_INTEGER_CAPABILITIES.len(), 5);
}

#[test]
fn all_visible_examples_have_independent_assertions() {
    assert_eq!(documentation::DOCUMENTATION.examples.len(), 3);
    for example in documentation::DOCUMENTATION.examples {
        assert!(example.display_output.is_some());
        assert!(matches!(
            example.verification,
            crate::BuiltinExampleVerification::Assertions { .. }
        ));
    }
}

#[test]
fn forwarded_argument_participates_in_callback_contract() {
    let callback = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: CapabilitySet::default(),
        parameters: vec![scalar(), scalar()],
        parameters_complete: true,
        outputs: vec![scalar()],
        outputs_complete: true,
        variadic_inputs: false,
        variadic_outputs: false,
        captures: vec![],
        captures_complete: true,
    }));
    let empty = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(0), DimensionFact::Known(0)],
        },
        StorageFact::Dense,
    );
    let call = request(
        vec![callback, scalar(), scalar(), empty.clone(), empty, scalar()],
        2,
    );
    let result = inference::infer(&call, &QUAD_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(
        result.outputs[1].numeric().unwrap().class,
        NumericClass::Double
    );
}

#[test]
fn known_invalid_empty_text_tolerance_does_not_prove_result() {
    let mut text = ValueFact::scalar(ValueKindFact::String);
    text.shape = ShapeFact::Shaped {
        dims: vec![DimensionFact::Known(0), DimensionFact::Known(0)],
    };
    let call = request(
        vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            scalar(),
            scalar(),
            text,
        ],
        1,
    );
    let result = inference::infer(&call, &QUAD_CATALOG_ENTRY);
    assert!(!result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn known_invalid_callback_result_never_proves_quad_outputs() {
    let mut vector = scalar();
    vector.shape = ShapeFact::Shaped {
        dims: vec![DimensionFact::Known(2), DimensionFact::Known(1)],
    };
    for output in [
        ValueFact::scalar(ValueKindFact::String),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        })),
        vector,
    ] {
        let callback = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
            identity: None,
            capabilities: CapabilitySet::default(),
            parameters: vec![scalar()],
            parameters_complete: true,
            outputs: vec![output],
            outputs_complete: true,
            variadic_inputs: false,
            variadic_outputs: false,
            captures: vec![],
            captures_complete: true,
        }));
        let call = request(vec![callback, scalar(), scalar()], 2);
        let result = inference::infer(&call, &QUAD_CATALOG_ENTRY);
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-QUAD-CALLBACK-RESULT"));
        assert!(result
            .outputs
            .iter()
            .all(|output| matches!(output.kind, ValueKindFact::Unknown)));
    }
}
