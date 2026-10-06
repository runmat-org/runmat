use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn scalar_numeric() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

#[test]
fn current_function_forms_are_zero_output_with_known_and_unknown_bounds() {
    let bounds = vec![
        scalar_numeric(),
        ValueFact::unknown(DynamicReason::RuntimeValue),
    ];
    for (entry, infer) in [
        (
            &NARGINCHK_CATALOG_ENTRY,
            inference::infer_narginchk as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
        ),
        (&NARGOUTCHK_CATALOG_ENTRY, inference::infer_nargoutchk),
    ] {
        let result = infer(
            &request(
                bounds.clone(),
                vec![LiteralValue::Number(0.0), LiteralValue::Unknown],
                0,
            ),
            entry,
        );
        assert!(result.outputs.is_empty());
        assert!(result.diagnostics.is_empty());
        assert!(entry
            .bindings
            .iter()
            .any(|binding| binding.variant == "default"));
    }
}

#[test]
fn invalid_bound_and_residency_are_rejected_without_invented_output() {
    let mut resident = scalar_numeric();
    resident.residency = ResidencyFact::Device { provider: None };
    let request = request(
        vec![resident, ValueFact::scalar(ValueKindFact::Logical)],
        vec![LiteralValue::Number(-1.0), LiteralValue::Bool(true)],
        0,
    );
    for (entry, infer) in [
        (
            &NARGINCHK_CATALOG_ENTRY,
            inference::infer_narginchk as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
        ),
        (&NARGOUTCHK_CATALOG_ENTRY, inference::infer_nargoutchk),
    ] {
        let result = infer(&request, entry);
        assert!(result.outputs.is_empty());
        assert!(result.diagnostics.len() >= 2);
    }
}

#[test]
fn legacy_output_type_tracks_text_and_struct_without_guessing_empty_shape() {
    let arguments = vec![scalar_numeric(), scalar_numeric(), scalar_numeric()];
    let text = inference::infer_nargoutchk(
        &request(arguments.clone(), vec![], 1),
        &NARGOUTCHK_CATALOG_ENTRY,
    );
    assert_eq!(text.outputs.len(), 1);
    assert_eq!(text.outputs[0].kind, ValueKindFact::Character);
    assert_eq!(text.outputs[0].shape, ShapeFact::from(vec![None, None]));

    let mut structured = arguments;
    structured.push(ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(6)]),
        StorageFact::Dense,
    ));
    let value = inference::infer_nargoutchk(
        &request(
            structured,
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Character("struct".into()),
            ],
            1,
        ),
        &NARGOUTCHK_CATALOG_ENTRY,
    );
    assert!(matches!(value.outputs[0].kind, ValueKindFact::Struct(_)));
    assert_eq!(value.outputs[0].shape, ShapeFact::from(vec![None, None]));
    assert!(value.diagnostics.is_empty());
}

#[test]
fn canonical_contract_preserves_all_legacy_forms_and_error_identifiers() {
    assert_eq!(NARGINCHK_DESCRIPTOR.signatures.len(), 1);
    assert_eq!(NARGOUTCHK_DESCRIPTOR.signatures.len(), 3);
    assert_eq!(NARGOUTCHK_DESCRIPTOR.signatures[0].inputs.len(), 2);
    assert_eq!(NARGOUTCHK_DESCRIPTOR.signatures[1].inputs.len(), 3);
    assert_eq!(NARGOUTCHK_DESCRIPTOR.signatures[2].inputs.len(), 4);
    assert!(NARGOUTCHK_DESCRIPTOR.signatures[0].outputs.is_empty());
    assert_eq!(NARGOUTCHK_DESCRIPTOR.signatures[1].outputs.len(), 1);
    assert_eq!(NARGOUTCHK_DESCRIPTOR.signatures[2].outputs.len(), 1);
    assert_eq!(NARGINCHK_ERRORS.len(), 5);
    assert_eq!(NARGOUTCHK_ERRORS.len(), 5);
    assert_eq!(
        NARGINCHK_ERROR_CONTEXT_UNAVAILABLE.identifier,
        Some("RunMat:NarginchkContextUnavailable")
    );
    assert_eq!(
        NARGOUTCHK_ERROR_CONTEXT_UNAVAILABLE.identifier,
        Some("RunMat:NargoutchkContextUnavailable")
    );
    assert_eq!(NARGINCHK_INTEGER_CAPABILITIES.len(), 1);
    assert_eq!(NARGOUTCHK_INTEGER_CAPABILITIES.len(), 2);
}
