use super::*;
fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::default(),
    }
}

#[test]
fn text_empty_geometry_and_requested_outputs_remain_exact() {
    let entry = &super::super::extract_html_text::EXTRACT_HTML_TEXT_CATALOG_ENTRY;
    let scalarized = request(vec![ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(1), Some(1), Some(1)]),
        StorageFact::Dense,
    )]);
    assert_eq!(
        infer(&scalarized, entry, ExtractionOperation::Text).outputs[0].shape,
        ShapeFact::Scalar
    );
    for kind in [ValueKindFact::String, ValueKindFact::Character] {
        let mut call = request(vec![ValueFact::proven(
            kind.clone(),
            ShapeFact::from(vec![Some(0), Some(3)]),
            StorageFact::Dense,
        )]);
        let result = infer(&call, entry, ExtractionOperation::Text);
        assert!(result.diagnostics.is_empty());
        let columns = if matches!(kind, ValueKindFact::Character) {
            1
        } else {
            3
        };
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(0), Some(columns)])
        );
        assert_eq!(result.outputs[0].kind, ValueKindFact::String);
        call.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
        assert!(!infer(&call, entry, ExtractionOperation::Text)
            .diagnostics
            .is_empty());
    }
}

#[test]
fn selector_cardinality_is_dynamic_but_elements_have_html_identity() {
    let entry = &super::super::find_element::FIND_ELEMENT_CATALOG_ENTRY;
    let call = request(vec![ValueFact::unknown(DynamicReason::RuntimeValue); 2]);
    let result = infer(&call, entry, ExtractionOperation::Subtrees);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![None, Some(1)])
    );
    let ValueKindFact::Cell(cell) = &result.outputs[0].kind else {
        panic!("cell result");
    };
    let ValueKindFact::Object(object) = &cell.element.kind else {
        panic!("htmlTree element");
    };
    assert_eq!(
        object.runtime_class,
        Some(StaticClassIdentity::new("htmlTree").owned())
    );
    assert!(!cell.elements_complete);
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    assert!(!infer(
        &request(vec![numeric, call.arguments[1].clone()]),
        entry,
        ExtractionOperation::Subtrees
    )
    .diagnostics
    .is_empty());
}
