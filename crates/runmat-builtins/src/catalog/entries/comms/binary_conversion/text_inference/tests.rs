use super::*;
use runmat_types::{CellFact, LiteralContext, OutputSelection, RequestedOutputCount};

fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

#[test]
fn decimal_text_inference_knows_rows_but_not_unobserved_digits() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let width = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    for (conversion, entry) in [
        (
            TextConversion::Dec2Bin,
            &super::super::DEC2BIN_CATALOG_ENTRY,
        ),
        (
            TextConversion::Dec2Hex,
            &super::super::DEC2HEX_CATALOG_ENTRY,
        ),
    ] {
        let result = infer(
            &request(vec![input.clone(), width.clone()]),
            entry,
            conversion,
        );
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs[0].kind, ValueKindFact::Character);
        assert_eq!(
            result.outputs[0].shape.known_dims(),
            Some(vec![Some(6), None])
        );
        assert_eq!(
            result.outputs[0].residency,
            runmat_types::ResidencyFact::Host
        );
        assert_eq!(entry.bindings, &crate::REQUIRED_DEFAULT_BINDING);
    }
}

#[test]
fn text_to_decimal_preserves_string_shape_and_char_row_count() {
    let strings = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let chars = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(3), Some(8)]),
        StorageFact::Dense,
    );
    for (conversion, entry) in [
        (
            TextConversion::Bin2Dec,
            &super::super::BIN2DEC_CATALOG_ENTRY,
        ),
        (
            TextConversion::Hex2Dec,
            &super::super::HEX2DEC_CATALOG_ENTRY,
        ),
    ] {
        let result = infer(&request(vec![strings.clone()]), entry, conversion);
        assert_eq!(result.outputs[0].shape, strings.shape);
        assert_eq!(
            result.outputs[0].numeric().unwrap().class,
            NumericClass::Double
        );
        let result = infer(&request(vec![chars.clone()]), entry, conversion);
        assert_eq!(
            result.outputs[0].shape.known_dims(),
            Some(vec![Some(3), Some(1)])
        );
    }
}

#[test]
fn invalid_class_and_width_are_diagnosed_without_claiming_width_from_shape() {
    let invalid = ValueFact::scalar(ValueKindFact::String);
    let width = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    let result = infer(
        &request(vec![invalid, width]),
        &super::super::DEC2BIN_CATALOG_ENTRY,
        TextConversion::Dec2Bin,
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM.TEXTCONVERSION.INPUT"));
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM.TEXTCONVERSION.WIDTH"));
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let result = infer(
        &request(vec![unknown]),
        &super::super::DEC2HEX_CATALOG_ENTRY,
        TextConversion::Dec2Hex,
    );
    assert_eq!(result.outputs[0].shape.known_dims(), Some(vec![None, None]));
}

#[test]
fn requested_and_discarded_outputs_keep_fixed_single_slot() {
    let mut call = request(vec![ValueFact::scalar(ValueKindFact::String)]);
    call.outputs.discarded.insert(0);
    let result = infer(
        &call,
        &super::super::BIN2DEC_CATALOG_ENTRY,
        TextConversion::Bin2Dec,
    );
    assert_eq!(result.outputs.len(), 1);
    assert_eq!(result.discarded, vec![0]);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    call.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
    let invalid = infer(
        &call,
        &super::super::BIN2DEC_CATALOG_ENTRY,
        TextConversion::Bin2Dec,
    );
    assert!(!invalid.diagnostics.is_empty());
}

#[test]
fn logical_width_is_admitted_but_nonscalar_logical_width_is_not() {
    let input = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let logical = ValueFact::scalar(ValueKindFact::Logical);
    for (conversion, entry) in [
        (
            TextConversion::Dec2Bin,
            &super::super::DEC2BIN_CATALOG_ENTRY,
        ),
        (
            TextConversion::Dec2Hex,
            &super::super::DEC2HEX_CATALOG_ENTRY,
        ),
    ] {
        let result = infer(
            &request(vec![input.clone(), logical.clone()]),
            entry,
            conversion,
        );
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        let nonscalar = ValueFact::proven(
            ValueKindFact::Logical,
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        );
        let invalid = infer(&request(vec![input.clone(), nonscalar]), entry, conversion);
        assert!(invalid
            .diagnostics
            .iter()
            .any(|d| d.code == "RM.TEXTCONVERSION.WIDTH"));
    }
}

#[test]
fn singleton_text_arrays_scalarize_but_empty_string_shape_survives() {
    for (conversion, entry) in [
        (
            TextConversion::Bin2Dec,
            &super::super::BIN2DEC_CATALOG_ENTRY,
        ),
        (
            TextConversion::Hex2Dec,
            &super::super::HEX2DEC_CATALOG_ENTRY,
        ),
    ] {
        for dims in [vec![Some(1)], vec![Some(1), Some(1), Some(1)]] {
            let input = ValueFact::proven(
                ValueKindFact::String,
                ShapeFact::from(dims),
                StorageFact::Dense,
            );
            let result = infer(&request(vec![input]), entry, conversion);
            assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
            assert_eq!(result.outputs[0].storage, StorageFact::Scalar);
        }
        let empty_shape = ShapeFact::from(vec![Some(0), Some(2)]);
        let empty = ValueFact::proven(
            ValueKindFact::String,
            empty_shape.clone(),
            StorageFact::Dense,
        );
        let result = infer(&request(vec![empty]), entry, conversion);
        assert_eq!(result.outputs[0].shape, empty_shape);
        assert_eq!(result.outputs[0].storage, StorageFact::Dense);
    }
}

#[test]
fn known_invalid_nested_text_geometry_is_diagnosed() {
    let cell = |element: ValueFact| {
        ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(element.clone()),
                elements: vec![element],
                elements_complete: true,
            }),
            ShapeFact::Scalar,
            StorageFact::Dense,
        )
    };
    let multi_string = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    let multi_char = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    for nested in [multi_string, multi_char] {
        let result = infer(
            &request(vec![cell(nested)]),
            &super::super::HEX2DEC_CATALOG_ENTRY,
            TextConversion::Hex2Dec,
        );
        assert!(result
            .diagnostics
            .iter()
            .any(|d| d.code == "RM.TEXTCONVERSION.CELL"));
    }
    let uncertain = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![None, Some(1)]),
        StorageFact::Dense,
    );
    let result = infer(
        &request(vec![cell(uncertain)]),
        &super::super::BIN2DEC_CATALOG_ENTRY,
        TextConversion::Bin2Dec,
    );
    assert!(!result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM.TEXTCONVERSION.CELL"));
}
