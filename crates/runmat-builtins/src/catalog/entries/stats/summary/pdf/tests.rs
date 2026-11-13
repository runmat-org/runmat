use super::*;
use runmat_types::*;
use std::collections::BTreeMap;

#[test]
fn pdf_tracks_fitted_object_shape_and_single_output_without_claiming_residency() {
    let object = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::PROBABILITY_DISTRIBUTION.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let mut x = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    x.residency = ResidencyFact::Unknown;
    let request = CallRequest {
        arguments: vec![object, x],
        literals: LiteralContext::default(),
        outputs: OutputSelection::default(),
    };
    let result = infer(&request, &PDF_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
}
