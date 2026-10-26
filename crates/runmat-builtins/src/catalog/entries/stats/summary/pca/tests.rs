use super::*;
use runmat_types::*;
#[test]
fn six_output_metadata_preserves_class_and_stable_axes_without_claiming_complete_rows() {
    let request = CallRequest {
        arguments: vec![ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(200), Some(4)]),
            StorageFact::Dense,
        )],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(6)),
    };
    let result = inference::infer(&request, &PCA_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs.len(), 6);
    for output in &result.outputs {
        assert!(matches!(
            output.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real
            })
        ));
    }
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(4), None])
    );
    assert_eq!(
        result.outputs[1].shape,
        ShapeFact::from(vec![Some(200), None])
    );
    assert_eq!(
        result.outputs[3].shape,
        ShapeFact::from(vec![Some(200), Some(1)])
    );
    assert_eq!(
        result.outputs[5].shape,
        ShapeFact::from(vec![Some(1), Some(4)])
    );
    assert_eq!(
        PcaOptionName::from_public("NumComponents"),
        Some(PcaOptionName::NumComponents)
    );
    assert_eq!(PcaOptionName::from_public("Weights"), None);
    assert_eq!(PcaRows::from_public("pairwise"), None);
    assert_eq!(PCA_CATALOG_ENTRY.documentation.examples.len(), 2);
}
