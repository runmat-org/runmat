use super::*;
use crate::builtin_catalog_entry_by_name;
use runmat_types::{
    AliasFact, CellFact, ContiguityFact, LayoutFact, MutationFact, ResidencyFact, ViewFact,
};

#[test]
fn outputs_are_fresh_materializations() {
    let perms = builtin_catalog_entry_by_name("perms").expect("perms entry");
    let mut input = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(1), Some(3)]),
    );
    input.layout = LayoutFact::Strided;
    input.contiguity = ContiguityFact::NonContiguous;
    input.view = ViewFact::ReadOnlyView;
    input.alias = AliasFact::Shared;
    input.mutation = MutationFact::Immutable;
    input.residency = ResidencyFact::Device {
        provider: Some("test".into()),
    };
    let inferred = crate::infer_catalog_call(perms, &request(vec![input]));
    let output = &inferred.outputs[0];
    assert_eq!(output.layout, LayoutFact::ColumnMajor);
    assert_eq!(output.contiguity, ContiguityFact::Contiguous);
    assert_eq!(output.view, ViewFact::Materialized);
    assert_eq!(output.alias, AliasFact::Unique);
    assert_eq!(output.mutation, MutationFact::ValueSemantics);
    assert_eq!(
        output.residency,
        ResidencyFact::Device {
            provider: Some("test".into())
        }
    );

    let cells = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::scalar(ValueKindFact::Logical)),
            elements: vec![
                ValueFact::scalar(ValueKindFact::Logical),
                ValueFact::scalar(ValueKindFact::Logical),
            ],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    let inferred = crate::infer_catalog_call(perms, &request(vec![cells]));
    let ValueKindFact::Cell(cell) = &inferred.outputs[0].kind else {
        panic!("expected cell fact")
    };
    assert!(cell.elements.is_empty());
    assert!(!cell.elements_complete);

    let nchoosek = builtin_catalog_entry_by_name("nchoosek").expect("nchoosek entry");
    let input = numeric(
        NumericClass::UInt32,
        ShapeFact::from(vec![Some(1), Some(4)]),
    );
    let inferred = crate::infer_catalog_call(
        nchoosek,
        &request(vec![
            input,
            numeric(NumericClass::Double, ShapeFact::Scalar),
        ]),
    );
    assert_eq!(inferred.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(inferred.outputs[0].view, ViewFact::Materialized);
    assert_eq!(inferred.outputs[0].alias, AliasFact::Unique);
}
