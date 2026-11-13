use super::*;
use runmat_types::{DynamicReason, StorageFact, ValueKindFact};

#[test]
fn trailing_dimensions_and_unknown_kind_are_conservative() {
    for (dims, supported) in [
        (vec![Some(1), Some(3), Some(1)], true),
        (vec![Some(1), Some(3), Some(2)], false),
        (vec![Some(1), Some(3), None], true),
        (vec![Some(0), Some(3)], false),
    ] {
        assert_eq!(
            scalar(&ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::from(dims),
                StorageFact::Dense
            )),
            supported
        );
    }
    let mut unknown = ValueFact::unknown(DynamicReason::Unspecified);
    unknown.shape = ShapeFact::from(vec![Some(1), Some(5)]);
    assert!(scalar(&unknown));
    unknown.kind = ValueKindFact::String;
    assert!(!scalar(&unknown));
    unknown.kind = ValueKindFact::Logical;
    assert!(!scalar(&unknown));
}

#[test]
fn scalar_path_consumers_share_conservative_effective_row_validation() {
    for name in ["ls", "what", "dir"] {
        let entry = crate::builtin_catalog_entry_by_name(name).unwrap();
        for (dims, invalid) in [
            (vec![Some(1), Some(2), Some(1)], false),
            (vec![Some(1), Some(1), Some(2)], true),
            (vec![Some(0), Some(0)], true),
            (vec![None, Some(2), None], false),
        ] {
            let request = runmat_types::CallRequest {
                arguments: vec![ValueFact::proven(
                    ValueKindFact::Character,
                    ShapeFact::from(dims),
                    runmat_types::StorageFact::Dense,
                )],
                literals: runmat_types::LiteralContext::default(),
                outputs: runmat_types::OutputSelection::new(
                    runmat_types::RequestedOutputCount::One,
                ),
            };
            assert_eq!(
                !crate::infer_catalog_call(entry, &request)
                    .diagnostics
                    .is_empty(),
                invalid,
                "{name}"
            );
        }
    }
}
