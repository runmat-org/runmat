use super::{DAMP_CATALOG_ENTRY, DCGAIN_CATALOG_ENTRY, ISSTABLE_CATALOG_ENTRY};
use runmat_types::{
    CallRequest, ClassId, LiteralContext, ObjectFact, OutputSelection, RequestedOutputCount,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

#[test]
fn model_admission_distinguishes_unknown_identity_from_known_other_class() {
    for entry in [
        &DAMP_CATALOG_ENTRY,
        &DCGAIN_CATALOG_ENTRY,
        &ISSTABLE_CATALOG_ENTRY,
    ] {
        for class in [None, Some(ClassId(7))] {
            let request = CallRequest {
                arguments: vec![ValueFact::proven(
                    ValueKindFact::Object(ObjectFact {
                        class,
                        runtime_class: None,
                        properties: BTreeMap::new(),
                        properties_complete: false,
                        handle_semantics: None,
                    }),
                    ShapeFact::Scalar,
                    StorageFact::Opaque,
                )],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
            };
            let result = crate::infer_catalog_call(entry, &request);
            assert_eq!(
                result.diagnostics.is_empty(),
                class.is_none(),
                "{}: {:?}",
                entry.identity.name,
                result.diagnostics,
            );
        }
    }
}
