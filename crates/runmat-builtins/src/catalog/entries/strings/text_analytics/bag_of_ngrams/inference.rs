use crate::catalog::inference::finish_fixed;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(StaticClassIdentity::new("bagOfNgrams").owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    finish_fixed(entry, request, output, Vec::new())
}
