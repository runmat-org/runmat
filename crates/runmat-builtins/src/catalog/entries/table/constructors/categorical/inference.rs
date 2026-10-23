use super::super::facts;
use crate::{catalog::inference::finish_fixed, BuiltinCatalogEntry};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let output = match request.arguments.first() {
        None => facts::categorical(ShapeFact::from(vec![Some(0), Some(0)])),
        Some(source)
            if request.arguments.len() == 1
                && matches!(&source.kind,ValueKindFact::Object(object) if object.runtime_class.as_ref()==Some(&standard::CATEGORICAL.owned())) =>
        {
            facts::retained(source)
        }
        Some(source) => facts::categorical(facts::code_shape(source)),
    };
    finish_fixed(entry, request, output, Vec::new())
}
