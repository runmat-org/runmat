use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::super::insertbefore::infer_insert(request, entry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection, ValueFact, ValueKindFact};

    #[test]
    fn returns_string_for_string_source() {
        let request = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::String); 3],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        assert_eq!(
            infer(&request, &super::super::INSERT_AFTER_CATALOG_ENTRY).outputs[0].kind,
            ValueKindFact::String
        );
    }
}
