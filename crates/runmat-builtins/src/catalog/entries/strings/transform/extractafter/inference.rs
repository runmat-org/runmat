use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::super::extractbefore::infer_boundary(request, entry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{DynamicReason, LiteralContext, OutputSelection, ValueFact, ValueKindFact};

    #[test]
    fn preserves_known_string_kind_and_keeps_unknown_dynamic() {
        let entry = &super::super::EXTRACT_AFTER_CATALOG_ENTRY;
        let request = CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        assert_eq!(
            infer(&request, entry).outputs[0].kind,
            ValueKindFact::String
        );
    }
}
