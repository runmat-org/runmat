use crate::{catalog::inference::finish_fixed, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    finish_fixed(
        entry,
        request,
        super::facts::pattern_fact(),
        super::facts::text_input_diagnostics(request, "pattern"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        LiteralContext, OutputSelection, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    };
    #[test]
    fn char_rows_are_valid_but_known_nonscalar_strings_reject() {
        let mut request = CallRequest {
            arguments: vec![ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::from(vec![Some(1), Some(4)]),
                StorageFact::Dense,
            )],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        let entry = &super::super::PATTERN_CATALOG_ENTRY;
        assert!(infer(&request, entry).diagnostics.is_empty());
        request.arguments[0] = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        );
        assert!(infer(&request, entry)
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-pattern-TEXT"));
    }
}
