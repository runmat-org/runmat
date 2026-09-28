use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{CallInference, CallRequest, ValueFact, ValueKindFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = if request.arguments.len() == 1 {
        vec![]
    } else {
        vec![argument_error(
            "RM-CATALOG-IS_STRING_SCALAR-ARITY",
            "isStringScalar requires exactly one input",
            0,
        )]
    };
    finish_fixed(
        entry,
        request,
        ValueFact::scalar(ValueKindFact::Logical),
        diagnostics,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{DynamicReason, LiteralContext, OutputSelection};
    #[test]
    fn predicate_result_is_scalar_even_for_unknown_input() {
        let request = CallRequest {
            arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, &super::super::IS_STRING_SCALAR_CATALOG_ENTRY);
        assert_eq!(result.outputs[0], ValueFact::scalar(ValueKindFact::Logical));
        assert!(result.diagnostics.is_empty());
    }
}
