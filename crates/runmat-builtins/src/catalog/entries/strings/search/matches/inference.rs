use crate::catalog::entries::strings::transform::text_facts::*;
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 2, 2, &mut diagnostics);
    if request
        .arguments
        .first()
        .is_some_and(|v| invalid_text(v, CellTextPolicy::ScalarElements))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-MATCHES-TEXT",
            "matches requires text input",
            0,
        ));
    }
    if request.arguments.get(1).is_some_and(|v| !pattern(v)) {
        diagnostics.push(argument_error(
            "RM-CATALOG-MATCHES-PATTERN",
            "matches requires a scalar text or pattern object",
            1,
        ));
    }
    let shape = request
        .arguments
        .first()
        .map_or(ShapeFact::Unknown, text_shape);
    finish_fixed(
        entry,
        request,
        host(ValueKindFact::Logical, shape),
        diagnostics,
    )
}
fn arity(
    request: &CallRequest,
    min: usize,
    max: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if !(min..=max).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-MATCHES-ARITY",
            "unsupported input count",
            request.arguments.len().min(max),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identity_facts_reject_missing_and_nontext_inputs() {
        let entry = &super::super::MATCHES_CATALOG_ENTRY;
        let mut request = CallRequest {
            arguments: Vec::new(),
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        assert!(!infer(&request, entry).diagnostics.is_empty());
        request.arguments = vec![ValueFact::scalar(ValueKindFact::Logical); 2];
        assert!(!infer(&request, entry).diagnostics.is_empty());
        let scalar = ValueFact::scalar(ValueKindFact::String);
        let text = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        request.arguments = vec![text, scalar.clone()];
        let result = infer(&request, entry);
        assert!(result.diagnostics.is_empty(), "{:#?}", result.diagnostics);
        assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
    }
}
