use super::super::facts;
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    facts::arity(request, 1, Some(4), &mut diagnostics);
    if request.arguments.len() == 4
        && request
            .arguments
            .get(2)
            .and_then(|v| v.shape.element_count())
            .is_some_and(|n| n != 0)
    {
        diagnostics.push(argument_error(
            crate::TABLE_ERROR_INVALID_ARGUMENT.code,
            "edges require an empty third levels argument",
            2,
        ));
    }
    for index in [2, 3] {
        if let Some(value) = request.arguments.get(index) {
            if value
                .shape
                .known_dims()
                .is_some_and(|dims| dims.iter().filter(|d| d.is_some_and(|d| d > 1)).count() > 1)
            {
                diagnostics.push(argument_error(
                    crate::TABLE_ERROR_INVALID_ARGUMENT.code,
                    "ordinal levels and edges must be vectors",
                    index,
                ));
            }
        }
    }
    let shape = request
        .arguments
        .first()
        .map(facts::code_shape)
        .unwrap_or(ShapeFact::Unknown);
    finish_fixed(entry, request, facts::categorical(shape), diagnostics)
}
