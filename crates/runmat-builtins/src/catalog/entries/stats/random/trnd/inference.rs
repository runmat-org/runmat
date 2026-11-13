use super::super::random_inference::*;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = arity(request, 1, None);
    let shape = if request.arguments.len() == 1 {
        request
            .arguments
            .first()
            .map_or(ShapeFact::Unknown, |arg| normalized_shape(&arg.shape))
    } else {
        dimensions(request, 1)
    };
    let class = precision(&request.arguments[..request.arguments.len().min(1)]);
    let residency = if request
        .arguments
        .first()
        .is_some_and(|arg| arg.residency == ResidencyFact::Host)
    {
        ResidencyFact::Host
    } else {
        ResidencyFact::Unknown
    };
    finish(
        entry,
        request,
        vec![numeric(shape, class, residency)],
        diagnostics,
        false,
    )
}
