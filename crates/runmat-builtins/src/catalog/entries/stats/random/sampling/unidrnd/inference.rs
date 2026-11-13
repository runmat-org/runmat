use super::super::super::random_inference::*;
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
    let class = Some(NumericClass::Double);
    let residency = ResidencyFact::Host;
    finish(
        entry,
        request,
        vec![numeric(shape, class, residency)],
        diagnostics,
        false,
    )
}
