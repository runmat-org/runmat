use super::super::random_inference::*;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = arity(request, 2, Some(3));
    let mu = request
        .arguments
        .first()
        .and_then(|arg| arg.shape.known_dims());
    let sigma = request
        .arguments
        .get(1)
        .and_then(|arg| arg.shape.known_dims());
    let rows = if request.arguments.len() == 3 {
        literal_count(request, 2)
    } else {
        match (mu.as_deref(), sigma.as_deref()) {
            (Some([Some(m), _]), Some([_, _, Some(p)])) if *m == 1 || *p == 1 || m == p => {
                Some((*m).max(*p))
            }
            (Some([m, _]), Some([_, _])) => *m,
            _ => None,
        }
    };
    let columns = mu
        .as_deref()
        .and_then(|dims| dims.get(1))
        .copied()
        .flatten()
        .or_else(|| {
            request
                .arguments
                .first()
                .filter(|arg| arg.shape == ShapeFact::Scalar)
                .map(|_| 1)
        });
    let shape = ShapeFact::from(vec![rows, columns]);
    let data = &request.arguments[..request.arguments.len().min(2)];
    let residency = if data.iter().all(|arg| arg.residency == ResidencyFact::Host) {
        ResidencyFact::Host
    } else {
        ResidencyFact::Unknown
    };
    finish(
        entry,
        request,
        vec![numeric(shape, precision(data), residency)],
        diagnostics,
        false,
    )
}
