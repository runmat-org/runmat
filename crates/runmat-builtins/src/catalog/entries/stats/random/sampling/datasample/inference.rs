use super::super::super::random_inference::*;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = arity(request, 2, None);
    let k = literal_count(request, 1);
    let mut sample = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    sample.shape = ShapeFact::Unknown;
    if let Some(mut dims) = request
        .arguments
        .first()
        .and_then(|arg| arg.shape.known_dims())
    {
        let axis = if request.arguments.len() == 2 {
            dims.iter()
                .position(|dim| dim.is_some_and(|n| n != 1))
                .filter(|axis| dims[..*axis].iter().all(|dim| *dim == Some(1)))
                .or_else(|| dims.iter().all(|dim| *dim == Some(1)).then_some(0))
        } else if request.arguments.len() == 3 {
            literal_count(request, 2).and_then(|n| n.checked_sub(1))
        } else {
            None
        };
        if let Some(axis) = axis.filter(|axis| *axis < dims.len()) {
            dims[axis] = k;
            sample.shape = ShapeFact::from(dims);
        }
    }
    sample.residency = ResidencyFact::Host;
    sample.alias = AliasFact::Unique;
    sample.view = ViewFact::Materialized;
    let indices = numeric(
        ShapeFact::from(vec![k, Some(1)]),
        Some(NumericClass::Double),
        ResidencyFact::Host,
    );
    finish(entry, request, vec![sample, indices], diagnostics, true)
}
