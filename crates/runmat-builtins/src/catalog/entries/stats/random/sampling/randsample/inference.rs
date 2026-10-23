use super::super::super::random_inference::*;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = arity(request, 2, Some(4));
    let k = literal_count(request, 1);
    let population = request.arguments.first();
    let range = population.is_some_and(|arg| {
        (arg.numeric().is_some() || arg.kind == ValueKindFact::Logical)
            && arg.shape.element_count() == Some(1)
    });
    let dimensions = population.and_then(|arg| arg.shape.known_dims());
    let mut output = if range {
        numeric(
            ShapeFact::Unknown,
            Some(NumericClass::Double),
            ResidencyFact::Host,
        )
    } else {
        population
            .cloned()
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue))
    };
    // A possibly scalar numeric/logical input may be a range bound rather than
    // a population. That branch returns double, even for an integer bound.
    if population.is_some_and(|arg| {
        arg.shape.element_count().is_none()
            && !arg
                .shape
                .known_dims()
                .is_some_and(|dims| dims.iter().any(|dim| dim.is_some_and(|n| n != 1)))
            && (arg.numeric().is_some() || arg.kind == ValueKindFact::Logical)
            && !arg
                .numeric()
                .is_some_and(|n| n.class == NumericClass::Double)
    }) {
        output.kind = ValueKindFact::Unknown;
        output.storage = StorageFact::Unknown;
    }
    output.shape = if range {
        ShapeFact::from(vec![k, Some(1)])
    } else {
        match dimensions.as_deref() {
            Some([Some(1), Some(columns)]) if *columns != 1 => ShapeFact::from(vec![Some(1), k]),
            Some([Some(_), Some(1)]) => ShapeFact::from(vec![k, Some(1)]),
            _ => ShapeFact::Unknown,
        }
    };
    // Population owner restoration is conditional on physical class support.
    output.residency = if population.is_some_and(|arg| arg.residency == ResidencyFact::Host) {
        ResidencyFact::Host
    } else {
        ResidencyFact::Unknown
    };
    output.alias = AliasFact::Unique;
    output.view = ViewFact::Materialized;
    finish(entry, request, vec![output], diagnostics, false)
}
