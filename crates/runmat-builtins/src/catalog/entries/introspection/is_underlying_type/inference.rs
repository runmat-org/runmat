use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-IS_UNDERLYING_TYPE-ARITY",
            "isUnderlyingType requires 2 arguments",
            request.arguments.len(),
        ));
    }
    if let Some(designator) = request.arguments.get(1) {
        let valid = match designator.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::String => designator
                .shape
                .element_count()
                .is_none_or(|count| count == 1),
            ValueKindFact::Character => {
                !matches!(&designator.shape, ShapeFact::Shaped { dims } if matches!(dims.first(), Some(DimensionFact::Known(rows)) if *rows != 1))
            }
            _ => false,
        };
        if !valid {
            diagnostics.push(argument_error(
                "RM-CATALOG-IS-UNDERLYING-TYPE-DESIGNATOR",
                "typename must be scalar string or character row",
                1,
            ));
        }
    }
    let mut contract = CallContract::fixed(vec![ValueFact::scalar(ValueKindFact::Logical)]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    if !diagnostics.is_empty() {
        result.outputs.iter_mut().for_each(|output| {
            *output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
        });
    }
    result.diagnostics.extend(diagnostics);
    result
}
