use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-INPUTNAME-ARITY",
            "expected one argument index",
            0,
        ));
    }
    if let Some(index) = request.arguments.first() {
        if matches!(
            index.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        ) {
            diagnostics.push(argument_error(
                "RM-INPUTNAME-RESIDENCY",
                "argument index must be host-resident",
                0,
            ));
        }
        if !matches!(
            index.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Unknown
        ) || index.shape.is_definitely_nonscalar()
        {
            diagnostics.push(argument_error(
                "RM-INPUTNAME-INDEX",
                "expected a positive numeric scalar index",
                0,
            ));
        }
    }
    // Length and the empty-name case depend on the caller's source expression.
    let output = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![None, None]),
        StorageFact::Dense,
    );
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
