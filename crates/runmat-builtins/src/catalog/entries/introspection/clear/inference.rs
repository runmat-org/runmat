use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer_clear(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(0), Some(0)]),
        StorageFact::Dense,
    );
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    for (index, arg) in request.arguments.iter().enumerate() {
        if !matches!(
            arg.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) || matches!(arg.kind, ValueKindFact::Character)
            && !crate::catalog::inference::possible_character_row(&arg.shape)
        {
            result.diagnostics.push(argument_error(
                "RM.CLEAR.NAME_ARG_TYPE",
                "expected text target",
                index,
            ));
        }
    }
    result
}
