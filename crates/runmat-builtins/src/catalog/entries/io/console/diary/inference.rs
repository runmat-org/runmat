use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    CallInference, CallRequest, CapabilityRequirement, LiteralValue, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut contract = runmat_types::CallContract::fixed(vec![]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = runmat_types::infer_call(&contract, request);
    if request.arguments.len() > 1 {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-DIARY-ARITY",
            "diary accepts at most one argument",
            1,
        ));
    }
    if let Some(value) = request.arguments.first() {
        if definitely_nontext(value) {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-DIARY-TEXT",
                "diary argument must be a text scalar",
                0,
            ));
        } else if let Some(LiteralValue::String(text) | LiteralValue::Character(text)) =
            request.literals.literal_args.first()
        {
            match super::DiaryCommand::decode(text) {
                None => result.diagnostics.push(argument_error(
                    "RM-CATALOG-DIARY-EMPTY",
                    "diary argument cannot be empty",
                    0,
                )),
                Some(super::DiaryCommand::Off) => {
                    result
                        .effects
                        .0
                        .remove(&runmat_types::EffectKind::FilesystemWrite);
                    result
                        .capabilities
                        .0
                        .remove(&CapabilityRequirement::Filesystem);
                }
                Some(_) => {}
            }
        }
    }
    result
}

fn definitely_nontext(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Character => value.shape.known_dims().is_some_and(|dims| {
            dims.first()
                .copied()
                .flatten()
                .is_some_and(|rows| rows != 1)
                || dims
                    .iter()
                    .skip(2)
                    .any(|extent| extent.is_some_and(|n| n > 1))
        }),
        ValueKindFact::String => value.shape.element_count().is_some_and(|count| count != 1),
        ValueKindFact::Unknown => false,
        _ => true,
    }
}
