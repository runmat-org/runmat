use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-NATIVE2UNICODE-ARITY",
            "native2unicode accepts bytes and optional encoding",
            0,
        ));
    }
    let output = match request.arguments.first().map(|v| &v.kind) {
        Some(ValueKindFact::String) => ValueFact::proven(
            ValueKindFact::String,
            request.arguments[0].shape.clone(),
            StorageFact::Dense,
        ),
        Some(ValueKindFact::Character) => ValueFact::proven(
            ValueKindFact::Character,
            request.arguments[0].shape.clone(),
            StorageFact::Dense,
        ),
        Some(ValueKindFact::Numeric(_)) => ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::Unknown,
            StorageFact::Dense,
        ),
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed(entry, request, output, diagnostics)
}
