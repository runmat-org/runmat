use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-UNICODE2NATIVE-ARITY",
            "unicode2native accepts text and optional encoding",
            0,
        ));
    }
    if let Some(input) = request.arguments.first() {
        if !matches!(
            input.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-UNICODE2NATIVE-TEXT",
                "unicode2native requires scalar text",
                0,
            ));
        }
    }
    finish_fixed(
        entry,
        request,
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt8,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(1), None]),
            StorageFact::Dense,
        ),
        diagnostics,
    )
}
