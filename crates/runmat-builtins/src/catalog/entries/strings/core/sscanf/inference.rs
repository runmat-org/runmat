use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, NumericClass,
    NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn double_scalar() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

fn infer_outputs(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    outputs: Vec<ValueFact>,
    diagnostics: Vec<runmat_types::InferenceDiagnostic>,
) -> CallInference {
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(diagnostics);
    inferred
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SSCANF-ARITY",
            "sscanf accepts text, format, and optional size",
            0,
        ));
    }
    if let Some(input) = request.arguments.first() {
        if !matches!(
            input.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-SSCANF-TEXT",
                "sscanf requires scalar text",
                0,
            ));
        }
    }
    let scanned = if request.arguments.len() == 1 {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::Unknown,
            StorageFact::Dense,
        )
    } else {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    infer_outputs(
        request,
        entry,
        vec![
            scanned,
            double_scalar(),
            ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::Unknown,
                StorageFact::Dense,
            ),
            double_scalar(),
        ],
        diagnostics,
    )
}
