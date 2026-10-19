use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, NumericClass,
    NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-PADE-ARITY",
            "pade expects a scalar delay and a positive integer order",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count != 2)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-PADE-OUTPUTS",
            "the supported coefficient form requires exactly two outputs",
            0,
        ));
    }
    for (index, argument) in request.arguments.iter().take(2).enumerate() {
        let valid = match &argument.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Numeric(numeric) => {
                numeric.domain == NumericDomain::Real
                    && (matches!(numeric.class, NumericClass::Single | NumericClass::Double)
                        || (index == 1
                            && matches!(
                                numeric.class,
                                NumericClass::Int8
                                    | NumericClass::Int16
                                    | NumericClass::Int32
                                    | NumericClass::Int64
                                    | NumericClass::UInt8
                                    | NumericClass::UInt16
                                    | NumericClass::UInt32
                                    | NumericClass::UInt64
                            )))
            }
            _ => false,
        };
        if !valid
            || argument
                .shape
                .element_count()
                .is_some_and(|count| count != 1)
            || !matches!(
                argument.residency,
                ResidencyFact::Host | ResidencyFact::Unknown
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-PADE-SCALAR",
                "T must be a host real floating scalar; N may also be a typed integer scalar",
                index,
            ));
        }
    }
    let mut output = match request.arguments.first().map(|argument| &argument.kind) {
        Some(ValueKindFact::Numeric(numeric))
            if numeric.domain == NumericDomain::Real
                && matches!(numeric.class, NumericClass::Single | NumericClass::Double) =>
        {
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: numeric.class,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(1), None]),
                StorageFact::Dense,
            )
        }
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    output.shape = ShapeFact::from(vec![Some(1), None]);
    output.storage = StorageFact::Dense;
    output.residency = ResidencyFact::Host;
    let mut contract = CallContract::fixed(vec![output.clone(), output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
