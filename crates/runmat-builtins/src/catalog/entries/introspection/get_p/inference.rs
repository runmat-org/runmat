use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-INTROSPECTION-ARITY",
            "expected 1 inputs",
            0,
        ));
    }

    let mut output = match request.arguments.first().map(|a| &a.kind) {
        Some(ValueKindFact::Object(object)) if object.handle_semantics == Some(false) => object
            .properties
            .get("p_backing")
            .cloned()
            .unwrap_or_else(|| {
                if object.properties_complete {
                    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real,
                    }))
                } else {
                    ValueFact::unknown(DynamicReason::RuntimeValue)
                }
            }),
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    // Returning a stored value can share backing storage with the receiver.
    output.alias = AliasFact::Unknown;
    validate_receiver(request, &mut diagnostics);
    let mut contract = CallContract::fixed(vec![output]);

    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn validate_receiver(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if let Some(input) = request.arguments.first() {
        if !matches!(
            &input.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Object(ObjectFact {
                    handle_semantics: None | Some(false),
                    ..
                })
        ) {
            diagnostics.push(argument_error(
                "RM-PROPERTY-RECEIVER",
                "expected a value-object receiver",
                0,
            ));
        }
    }
}
