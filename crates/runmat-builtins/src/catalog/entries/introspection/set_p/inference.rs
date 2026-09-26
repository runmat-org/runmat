use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-INTROSPECTION-ARITY",
            "expected 2 inputs",
            0,
        ));
    }

    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    if let (Some(receiver), Some(value)) = (request.arguments.first(), request.arguments.get(1)) {
        if let ValueKindFact::Object(object) = &receiver.kind {
            if object.handle_semantics != Some(true) {
                let mut object = object.clone();
                object.handle_semantics = Some(false);
                object.properties.insert("p_backing".into(), value.clone());
                output = ValueFact::proven(
                    ValueKindFact::Object(object),
                    ShapeFact::Scalar,
                    StorageFact::Opaque,
                );
                output.alias = AliasFact::Unknown;
            }
        }
    }
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
