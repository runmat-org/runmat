use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-ADDPROP-ARITY",
            "expected object and property name",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if let Some(receiver) = request.arguments.first() {
        if !matches!(
            &receiver.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Object(ObjectFact {
                    handle_semantics: None | Some(true),
                    ..
                })
        ) {
            diagnostics.push(argument_error(
                "RM-ADDPROP-TARGET",
                "expected a handle object",
                0,
            ));
        }
    }
    if let Some(name) = request.arguments.get(1) {
        if !matches!(
            name.kind,
            ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
        ) || name.shape.is_definitely_nonscalar()
        {
            diagnostics.push(argument_error(
                "RM-ADDPROP-NAME",
                "expected scalar text property name",
                1,
            ));
        }
    }
    let output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(ClassIdentity::from("matlab.metadata.DynamicProperty")),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(true),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
