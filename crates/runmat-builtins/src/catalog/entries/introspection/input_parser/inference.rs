use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

#[derive(Clone, Copy)]
enum Operation {
    Construct,
    AddParameter,
    Parse,
}

pub(super) fn construct(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, Operation::Construct)
}
pub(super) fn add_parameter(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, Operation::AddParameter)
}
pub(super) fn parse(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, Operation::Parse)
}

fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    operation: Operation,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let valid_arity = match operation {
        Operation::Construct => request.arguments.is_empty(),
        Operation::AddParameter => request.arguments.len() == 3,
        Operation::Parse => {
            !request.arguments.is_empty() && (request.arguments.len() - 1).is_multiple_of(2)
        }
    };
    if !valid_arity {
        diagnostics.push(argument_error(
            "RM-INPUTPARSER-ARITY",
            "inputParser operation has invalid input arity",
            0,
        ));
    }
    let output = if matches!(operation, Operation::Construct) {
        ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(StaticClassIdentity::new("inputParser").owned()),
                properties: Default::default(),
                properties_complete: false,
                handle_semantics: Some(true),
            }),
            ShapeFact::Scalar,
            StorageFact::Opaque,
        )
    } else {
        let mut receiver = request
            .arguments
            .first()
            .cloned()
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
        if let ValueKindFact::Object(object) = &mut receiver.kind {
            // Both operations mutate the aliased handle. Old property facts
            // must not describe the post-publication Results generation.
            object.properties.clear();
            object.properties_complete = false;
        }
        receiver
    };
    finish_fixed(entry, request, output, diagnostics)
}
