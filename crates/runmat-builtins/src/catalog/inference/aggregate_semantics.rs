use super::{argument_error, finish_fixed, literal_text};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer_struct_builtin(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let output = match request.arguments.as_slice() {
        [] => struct_fact(BTreeMap::new(), true),
        [template] => match template.kind {
            ValueKindFact::Struct(_) => template.clone(),
            ValueKindFact::Unknown => ValueFact::unknown(DynamicReason::RuntimeValue),
            _ if template.shape.element_count() == Some(0) => struct_fact(BTreeMap::new(), true),
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-STRUCT-TEMPLATE",
                    "single-argument struct requires a struct or empty array template",
                    0,
                ));
                ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
            }
        },
        arguments if arguments.len() % 2 == 0 => {
            let mut fields = BTreeMap::new();
            let mut complete = true;
            for index in (0..arguments.len()).step_by(2) {
                if let Some(name) = request
                    .literals
                    .literal_args
                    .get(index)
                    .and_then(literal_text)
                {
                    fields.insert(name, arguments[index + 1].clone());
                } else {
                    complete = false;
                }
            }
            struct_fact(fields, complete)
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-STRUCT-PAIRS",
                "struct requires complete field/value pairs",
                request.arguments.len().saturating_sub(1),
            ));
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
        }
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn struct_fact(fields: BTreeMap<String, ValueFact>, fields_complete: bool) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Struct(StructFact {
        fields,
        fields_complete,
    }))
}
