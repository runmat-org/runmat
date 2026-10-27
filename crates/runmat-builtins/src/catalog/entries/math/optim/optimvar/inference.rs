use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry, OptimizationVariableProperty,
};
use runmat_types::*;
pub(super) fn constructor(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, Role::Constructor)
}
pub(super) fn read(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, Role::Read)
}
pub(super) fn write(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, Role::Write)
}
#[derive(Clone, Copy)]
enum Role {
    Constructor,
    Read,
    Write,
}
fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry, role: Role) -> CallInference {
    let mut diagnostics = Vec::new();
    let output = if matches!(role, Role::Constructor) {
        if request.arguments.is_empty() {
            diagnostics.push(argument_error(
                "RM-CATALOG-OPTIMVAR-NAME",
                "expected a variable name",
                0,
            ));
        }
        if request.arguments.first().is_some_and(|value| {
            !matches!(
                value.kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            )
        }) {
            diagnostics.push(argument_error(
                "RM-CATALOG-OPTIMVAR-NAME",
                "name must be scalar text",
                0,
            ));
        }
        let end = (1..request.arguments.len())
            .find(|&index| {
                request
                    .literals
                    .literal_string_at(index)
                    .as_deref()
                    .and_then(OptimizationVariableProperty::parse_name_value)
                    .is_some()
            })
            .unwrap_or(request.arguments.len());
        let mut dimensions = Vec::new();
        for index in 1..end {
            if let Some(number) = request.literals.numeric_at(index) {
                match decode_shape_dimension(number) {
                    Ok(n) if n > 0 => dimensions.push(Some(n)),
                    _ => diagnostics.push(argument_error(
                        "RM-CATALOG-OPTIMVAR-DIMENSION",
                        "dimensions must be finite positive integers",
                        index,
                    )),
                }
            } else {
                break;
            }
        }
        let shape = if request.arguments.is_empty() {
            ShapeFact::Unknown
        } else if end == 1 {
            ShapeFact::Scalar
        } else if dimensions.len() == end - 1 {
            if dimensions.len() == 1 {
                dimensions.push(Some(1));
            }
            ShapeFact::from(dimensions)
        } else {
            ShapeFact::Unknown
        };
        // Never claim a numeric result from a symbolic variable object.
        let mut value = ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(standard::OPTIMIZATION_VARIABLE.owned()),
                properties: std::collections::BTreeMap::new(),
                properties_complete: false,
                handle_semantics: Some(true),
            }),
            shape,
            StorageFact::Opaque,
        );
        value.alias = AliasFact::Identity;
        value.mutation = MutationFact::HandleSemantics;
        value
    } else {
        let expected = if matches!(role, Role::Read) { 2 } else { 3 };
        if request.arguments.len() != expected {
            diagnostics.push(argument_error(
                "RM-CATALOG-OPTIMVAR-PROTOCOL",
                "invalid class protocol arity",
                0,
            ));
        }
        if request.arguments.first().is_some_and(|value| {
            !matches!(
                value.kind,
                ValueKindFact::Object(_) | ValueKindFact::Unknown
            )
        }) {
            diagnostics.push(argument_error(
                "RM-CATALOG-OPTIMVAR-REFERENCE",
                "expected an optimization-variable reference",
                0,
            ));
        }
        if expected == 3 {
            request
                .arguments
                .first()
                .cloned()
                .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue))
        } else {
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
    };
    finish_fixed(entry, request, output, diagnostics)
}
