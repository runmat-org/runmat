//! Text-specific addition facts, separate from numeric operator inference.
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CertaintyFact, DynamicReason, ResidencyFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) fn infer_text(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> Option<CallInference> {
    if !request
        .arguments
        .iter()
        .take(2)
        .any(|argument| matches!(argument.kind, ValueKindFact::String))
    {
        return None;
    }
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-PLUS-TEXT-ARITY",
            "string addition requires exactly two operands; numeric output prototypes do not apply",
            2,
        ));
    }
    for (index, argument) in request.arguments.iter().take(2).enumerate() {
        if !matches!(
            argument.kind,
            ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Logical
                | ValueKindFact::Numeric(_)
                | ValueKindFact::Cell(_)
                | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-PLUS-TEXT-INPUT",
                "operand cannot be converted to a string array",
                index,
            ));
        }
        if matches!(argument.residency, ResidencyFact::Device { .. }) {
            diagnostics.push(argument_error(
                "RM-CATALOG-PLUS-TEXT-DEVICE",
                "string addition requires host values; gather device data explicitly",
                index,
            ));
        }
    }
    let shape = match request.arguments.first().zip(request.arguments.get(1)) {
        Some((left, right)) => match runmat_types::rules::broadcast_shape_checked(
            &converted_shape(left),
            &converted_shape(right),
        ) {
            Ok(shape) => shape,
            Err(error) => {
                diagnostics.push(error.into_diagnostic());
                ShapeFact::Unknown
            }
        },
        None => ShapeFact::Unknown,
    };
    let mut output = ValueFact::proven(ValueKindFact::String, shape, StorageFact::Dense);
    output.residency = ResidencyFact::Host;
    if !diagnostics.is_empty() {
        output.certainty = CertaintyFact::Dynamic(DynamicReason::UnsupportedRepresentation);
    } else if let Some(reason) =
        request
            .arguments
            .iter()
            .take(2)
            .find_map(|argument| match &argument.certainty {
                CertaintyFact::Dynamic(reason) => Some(reason.clone()),
                _ if argument.kind == ValueKindFact::Unknown => Some(DynamicReason::RuntimeValue),
                _ => None,
            })
    {
        output.certainty = CertaintyFact::Dynamic(reason);
    }
    Some(finish_fixed(entry, request, output, diagnostics))
}

fn converted_shape(value: &ValueFact) -> ShapeFact {
    match value.kind {
        ValueKindFact::Character => {
            value
                .shape
                .known_dims()
                .map_or(ShapeFact::Unknown, |dimensions| {
                    ShapeFact::from(vec![
                        dimensions
                            .first()
                            .copied()
                            .flatten()
                            .map(|rows| rows.max(1)),
                        Some(1),
                    ])
                })
        }
        ValueKindFact::Unknown => ShapeFact::Unknown,
        _ => value.shape.clone(),
    }
}
