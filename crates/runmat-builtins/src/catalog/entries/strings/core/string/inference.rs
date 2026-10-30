use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CertaintyFact, DynamicReason, LiteralValue, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-STRING-ARITY",
            "string requires input data or a format specification",
            0,
        ));
    }
    let first = request.arguments.first();
    let format_call = request.arguments.len() > 1
        && request
            .literals
            .literal_args
            .first()
            .is_some_and(is_format_literal);
    let encoding_call = request.arguments.len() == 2
        && request
            .literals
            .literal_string_at(1)
            .is_some_and(|s| matches!(s.as_str(), "utf-8" | "utf8" | "system"));
    if request.arguments.len() > 1
        && !format_call
        && !encoding_call
        && request.literals.literal_args.get(1).is_some()
        && !matches!(first.map(|v| &v.kind), Some(ValueKindFact::Unknown))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-STRING-OPTIONS",
            "additional arguments must select supported encoding or formatting",
            1,
        ));
    }
    if let Some(input) = first {
        if matches!(
            input.kind,
            ValueKindFact::Symbolic
                | ValueKindFact::Callable(_)
                | ValueKindFact::Execution(_)
                | ValueKindFact::Foreign(_)
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-STRING-INPUT",
                "string cannot convert this input class",
                0,
            ));
        }
    }
    let shape = if request.arguments.len() > 1 && !encoding_call {
        // A nonliteral format specification can broadcast its data. Argument
        // shape alone does not distinguish that form from a conversion option.
        ShapeFact::Unknown
    } else {
        match first {
            Some(input) if matches!(input.kind, ValueKindFact::Character) => {
                input.shape.known_dims().map_or(ShapeFact::Unknown, |dims| {
                    ShapeFact::from(vec![
                        dims.first().copied().flatten().map(|rows| rows.max(1)),
                        Some(1),
                    ])
                })
            }
            Some(input) if matches!(input.kind, ValueKindFact::Unknown) => ShapeFact::Unknown,
            Some(input) => input.shape.clone(),
            None => ShapeFact::Unknown,
        }
    };
    let mut output = ValueFact::proven(ValueKindFact::String, shape, StorageFact::Dense);
    if first.is_none_or(|input| matches!(input.kind, ValueKindFact::Unknown))
        || (request.arguments.len() > 1 && !encoding_call)
    {
        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn is_format_literal(value: &LiteralValue) -> bool {
    match value {
        LiteralValue::String(s) | LiteralValue::Character(s) | LiteralValue::Keyword(s) => {
            s.contains('%')
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection};
    fn call(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallInference {
        infer(
            &CallRequest {
                arguments,
                literals: LiteralContext::new(literals),
                outputs: OutputSelection::default(),
            },
            &super::super::STRING_CATALOG_ENTRY,
        )
    }
    #[test]
    fn numeric_shape_is_preserved_and_character_rows_become_string_column() {
        let numeric = ValueFact::proven(
            ValueKindFact::Logical,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        assert_eq!(
            call(vec![numeric], vec![]).outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
        let chars = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(4), Some(6)]),
            StorageFact::Dense,
        );
        assert_eq!(
            call(vec![chars], vec![]).outputs[0].shape,
            ShapeFact::from(vec![Some(4), Some(1)])
        );
    }
    #[test]
    fn formatting_shape_and_unknown_input_remain_dynamic() {
        let text = ValueFact::scalar(ValueKindFact::String);
        let data = ValueFact::proven(
            ValueKindFact::Logical,
            ShapeFact::from(vec![Some(1), Some(4)]),
            StorageFact::Dense,
        );
        assert_eq!(
            call(
                vec![text, data],
                vec![LiteralValue::String("Trial %d".into())]
            )
            .outputs[0]
                .shape,
            ShapeFact::Unknown
        );
        assert!(matches!(
            call(
                vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
                vec![]
            )
            .outputs[0]
                .certainty,
            CertaintyFact::Dynamic(_)
        ));
    }
}
