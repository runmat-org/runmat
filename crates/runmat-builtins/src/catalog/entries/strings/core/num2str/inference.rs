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
    if !(1..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-NUM2STR-ARITY",
            "num2str accepts one to three inputs",
            0,
        ));
    }
    if let Some(input) = request.arguments.first() {
        if !matches!(
            &input.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-NUM2STR-NUMERIC",
                "num2str requires numeric or logical data",
                0,
            ));
        }
    }
    if let Some(option) = request.arguments.get(1) {
        if !matches!(
            &option.kind,
            ValueKindFact::Numeric(_)
                | ValueKindFact::Logical
                | ValueKindFact::Character
                | ValueKindFact::String
                | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-NUM2STR-OPTION",
                "num2str requires precision, a format string, or 'local'",
                1,
            ));
        }
    }
    if request.arguments.len() >= 3 {
        if let Some(literal) = request.literals.literal_args.get(2) {
            if let Some(text) = literal_text(literal) {
                if !text.trim().eq_ignore_ascii_case("local") {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NUM2STR-LOCAL",
                        "the third argument must be 'local'",
                        2,
                    ));
                }
            }
        }
        if let Some(option) = request.arguments.get(2) {
            if !matches!(
                &option.kind,
                ValueKindFact::Character | ValueKindFact::String | ValueKindFact::Unknown
            ) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-NUM2STR-LOCAL",
                    "the third argument must be 'local'",
                    2,
                ));
            }
        }
    }
    if let Some(literal) = request.literals.literal_args.get(1) {
        if let Some(valid) = literal_precision_is_valid(literal) {
            if !valid {
                diagnostics.push(argument_error(
                    "RM-CATALOG-NUM2STR-PRECISION",
                    "precision must be an integer from 0 through 52",
                    1,
                ));
            }
        } else if let Some(text) = literal_text(literal) {
            if !text.trim().eq_ignore_ascii_case("local") && !text.starts_with('%') {
                diagnostics.push(argument_error(
                    "RM-CATALOG-NUM2STR-FORMAT",
                    "format must start with '%'",
                    1,
                ));
            }
        }
    }
    let shape = match request.arguments.first() {
        Some(input)
            if matches!(
                &input.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical
            ) =>
        {
            if let Some(dims) = input.shape.known_dims() {
                if dims.len() > 2 {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NUM2STR-DIMENSIONS",
                        "num2str accepts at most two dimensions",
                        0,
                    ));
                    ShapeFact::Unknown
                } else {
                    let rows = dims.first().copied().flatten();
                    let cols = dims.get(1).copied().flatten();
                    match (rows, cols) {
                        (Some(0), _) => ShapeFact::from(vec![Some(0), Some(0)]),
                        (Some(rows), Some(0)) => ShapeFact::from(vec![Some(rows), Some(0)]),
                        _ => ShapeFact::from(vec![rows, None]),
                    }
                }
            } else {
                ShapeFact::Unknown
            }
        }
        _ => ShapeFact::Unknown,
    };
    let mut output = ValueFact::proven(ValueKindFact::Character, shape, StorageFact::Dense);
    if request
        .arguments
        .first()
        .is_none_or(|input| matches!(&input.kind, ValueKindFact::Unknown))
    {
        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn literal_text(value: &LiteralValue) -> Option<&str> {
    match value {
        LiteralValue::String(text)
        | LiteralValue::Character(text)
        | LiteralValue::Keyword(text) => Some(text),
        _ => None,
    }
}

fn literal_precision_is_valid(value: &LiteralValue) -> Option<bool> {
    match value {
        LiteralValue::Number(n) => {
            Some(n.is_finite() && n.fract() == 0.0 && (0.0..=52.0).contains(n))
        }
        LiteralValue::Real { text, .. } => text
            .parse::<f64>()
            .ok()
            .map(|n| n.is_finite() && n.fract() == 0.0 && (0.0..=52.0).contains(&n)),
        LiteralValue::Integer { text, .. } => Some(
            text.parse::<i128>()
                .map(|n| (0..=52).contains(&n))
                .unwrap_or(false),
        ),
        LiteralValue::Bool(_) => Some(true),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection};

    fn call(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallInference {
        infer(
            &CallRequest {
                arguments,
                literals: LiteralContext::new(literals),
                outputs: OutputSelection::default(),
            },
            &super::super::NUM2STR_CATALOG_ENTRY,
        )
    }
    fn numeric(shape: ShapeFact) -> ValueFact {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            shape,
            StorageFact::Dense,
        )
    }
    #[test]
    fn numeric_rows_are_known_but_formatted_width_is_dynamic() {
        let result = call(
            vec![numeric(ShapeFact::from(vec![Some(2), Some(3)]))],
            vec![],
        );
        assert_eq!(result.outputs[0].kind, ValueKindFact::Character);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), None])
        );
        let empty_rows = call(
            vec![numeric(ShapeFact::from(vec![Some(0), Some(3)]))],
            vec![],
        );
        assert_eq!(
            empty_rows.outputs[0].shape,
            ShapeFact::from(vec![Some(0), Some(0)])
        );
        let empty_cols = call(
            vec![numeric(ShapeFact::from(vec![Some(3), Some(0)]))],
            vec![],
        );
        assert_eq!(
            empty_cols.outputs[0].shape,
            ShapeFact::from(vec![Some(3), Some(0)])
        );
    }
    #[test]
    fn options_and_unknown_input_are_checked_without_guessing_width() {
        let input = numeric(ShapeFact::from(vec![Some(1), Some(1)]));
        let bad_precision = call(
            vec![
                input.clone(),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            ],
            vec![LiteralValue::Unknown, LiteralValue::Number(53.0)],
        );
        assert!(bad_precision
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-NUM2STR-PRECISION"));
        let bad_local = call(
            vec![
                input,
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("%.2f".into()),
                LiteralValue::String("not-local".into()),
            ],
        );
        assert!(bad_local
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-NUM2STR-LOCAL"));
        let unknown = call(
            vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
            vec![],
        );
        assert_eq!(unknown.outputs[0].shape, ShapeFact::Unknown);
        assert!(matches!(
            unknown.outputs[0].certainty,
            CertaintyFact::Dynamic(_)
        ));
    }
}
