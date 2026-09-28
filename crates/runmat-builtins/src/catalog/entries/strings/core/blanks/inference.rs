use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CertaintyFact, DynamicReason, LiteralValue, NumericDomain,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-BLANKS-ARITY",
            "blanks requires exactly one input",
            0,
        ));
    }
    if let Some(input) = request.arguments.first() {
        let valid_kind = matches!(&input.kind, ValueKindFact::Numeric(fact) if fact.domain == NumericDomain::Real)
            || matches!(&input.kind, ValueKindFact::Unknown);
        if !valid_kind || input.shape.element_count().is_some_and(|count| count != 1) {
            diagnostics.push(argument_error(
                "RM-CATALOG-BLANKS-LENGTH",
                "n must be a real integer-valued scalar",
                0,
            ));
        }
    }
    let width = request
        .literals
        .literal_args
        .first()
        .and_then(literal_width);
    let shape = ShapeFact::from(vec![Some(1), width]);
    let mut output = ValueFact::proven(ValueKindFact::Character, shape, StorageFact::Dense);
    if width.is_none() {
        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn literal_width(value: &LiteralValue) -> Option<usize> {
    match value {
        LiteralValue::Integer { text, .. } => {
            if let Ok(signed) = text.parse::<i128>() {
                if signed <= 0 {
                    Some(0)
                } else {
                    usize::try_from(signed).ok()
                }
            } else {
                text.parse::<u128>()
                    .ok()
                    .and_then(|n| usize::try_from(n).ok())
            }
        }
        LiteralValue::Number(n) if n.is_finite() && n.fract() == 0.0 => {
            if *n <= 0.0 {
                Some(0)
            } else if *n <= usize::MAX as f64 && *n < 9_007_199_254_740_992.0 {
                Some(*n as usize)
            } else {
                None
            }
        }
        LiteralValue::Real { text, .. } => text
            .parse::<f64>()
            .ok()
            .and_then(|n| literal_width(&LiteralValue::Number(n))),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, NumericClass, NumericFact, OutputSelection};
    fn request(input: ValueFact, literal: LiteralValue) -> CallRequest {
        CallRequest {
            arguments: vec![input],
            literals: LiteralContext::new(vec![literal]),
            outputs: OutputSelection::default(),
        }
    }
    #[test]
    fn exact_integer_width_and_negative_clamp_are_inferred() {
        let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }));
        let entry = &super::super::BLANKS_CATALOG_ENTRY;
        let output = infer(
            &request(
                numeric.clone(),
                LiteralValue::Integer {
                    text: "5".into(),
                    class: NumericClass::UInt64,
                },
            ),
            entry,
        );
        assert_eq!(
            output.outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(5)])
        );
        let negative = infer(
            &request(
                numeric,
                LiteralValue::Integer {
                    text: "-2".into(),
                    class: NumericClass::Int64,
                },
            ),
            entry,
        );
        assert_eq!(
            negative.outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(0)])
        );
    }

    #[test]
    fn unknown_width_stays_dynamic_and_known_bad_shape_is_diagnosed() {
        let entry = &super::super::BLANKS_CATALOG_ENTRY;
        let unknown = infer(
            &request(
                ValueFact::unknown(DynamicReason::RuntimeValue),
                LiteralValue::String("dynamic".into()),
            ),
            entry,
        );
        assert_eq!(
            unknown.outputs[0].shape,
            ShapeFact::from(vec![Some(1), None])
        );
        assert!(matches!(
            unknown.outputs[0].certainty,
            CertaintyFact::Dynamic(_)
        ));

        let nonscalar = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        );
        let bad = infer(&request(nonscalar, LiteralValue::Number(2.0)), entry);
        assert!(bad
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-BLANKS-LENGTH"));
    }
}
