use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    });
    output.residency = ResidencyFact::Host;
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-MAGIC-ARG-COUNT",
            "magic requires one input",
            0,
        ));
    } else {
        match request.literals.literal_args.first().map(order) {
            Some(Ok(Some(n))) => {
                output.shape = ShapeFact::from(vec![Some(n), Some(n)]);
                output.storage = if n == 1 {
                    StorageFact::Scalar
                } else {
                    StorageFact::Dense
                };
            }
            Some(Err(())) => diagnostics.push(argument_error(
                "RM-MAGIC-ORDER",
                "magic requires a finite nonnegative representable order",
                0,
            )),
            _ => {}
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn order(value: &LiteralValue) -> Result<Option<usize>, ()> {
    match value {
        LiteralValue::Integer { text, .. } => text
            .parse::<u64>()
            .ok()
            .and_then(|n| usize::try_from(n).ok())
            .map(Some)
            .ok_or(()),
        LiteralValue::Number(value) => floating(*value),
        LiteralValue::Real { text, .. } | LiteralValue::Complex { real: text, .. } => {
            floating(text.parse().map_err(|_| ())?)
        }
        LiteralValue::Bool(value) => Ok(Some(usize::from(*value))),
        LiteralValue::Character(value) => value.chars().next().map(|c| Some(c as usize)).ok_or(()),
        LiteralValue::Vector(values) => values.first().ok_or(()).and_then(order),
        LiteralValue::Matrix(rows) => rows
            .first()
            .and_then(|r| r.first())
            .ok_or(())
            .and_then(order),
        LiteralValue::Empty | LiteralValue::String(_) => Err(()),
        _ => Ok(None),
    }
}

fn floating(value: f64) -> Result<Option<usize>, ()> {
    let n = value.floor();
    let fits_usize = n < usize::MAX as f64 || (usize::BITS < 64 && n == usize::MAX as f64);
    if !n.is_finite() || n < 0.0 || !fits_usize {
        return Err(());
    }
    Ok(Some(n as usize))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn infer_literal(literal: LiteralValue) -> CallInference {
        infer(
            &CallRequest {
                arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
                literals: LiteralContext::new(vec![literal]),
                outputs: OutputSelection::default(),
            },
            &super::super::MAGIC_CATALOG_ENTRY,
        )
    }
    #[test]
    fn order_comes_from_first_value_not_input_shape() {
        assert_eq!(
            infer_literal(LiteralValue::Vector(vec![
                LiteralValue::Number(3.9),
                LiteralValue::Number(10.0)
            ]))
            .outputs[0]
                .shape,
            ShapeFact::from(vec![Some(3), Some(3)])
        );
        assert_eq!(
            infer_literal(LiteralValue::Unknown).outputs[0].shape,
            ShapeFact::Unknown
        );
        assert_eq!(
            infer_literal(LiteralValue::Number(0.0)).outputs[0]
                .shape
                .element_count(),
            Some(0)
        );
        assert!(!infer_literal(LiteralValue::Number(-1.0))
            .diagnostics
            .is_empty());
        assert!(!infer_literal(LiteralValue::Empty).diagnostics.is_empty());
    }
    #[test]
    fn integer_order_never_rounds_through_double() {
        assert_eq!(floating(usize::MAX as f64).is_ok(), usize::BITS < 64);
        let exact = 9_007_199_254_740_993_u64;
        let result = order(&LiteralValue::Integer {
            text: exact.to_string(),
            class: NumericClass::UInt64,
        });
        assert_eq!(result, usize::try_from(exact).map(Some).map_err(|_| ()));
    }
}
