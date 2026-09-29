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
    let args = &request.arguments;
    let literals = &request.literals.literal_args;
    let mut size_literals = Vec::new();
    let mut prototype = None;
    let mut i = 0;
    while i < args.len() {
        match request.literals.literal_string_at(i).as_deref() {
            Some("like") => {
                if prototype.is_some() {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-STRINGS-LIKE",
                        "only one like prototype is accepted",
                        i,
                    ));
                }
                prototype = args.get(i + 1);
                if prototype.is_none() {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-STRINGS-LIKE",
                        "like requires a prototype",
                        i,
                    ));
                }
                i += 2;
            }
            Some("missing" | "empty") => i += 1,
            _ => {
                if !matches!(
                    args[i].kind,
                    ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
                ) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-STRINGS-SIZE",
                        "size inputs must be numeric or logical",
                        i,
                    ));
                }
                size_literals.push(literals.get(i));
                i += 1;
            }
        }
    }
    let shape = if size_literals.is_empty() {
        prototype.map_or_else(
            || ShapeFact::from(vec![Some(1), Some(1)]),
            |p| p.shape.clone(),
        )
    } else {
        inferred_dims(&size_literals)
    };
    let mut output = ValueFact::proven(ValueKindFact::String, shape, StorageFact::Dense);
    if output.shape.known_dims().is_none()
        || args
            .iter()
            .any(|a| matches!(a.kind, ValueKindFact::Unknown))
    {
        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn inferred_dims(literals: &[Option<&LiteralValue>]) -> ShapeFact {
    let mut dims = if literals.len() == 1 {
        match literals[0] {
            Some(LiteralValue::Vector(values)) => values.iter().map(dimension).collect::<Vec<_>>(),
            Some(value) => {
                let n = dimension(value);
                vec![n, n]
            }
            None => return ShapeFact::Unknown,
        }
    } else {
        literals.iter().map(|v| v.and_then(dimension)).collect()
    };
    if dims.is_empty() {
        return ShapeFact::from(vec![Some(0), Some(0)]);
    }
    if dims.len() == 1 {
        dims.push(dims[0]);
    }
    while dims.len() > 2 && dims.last() == Some(&Some(1)) {
        dims.pop();
    }
    ShapeFact::from(dims)
}

fn dimension(value: &LiteralValue) -> Option<usize> {
    match value {
        LiteralValue::Bool(b) => Some(usize::from(*b)),
        LiteralValue::Integer { text, .. } => {
            if text.starts_with('-') {
                Some(0)
            } else {
                text.parse::<usize>().ok()
            }
        }
        LiteralValue::Number(n)
            if n.is_finite() && n.fract() == 0.0 && n.abs() < 9_007_199_254_740_992.0 =>
        {
            if *n <= 0.0 {
                Some(0)
            } else {
                Some(*n as usize)
            }
        }
        LiteralValue::Real { text, .. } => text
            .parse::<f64>()
            .ok()
            .and_then(|n| dimension(&LiteralValue::Number(n))),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection};
    fn numeric() -> ValueFact {
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }))
    }
    fn call(args: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallInference {
        infer(
            &CallRequest {
                arguments: args,
                literals: LiteralContext::new(literals),
                outputs: OutputSelection::default(),
            },
            &super::super::STRINGS_CATALOG_ENTRY,
        )
    }
    #[test]
    fn scalar_vector_and_negative_sizes_track_runtime() {
        assert_eq!(
            call(vec![], vec![]).outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(1)])
        );
        assert_eq!(
            call(vec![numeric()], vec![LiteralValue::Number(3.0)]).outputs[0].shape,
            ShapeFact::from(vec![Some(3), Some(3)])
        );
        assert_eq!(
            call(
                vec![numeric()],
                vec![LiteralValue::Vector(vec![
                    LiteralValue::Number(0.0),
                    LiteralValue::Number(5.0)
                ])]
            )
            .outputs[0]
                .shape,
            ShapeFact::from(vec![Some(0), Some(5)])
        );
        assert_eq!(
            call(vec![numeric()], vec![LiteralValue::Number(-3.0)]).outputs[0].shape,
            ShapeFact::from(vec![Some(0), Some(0)])
        );
    }
    #[test]
    fn size_vector_shape_is_not_its_contents() {
        let vector = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        );
        assert_eq!(
            call(vec![vector], vec![]).outputs[0].shape,
            ShapeFact::Unknown
        );
    }
}
