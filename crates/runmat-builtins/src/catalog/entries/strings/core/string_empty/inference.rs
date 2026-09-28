use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CertaintyFact, DynamicReason, LiteralValue, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let args = &request.arguments;
    let literals = &request.literals.literal_args;
    let mut diagnostics = Vec::new();
    let mut sizes = Vec::new();
    let mut prototype = None;
    let mut i = 0;
    while i < args.len() {
        if request.literals.literal_string_at(i).as_deref() == Some("like") {
            if prototype.is_some() {
                diagnostics.push(argument_error(
                    "RM-CATALOG-STRING-EMPTY-LIKE",
                    "only one like prototype is accepted",
                    i,
                ));
            }
            prototype = args.get(i + 1);
            if prototype.is_none() {
                diagnostics.push(argument_error(
                    "RM-CATALOG-STRING-EMPTY-LIKE",
                    "like requires a prototype",
                    i,
                ));
            }
            i += 2;
        } else {
            if !matches!(
                args[i].kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
            ) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-STRING-EMPTY-SIZE",
                    "size inputs must be numeric or logical",
                    i,
                ));
            }
            sizes.push(literals.get(i));
            i += 1;
        }
    }
    let shape = if sizes.is_empty() {
        prototype.and_then(|p| p.shape.known_dims()).map_or_else(
            || ShapeFact::from(vec![Some(0), Some(0)]),
            |dims| {
                let mut dims = dims.to_vec();
                if dims.len() < 2 {
                    dims.resize(2, Some(0));
                }
                dims[0] = Some(0);
                ShapeFact::from(dims)
            },
        )
    } else {
        inferred_empty_shape(&sizes)
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

fn inferred_empty_shape(literals: &[Option<&LiteralValue>]) -> ShapeFact {
    let mut dims = if literals.len() == 1 {
        match literals[0] {
            Some(LiteralValue::Vector(values)) => values.iter().map(dimension).collect::<Vec<_>>(),
            Some(v) => vec![dimension(v), dimension(v)],
            None => return ShapeFact::Unknown,
        }
    } else {
        literals
            .iter()
            .map(|v| v.and_then(|v| dimension(v)))
            .collect()
    };
    if dims.len() < 2 {
        dims.resize(2, Some(0));
    }
    dims[0] = Some(0);
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
            if n.is_finite() && *n >= 0.0 && n.fract() == 0.0 && *n < 9_007_199_254_740_992.0 =>
        {
            Some(*n as usize)
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
    fn call(args: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallInference {
        infer(
            &CallRequest {
                arguments: args,
                literals: LiteralContext::new(literals),
                outputs: OutputSelection::default(),
            },
            &super::super::STRING_EMPTY_CATALOG_ENTRY,
        )
    }
    #[test]
    fn constructor_retains_zero_leading_dimension_and_known_trailing_shape() {
        assert_eq!(
            call(vec![], vec![]).outputs[0].shape,
            ShapeFact::from(vec![Some(0), Some(0)])
        );
        assert_eq!(
            call(
                vec![numeric(ShapeFact::from(vec![Some(1), Some(1)]))],
                vec![LiteralValue::Number(5.0)]
            )
            .outputs[0]
                .shape,
            ShapeFact::from(vec![Some(0), Some(5)])
        );
        assert_eq!(
            call(
                vec![numeric(ShapeFact::from(vec![Some(1), Some(3)]))],
                vec![LiteralValue::Vector(vec![
                    LiteralValue::Number(2.0),
                    LiteralValue::Number(0.0),
                    LiteralValue::Number(4.0)
                ])]
            )
            .outputs[0]
                .shape,
            ShapeFact::from(vec![Some(0), Some(0), Some(4)])
        );
    }
    #[test]
    fn shape_of_size_vector_is_not_size_value() {
        let vector = numeric(ShapeFact::from(vec![Some(1), Some(2)]));
        assert_eq!(
            call(vec![vector], vec![]).outputs[0].shape,
            ShapeFact::Unknown
        );
    }
}
