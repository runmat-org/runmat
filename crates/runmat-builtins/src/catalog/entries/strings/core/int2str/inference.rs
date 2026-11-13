use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CertaintyFact, DynamicReason, NumericDomain, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-INT2STR-ARITY",
            "int2str requires exactly one input",
            0,
        ));
    }
    let shape = match request.arguments.first() {
        Some(input) => {
            let allowed = matches!(&input.kind, ValueKindFact::Numeric(n) if n.domain == NumericDomain::Real)
                || matches!(&input.kind, ValueKindFact::Logical | ValueKindFact::Unknown);
            if !allowed {
                diagnostics.push(argument_error(
                    "RM-CATALOG-INT2STR-NUMERIC",
                    "int2str requires real numeric or logical input",
                    0,
                ));
                ShapeFact::Unknown
            } else if let Some(dims) = input.shape.known_dims() {
                if dims.len() > 2 {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-INT2STR-DIMENSIONS",
                        "int2str accepts at most two dimensions",
                        0,
                    ));
                    ShapeFact::Unknown
                } else if input.shape.element_count() == Some(0) {
                    ShapeFact::from(vec![Some(0), Some(0)])
                } else {
                    ShapeFact::from(vec![dims.first().copied().flatten(), None])
                }
            } else {
                ShapeFact::Unknown
            }
        }
        None => ShapeFact::Unknown,
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

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, NumericClass, NumericFact, OutputSelection};

    fn call(input: ValueFact) -> CallInference {
        infer(
            &CallRequest {
                arguments: vec![input],
                literals: LiteralContext::default(),
                outputs: OutputSelection::default(),
            },
            &super::super::INT2STR_CATALOG_ENTRY,
        )
    }
    #[test]
    fn row_count_is_known_but_formatted_width_is_not() {
        let input = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        let output = call(input);
        assert_eq!(output.outputs[0].kind, ValueKindFact::Character);
        assert_eq!(
            output.outputs[0].shape,
            ShapeFact::from(vec![Some(2), None])
        );
        let empty = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(0), Some(3)]),
            StorageFact::Dense,
        );
        assert_eq!(
            call(empty).outputs[0].shape,
            ShapeFact::from(vec![Some(0), Some(0)])
        );
    }
    #[test]
    fn unknown_complex_and_nd_inputs_stay_conservative() {
        let unknown = call(ValueFact::unknown(DynamicReason::RuntimeValue));
        assert_eq!(unknown.outputs[0].shape, ShapeFact::Unknown);
        assert!(matches!(
            unknown.outputs[0].certainty,
            CertaintyFact::Dynamic(_)
        ));
        let complex = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }));
        assert!(call(complex)
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-INT2STR-NUMERIC"));
        let nd = ValueFact::proven(
            ValueKindFact::Logical,
            ShapeFact::from(vec![Some(2), Some(3), Some(4)]),
            StorageFact::Dense,
        );
        assert!(call(nd)
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-INT2STR-DIMENSIONS"));
    }
}
