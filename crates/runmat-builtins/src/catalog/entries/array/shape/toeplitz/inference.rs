use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-TOEPLITZ-ARITY",
            "toeplitz requires one or two inputs",
            0,
        ));
        return finish_fixed(entry, request, output, diagnostics);
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        if !matches!(
            argument.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Unknown
        ) || matches!(argument.storage, StorageFact::Sparse)
        {
            diagnostics.push(argument_error(
                "RM-TOEPLITZ-INPUT",
                "toeplitz requires dense numeric vectors",
                index,
            ));
        }
        if let Some(dims) = argument.shape.known_dims() {
            if argument.shape.element_count().is_some_and(|n| n > 1)
                && dims.iter().filter(|d| d.is_some_and(|n| n != 1)).count() > 1
            {
                diagnostics.push(argument_error(
                    "RM-TOEPLITZ-SHAPE",
                    "toeplitz inputs must be vectors",
                    index,
                ));
            }
        }
    }
    let column = &request.arguments[0];
    let row = request.arguments.get(1).unwrap_or(column);
    output.shape = ShapeFact::from(vec![
        column.shape.element_count(),
        row.shape.element_count(),
    ]);
    if let (Some(c), Some(r)) = (column.numeric(), row.numeric()) {
        if (c.class.integer_class().is_some() || r.class.integer_class().is_some())
            && (c.class != r.class || c.domain != r.domain)
        {
            diagnostics.push(argument_error(
                "RM-TOEPLITZ-CLASS",
                "integer inputs require a common class and complexity",
                1,
            ));
        } else {
            let domain = if c.domain == NumericDomain::Complex || r.domain == NumericDomain::Complex
            {
                NumericDomain::Complex
            } else {
                NumericDomain::Real
            };
            let class = if c.class == r.class {
                c.class
            } else if domain == NumericDomain::Complex {
                NumericClass::Single
            } else {
                NumericClass::Double
            };
            output.kind = ValueKindFact::Numeric(NumericFact { class, domain });
        }
    }
    output.residency = if request
        .arguments
        .iter()
        .all(|a| a.residency == ResidencyFact::Host)
    {
        ResidencyFact::Host
    } else {
        // Automatic resident inputs may fall back to the host. ValueFact has
        // no explicit-device-intent fact, so device input alone is insufficient.
        ResidencyFact::Unknown
    };
    output.layout = LayoutFact::ColumnMajor;
    output.storage = if output.residency == ResidencyFact::Host {
        match (output.numeric(), output.shape.element_count()) {
            (
                Some(NumericFact {
                    class: NumericClass::Double,
                    ..
                }),
                Some(1),
            ) => StorageFact::Scalar,
            (
                Some(NumericFact {
                    class: NumericClass::Double,
                    ..
                }),
                None,
            ) => StorageFact::Unknown,
            (Some(_), _) => StorageFact::Dense,
            _ => StorageFact::Unknown,
        }
    } else {
        StorageFact::Unknown
    };
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fact(class: NumericClass, dims: Vec<Option<usize>>) -> ValueFact {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(dims),
            StorageFact::Dense,
        )
    }
    fn call(arguments: Vec<ValueFact>) -> CallInference {
        infer(
            &CallRequest {
                arguments,
                literals: LiteralContext::default(),
                outputs: OutputSelection::default(),
            },
            &super::super::TOEPLITZ_CATALOG_ENTRY,
        )
    }
    #[test]
    fn vector_lengths_determine_rectangular_output() {
        let out = call(vec![
            fact(NumericClass::UInt64, vec![Some(3), Some(1)]),
            fact(NumericClass::UInt64, vec![Some(1), Some(2)]),
        ]);
        assert_eq!(
            out.outputs[0].shape,
            ShapeFact::from(vec![Some(3), Some(2)])
        );
        assert_eq!(
            out.outputs[0].numeric().unwrap().class,
            NumericClass::UInt64
        );
        assert!(out.diagnostics.is_empty());
    }
    #[test]
    fn unknown_vector_length_stays_unknown() {
        let out = call(vec![fact(NumericClass::Single, vec![Some(1), None])]);
        assert_eq!(out.outputs[0].shape, ShapeFact::from(vec![None, None]));
        assert_eq!(out.outputs[0].storage, StorageFact::Dense);
    }
    #[test]
    fn invalid_inputs_and_output_count_are_diagnosed() {
        assert!(!call(vec![]).diagnostics.is_empty());
        assert!(
            !call(vec![fact(NumericClass::Double, vec![Some(2), Some(2)])])
                .diagnostics
                .is_empty()
        );
        assert!(!call(vec![
            fact(NumericClass::Int64, vec![Some(1)]),
            fact(NumericClass::Double, vec![Some(1)])
        ])
        .diagnostics
        .is_empty());
        let request = CallRequest {
            arguments: vec![fact(NumericClass::Single, vec![Some(1)])],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        };
        assert!(!infer(&request, &super::super::TOEPLITZ_CATALOG_ENTRY)
            .diagnostics
            .is_empty());
    }
    #[test]
    fn empty_and_resident_outputs_remain_sound() {
        let out = call(vec![fact(NumericClass::Double, vec![Some(0), Some(3)])]);
        assert_eq!(
            out.outputs[0].shape,
            ShapeFact::from(vec![Some(0), Some(0)])
        );
        let mut input = fact(NumericClass::Double, vec![Some(1)]);
        input.residency = ResidencyFact::Device { provider: None };
        let out = call(vec![input]);
        assert_eq!(out.outputs[0].residency, ResidencyFact::Unknown);
        assert_eq!(out.outputs[0].storage, StorageFact::Unknown);
    }
}
