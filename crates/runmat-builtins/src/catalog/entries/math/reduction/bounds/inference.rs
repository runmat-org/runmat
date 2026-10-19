use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    if !(1..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-BOUNDS-ARITY",
            "expected an array, optional dimension selection and missing flag",
            0,
        ));
    }
    if let Some(input) = request.arguments.first() {
        if !matches!(
            input.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
        ) || input.storage == StorageFact::Sparse
            || input.numeric().is_some_and(|n| {
                n.class.integer_class().is_some() && n.domain == NumericDomain::Complex
            })
        {
            diagnostics.push(argument_error(
                "RM-BOUNDS-INPUT",
                "expected a dense real integer, floating or logical array",
                0,
            ));
        }
        output.kind = match input.kind {
            ValueKindFact::Logical => ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            _ => input.kind.clone(),
        };
        output.shape = result_shape(request, &input.shape, &mut diagnostics);
        output.layout = LayoutFact::ColumnMajor;
        output.residency = if input.residency == ResidencyFact::Host {
            ResidencyFact::Host
        } else {
            ResidencyFact::Unknown
        };
        output.storage = if output.residency != ResidencyFact::Host {
            StorageFact::Unknown
        } else {
            match output.shape.element_count() {
                Some(1) if matches!(output.kind, ValueKindFact::Unknown) => StorageFact::Unknown,
                Some(1)
                    if !output
                        .numeric()
                        .is_some_and(|n| n.class == NumericClass::Single) =>
                {
                    StorageFact::Scalar
                }
                Some(_) => StorageFact::Dense,
                None => StorageFact::Unknown,
            }
        };
    }
    let mut contract = CallContract::fixed(vec![output.clone(), output]);
    // Extra slots are independently gated at runtime, not a second output carrier.
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(ValueFact::scalar(ValueKindFact::Numeric(
        NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        },
    ))));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    inference.diagnostics.extend(diagnostics);
    inference
}

fn literal_dim(value: &LiteralValue) -> Option<usize> {
    match value {
        LiteralValue::Integer { text, .. } => text.parse::<usize>().ok().filter(|n| *n > 0),
        value => LiteralContext::numeric_from_literal(value).and_then(|n| {
            (n.is_finite() && n >= 1.0 && n.fract() == 0.0 && n < usize::MAX as f64)
                .then_some(n as usize)
        }),
    }
}

fn result_shape(
    request: &CallRequest,
    input: &ShapeFact,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ShapeFact {
    let mut selected: Option<Vec<usize>> = None;
    let mut all = false;
    let mut unknown = false;
    for index in 1..request.arguments.len() {
        match request.literals.literal_string_at(index).as_deref() {
            Some("omitnan" | "omitmissing" | "includenan" | "includemissing") => continue,
            Some("all" | "linear") => {
                if all || selected.is_some() {
                    diagnostics.push(argument_error(
                        "RM-BOUNDS-DIM",
                        "only one dimension selection is allowed",
                        index,
                    ));
                }
                all = true;
            }
            Some("comparisonmethod") => {
                // This historical two-option form delegates to the extrema engine.
                if request
                    .literals
                    .literal_string_at(index + 1)
                    .is_some_and(|s| ["auto", "real", "abs"].contains(&s.as_str()))
                {
                    break;
                }
                diagnostics.push(argument_error(
                    "RM-BOUNDS-OPTION",
                    "unsupported comparison option",
                    index,
                ));
                break;
            }
            Some(_) => diagnostics.push(argument_error(
                "RM-BOUNDS-OPTION",
                "unsupported missing flag or dimension selection",
                index,
            )),
            None => {
                let literal = request.literals.literal_args.get(index);
                let dims = match literal {
                    Some(LiteralValue::Vector(values)) => {
                        values.iter().map(literal_dim).collect::<Option<Vec<_>>>()
                    }
                    Some(LiteralValue::Empty) => Some(vec![]),
                    Some(LiteralValue::Unknown) | None => None,
                    Some(value) => literal_dim(value).map(|n| vec![n]),
                };
                if dims.is_none() {
                    unknown = true;
                    if literal.is_some_and(|l| !matches!(l, LiteralValue::Unknown)) {
                        diagnostics.push(argument_error(
                            "RM-BOUNDS-DIM",
                            "dimensions must be positive integer scalars or vectors",
                            index,
                        ));
                    }
                } else {
                    if all || selected.is_some() {
                        diagnostics.push(argument_error(
                            "RM-BOUNDS-DIM",
                            "only one dimension selection is allowed",
                            index,
                        ));
                    }
                    selected = dims;
                }
            }
        }
    }
    if unknown {
        return ShapeFact::Unknown;
    }
    let Some(mut shape) = input.known_dims().map(|dims| dims.to_vec()) else {
        return ShapeFact::Unknown;
    };
    if input.element_count() == Some(0) {
        return input.clone();
    }
    // Empty extrema preserve their source shape. Unknown extents can be zero,
    // even when an explicit selector would otherwise collapse that dimension.
    if input.element_count().is_none() {
        return ShapeFact::Unknown;
    }
    if all {
        shape.fill(Some(1));
        return ShapeFact::from(shape);
    }
    let dims = match selected {
        Some(dims) if !dims.is_empty() => dims,
        _ => {
            let mut axis = None;
            for (index, dim) in shape.iter().enumerate() {
                match dim {
                    Some(n) if *n > 1 => {
                        axis = Some(index + 1);
                        break;
                    }
                    None => return ShapeFact::Unknown,
                    _ => {}
                }
            }
            vec![axis.unwrap_or(1)]
        }
    };
    for dim in dims {
        if let Some(extent) = shape.get_mut(dim - 1) {
            *extent = Some(1);
        }
    }
    ShapeFact::from(shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unknown_scalar_class_keeps_storage_unknown() {
        let mut req = request(vec![Some(1), Some(1)], vec![LiteralValue::Unknown]);
        req.arguments[0].kind = ValueKindFact::Unknown;
        let result = infer(&req, &super::super::BOUNDS_CATALOG_ENTRY);
        assert_eq!(result.outputs[0].storage, StorageFact::Unknown);
    }
    fn request(shape: Vec<Option<usize>>, literals: Vec<LiteralValue>) -> CallRequest {
        let input = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(shape),
            StorageFact::Dense,
        );
        let mut arguments = vec![input];
        arguments
            .extend((1..literals.len()).map(|_| ValueFact::unknown(DynamicReason::RuntimeValue)));
        CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        }
    }
    #[test]
    fn known_dims_preserve_integer_and_reduce_both_outputs() {
        let req = request(
            vec![Some(2), Some(3), Some(4)],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Vector(vec![LiteralValue::Number(1.0), LiteralValue::Number(3.0)]),
            ],
        );
        let result = infer(&req, &super::super::BOUNDS_CATALOG_ENTRY);
        for output in result.outputs {
            assert_eq!(
                output.shape,
                ShapeFact::from(vec![Some(1), Some(3), Some(1)])
            );
            assert_eq!(output.numeric().unwrap().class, NumericClass::UInt64);
        }
    }
    #[test]
    fn unknown_selector_contents_do_not_become_dimensions() {
        let mut req = request(
            vec![Some(2), Some(3)],
            vec![LiteralValue::Unknown, LiteralValue::Unknown],
        );
        req.arguments[1] = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        );
        assert_eq!(
            infer(&req, &super::super::BOUNDS_CATALOG_ENTRY).outputs[0].shape,
            ShapeFact::Unknown
        );
    }
    #[test]
    fn padded_outputs_are_host_double_and_invalid_options_are_diagnosed() {
        let mut req = request(
            vec![Some(2), Some(3)],
            vec![LiteralValue::Unknown, LiteralValue::String("all".into())],
        );
        req.outputs = OutputSelection::new(RequestedOutputCount::Exactly(3));
        let result = infer(&req, &super::super::BOUNDS_CATALOG_ENTRY);
        assert_eq!(
            result.outputs[2].numeric().unwrap().class,
            NumericClass::Double
        );
        req.literals.literal_args[1] = LiteralValue::Number(0.0);
        assert!(!infer(&req, &super::super::BOUNDS_CATALOG_ENTRY)
            .diagnostics
            .is_empty());
    }
}
