use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-KRON-ARITY",
            "kron requires exactly two inputs",
            0,
        ));
    }
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    let [left, right] = request.arguments.as_slice() else {
        return finish_fixed(entry, request, output, diagnostics);
    };
    for (index, operand) in request.arguments.iter().enumerate() {
        if matches!(operand.residency, ResidencyFact::Remote { .. }) {
            diagnostics.push(argument_error(
                "RM-KRON-REMOTE",
                "kron does not accept distributed operands",
                index,
            ));
        }
        if !matches!(
            operand.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Numeric(_)
                | ValueKindFact::Logical
                | ValueKindFact::Character
        ) {
            diagnostics.push(argument_error(
                "RM-KRON-INPUT",
                "kron requires numeric, logical, or character operands",
                index,
            ));
        }
        if operand.storage == StorageFact::Sparse
            && operand.shape.rank().is_some_and(|rank| rank != 2)
        {
            diagnostics.push(argument_error(
                "RM-KRON-SPARSE-RANK",
                "sparse kron operands must be two-dimensional",
                index,
            ));
        }
    }
    let sparse = left.storage == StorageFact::Sparse || right.storage == StorageFact::Sparse;
    if sparse
        && request
            .arguments
            .iter()
            .any(|operand| operand.shape.rank().is_some_and(|rank| rank != 2))
    {
        diagnostics.push(argument_error(
            "RM-KRON-SPARSE-RANK",
            "sparse kron operands must be two-dimensional",
            0,
        ));
    }
    let (Some(mut left_dims), Some(mut right_dims)) =
        (left.shape.known_dims(), right.shape.known_dims())
    else {
        output.kind = result_kind(left, right, &mut diagnostics);
        output.storage = if sparse {
            StorageFact::Sparse
        } else {
            StorageFact::Unknown
        };
        return finish_fixed(entry, request, output, diagnostics);
    };
    let rank = left_dims.len().max(right_dims.len()).max(1);
    left_dims.resize(rank, Some(1));
    right_dims.resize(rank, Some(1));
    let mut dimensions = Vec::with_capacity(rank);
    for (a, b) in left_dims.into_iter().zip(right_dims) {
        let dimension = match (a, b) {
            (Some(a), Some(b)) => match a.checked_mul(b) {
                Some(product) => Some(product),
                None => {
                    diagnostics.push(argument_error(
                        "RM-KRON-SIZE",
                        "kron output extent overflows",
                        0,
                    ));
                    None
                }
            },
            (Some(0), _) | (_, Some(0)) => Some(0),
            _ => None,
        };
        dimensions.push(dimension);
    }
    output.shape = ShapeFact::from(dimensions);
    output.kind = result_kind(left, right, &mut diagnostics);
    output.storage = if sparse {
        StorageFact::Sparse
    } else if output.kind == ValueKindFact::Unknown {
        StorageFact::Unknown
    } else if output.shape.element_count() == Some(1)
        && output
            .numeric()
            .is_some_and(|number| number.class != NumericClass::Single)
    {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    output.layout = LayoutFact::ColumnMajor;
    output.residency =
        if left.residency == ResidencyFact::Host && right.residency == ResidencyFact::Host {
            ResidencyFact::Host
        } else {
            ResidencyFact::Unknown
        };
    finish_fixed(entry, request, output, diagnostics)
}

fn result_kind(
    left: &ValueFact,
    right: &ValueFact,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ValueKindFact {
    let numeric = |value: &ValueFact| match value.kind {
        ValueKindFact::Numeric(numeric) => Some(numeric),
        ValueKindFact::Logical | ValueKindFact::Character => Some(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        _ => None,
    };
    let (Some(left_numeric), Some(right_numeric)) = (numeric(left), numeric(right)) else {
        return ValueKindFact::Unknown;
    };
    let left_integer = left_numeric.class.integer_class().is_some();
    let right_integer = right_numeric.class.integer_class().is_some();
    let scalar_double = |value: &ValueFact, number: NumericFact| {
        value.shape.element_count() == Some(1)
            && matches!(value.kind, ValueKindFact::Numeric(_))
            && number.class == NumericClass::Double
    };
    let class = if left_integer || right_integer {
        if left_numeric.domain == NumericDomain::Complex
            || right_numeric.domain == NumericDomain::Complex
        {
            diagnostics.push(argument_error(
                "RM-KRON-COMPLEX-INTEGER",
                "complex integer kron arithmetic is unsupported",
                0,
            ));
            return ValueKindFact::Unknown;
        }
        if left_integer
            && (left_numeric.class == right_numeric.class || scalar_double(right, right_numeric))
        {
            left_numeric.class
        } else if right_integer && scalar_double(left, left_numeric) {
            right_numeric.class
        } else {
            diagnostics.push(argument_error(
                "RM-KRON-INTEGER-CLASS",
                "integer kron operands require the same class or a scalar double",
                0,
            ));
            return ValueKindFact::Unknown;
        }
    } else if left_numeric.class == NumericClass::Single
        || right_numeric.class == NumericClass::Single
    {
        NumericClass::Single
    } else {
        NumericClass::Double
    };
    let domain = if left_numeric.domain == NumericDomain::Complex
        || right_numeric.domain == NumericDomain::Complex
    {
        NumericDomain::Complex
    } else {
        NumericDomain::Real
    };
    ValueKindFact::Numeric(NumericFact { class, domain })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fact(class: NumericClass, shape: Vec<Option<usize>>, storage: StorageFact) -> ValueFact {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(shape),
            storage,
        )
    }
    fn call(arguments: Vec<ValueFact>) -> CallInference {
        infer(
            &CallRequest {
                arguments,
                literals: LiteralContext::default(),
                outputs: OutputSelection::default(),
            },
            &super::super::KRON_CATALOG_ENTRY,
        )
    }
    #[test]
    fn exact_extents_class_and_sparse_storage_follow_host_product() {
        let result = call(vec![
            fact(
                NumericClass::Single,
                vec![Some(2), Some(3)],
                StorageFact::Sparse,
            ),
            fact(
                NumericClass::Double,
                vec![Some(4), Some(0)],
                StorageFact::Dense,
            ),
        ]);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(8), Some(0)])
        );
        assert_eq!(
            result.outputs[0].numeric().unwrap().class,
            NumericClass::Single
        );
        assert_eq!(result.outputs[0].storage, StorageFact::Sparse);
    }
    #[test]
    fn unknown_extents_are_not_inferred_from_operand_rank() {
        let result = call(vec![
            fact(
                NumericClass::Double,
                vec![Some(2), None],
                StorageFact::Dense,
            ),
            fact(
                NumericClass::Double,
                vec![Some(3), Some(4)],
                StorageFact::Dense,
            ),
        ]);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(6), None])
        );
    }
    #[test]
    fn invalid_classes_and_arity_are_diagnosed() {
        assert!(!call(vec![]).diagnostics.is_empty());
        assert!(!call(vec![
            ValueFact::scalar(ValueKindFact::String),
            fact(
                NumericClass::Double,
                vec![Some(1), Some(1)],
                StorageFact::Dense
            )
        ])
        .diagnostics
        .is_empty());
        assert!(!call(vec![
            fact(
                NumericClass::Int64,
                vec![Some(2), Some(2)],
                StorageFact::Dense
            ),
            fact(
                NumericClass::Double,
                vec![Some(2), Some(2)],
                StorageFact::Dense
            )
        ])
        .diagnostics
        .is_empty());
    }
}
