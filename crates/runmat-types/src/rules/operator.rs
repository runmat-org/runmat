use crate::{
    broadcast_shape, left_divide_shape, matrix_shape, right_divide_shape, transpose_shape,
    CertaintyFact, DynamicReason, FactInference, InferenceDiagnostic, NumericFact, OperatorKind,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub fn infer_unary(operator: OperatorKind, operand: &ValueFact) -> FactInference {
    match operator {
        OperatorKind::UnaryPlus | OperatorKind::UnaryMinus => {
            if is_known_unsupported_numeric(operand) {
                return unsupported(
                    "RM-TYPE-UNARY",
                    "unary numeric operator requires a numeric value",
                );
            }
            let kind = numeric_fact(operand)
                .map(ValueKindFact::Numeric)
                .unwrap_or(ValueKindFact::Unknown);
            FactInference::exact(derived(kind, operand.shape.clone(), operand))
        }
        OperatorKind::Not => {
            if is_known_unsupported_logical(operand) {
                return unsupported(
                    "RM-TYPE-UNARY",
                    "logical negation requires a numeric, logical, or character value",
                );
            }
            FactInference::exact(derived(
                ValueKindFact::Logical,
                operand.shape.clone(),
                operand,
            ))
        }
        OperatorKind::Transpose | OperatorKind::ConjugateTranspose => {
            if is_known_unsupported_numeric(operand) {
                return unsupported(
                    "RM-TYPE-TRANSPOSE",
                    "transpose requires a numeric, logical, or character value",
                );
            }
            if let Err(diagnostic) =
                crate::rules::shape::reject_proven_nonmatrix(&operand.shape, "transpose operand")
            {
                return FactInference {
                    fact: ValueFact::unknown(DynamicReason::ConflictingControlFlow),
                    diagnostics: vec![diagnostic],
                };
            }
            FactInference::exact(derived(
                operand.kind.clone(),
                transpose_shape(&operand.shape),
                operand,
            ))
        }
        _ => unsupported("RM-TYPE-UNARY", "operator is not unary"),
    }
}

pub fn infer_binary(operator: OperatorKind, left: &ValueFact, right: &ValueFact) -> FactInference {
    let comparison = matches!(
        operator,
        OperatorKind::Equal
            | OperatorKind::NotEqual
            | OperatorKind::Less
            | OperatorKind::LessEqual
            | OperatorKind::Greater
            | OperatorKind::GreaterEqual
            | OperatorKind::ElementwiseAnd
            | OperatorKind::ElementwiseOr
            | OperatorKind::ShortCircuitAnd
            | OperatorKind::ShortCircuitOr
    );
    let logical_operator = matches!(
        operator,
        OperatorKind::ElementwiseAnd
            | OperatorKind::ElementwiseOr
            | OperatorKind::ShortCircuitAnd
            | OperatorKind::ShortCircuitOr
    );
    let operands_supported = if logical_operator {
        !is_known_unsupported_logical(left) && !is_known_unsupported_logical(right)
    } else if comparison {
        !is_known_unsupported_comparison(left) && !is_known_unsupported_comparison(right)
    } else {
        !is_known_unsupported_numeric(left) && !is_known_unsupported_numeric(right)
    };
    if !operands_supported {
        return unsupported(
            "RM-TYPE-BINARY-OPERAND",
            "operator is not defined for the proven operand value category",
        );
    }
    if matches!(
        operator,
        OperatorKind::ShortCircuitAnd | OperatorKind::ShortCircuitOr
    ) && [left, right].iter().any(|operand| {
        operand
            .shape
            .element_count()
            .is_some_and(|count| count != 1)
    }) {
        return unsupported(
            "RM-TYPE-SHORT-CIRCUIT-SCALAR",
            "short-circuit logical operators require scalar operands",
        );
    }
    let shape = match operator {
        OperatorKind::MatrixMultiply => matrix_shape(&left.shape, &right.shape),
        OperatorKind::Mldivide => left_divide_shape(&left.shape, &right.shape),
        OperatorKind::Mrdivide => right_divide_shape(&left.shape, &right.shape),
        OperatorKind::MatrixPower => matrix_power_shape(left, right),
        OperatorKind::Add
        | OperatorKind::Subtract
        | OperatorKind::ElementwiseMultiply
        | OperatorKind::ElementwisePower
        | OperatorKind::ElementwiseDivide
        | OperatorKind::ElementwiseLeftDivide
        | OperatorKind::Equal
        | OperatorKind::NotEqual
        | OperatorKind::Less
        | OperatorKind::LessEqual
        | OperatorKind::Greater
        | OperatorKind::GreaterEqual
        | OperatorKind::ElementwiseAnd
        | OperatorKind::ElementwiseOr => broadcast_shape(&left.shape, &right.shape),
        OperatorKind::ShortCircuitAnd | OperatorKind::ShortCircuitOr => Ok(ShapeFact::Scalar),
        _ => {
            return unsupported("RM-TYPE-BINARY", "operator is not binary");
        }
    };
    let shape = match shape {
        Ok(shape) => shape,
        Err(diagnostic) => {
            let mut fact = ValueFact::unknown(DynamicReason::ConflictingControlFlow);
            fact.shape = ShapeFact::Unknown;
            return FactInference {
                fact,
                diagnostics: vec![diagnostic],
            };
        }
    };
    let kind = if comparison {
        ValueKindFact::Logical
    } else {
        numeric_result_kind(numeric_fact(left), numeric_fact(right))
    };
    let mut fact = derived(kind.clone(), shape, left);
    if matches!(kind, ValueKindFact::Unknown) {
        fact.certainty = CertaintyFact::Dynamic(DynamicReason::UnsupportedRepresentation);
    }
    if left.residency == right.residency {
        fact.residency = left.residency.clone();
    } else {
        fact.residency = crate::ResidencyFact::Unknown;
    }
    fact.storage = binary_storage(operator, left, right, &fact.shape);
    FactInference::exact(fact)
}

fn binary_storage(
    operator: OperatorKind,
    left: &ValueFact,
    right: &ValueFact,
    shape: &ShapeFact,
) -> StorageFact {
    if shape.element_count() == Some(1) {
        return StorageFact::Scalar;
    }
    match (left.storage, right.storage) {
        (StorageFact::Dense | StorageFact::Scalar, StorageFact::Dense | StorageFact::Scalar) => {
            StorageFact::Dense
        }
        (StorageFact::Sparse, StorageFact::Sparse)
            if matches!(
                operator,
                OperatorKind::Add
                    | OperatorKind::Subtract
                    | OperatorKind::ElementwiseMultiply
                    | OperatorKind::MatrixMultiply
            ) =>
        {
            StorageFact::Sparse
        }
        _ => StorageFact::Unknown,
    }
}

fn matrix_power_shape(
    left: &ValueFact,
    right: &ValueFact,
) -> Result<ShapeFact, InferenceDiagnostic> {
    if right
        .shape
        .element_count()
        .is_some_and(|element_count| element_count != 1)
    {
        return Err(InferenceDiagnostic::error(
            "RM-TYPE-MPOWER",
            "matrix power exponent must be scalar",
        ));
    }
    let Some(dims) = left.shape.known_dims() else {
        return Ok(ShapeFact::Unknown);
    };
    if dims.len() >= 2 {
        if let (Some(rows), Some(columns)) = (dims[0], dims[1]) {
            if rows != columns {
                return Err(InferenceDiagnostic::error(
                    "RM-TYPE-MPOWER",
                    "matrix power base must be square",
                ));
            }
        }
    }
    Ok(left.shape.clone())
}

fn numeric_result_kind(left: Option<NumericFact>, right: Option<NumericFact>) -> ValueKindFact {
    match (left, right) {
        (Some(left), Some(right)) if left == right => ValueKindFact::Numeric(left),
        (Some(left), Some(right)) if left.class == right.class => {
            ValueKindFact::Numeric(NumericFact {
                class: left.class,
                domain: if matches!(left.domain, crate::NumericDomain::Complex)
                    || matches!(right.domain, crate::NumericDomain::Complex)
                {
                    crate::NumericDomain::Complex
                } else {
                    crate::NumericDomain::Real
                },
            })
        }
        _ => ValueKindFact::Unknown,
    }
}

fn numeric_fact(value: &ValueFact) -> Option<NumericFact> {
    match &value.kind {
        ValueKindFact::Numeric(fact) => Some(*fact),
        ValueKindFact::Logical | ValueKindFact::Character => Some(NumericFact {
            class: crate::NumericClass::Double,
            domain: crate::NumericDomain::Real,
        }),
        _ => None,
    }
}

fn is_known_unsupported_numeric(value: &ValueFact) -> bool {
    !matches!(
        value.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Numeric(_)
            | ValueKindFact::Logical
            | ValueKindFact::Character
    )
}

fn is_known_unsupported_logical(value: &ValueFact) -> bool {
    is_known_unsupported_numeric(value)
}

fn is_known_unsupported_comparison(value: &ValueFact) -> bool {
    !matches!(
        value.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Numeric(_)
            | ValueKindFact::Logical
            | ValueKindFact::Character
            | ValueKindFact::String
    )
}

fn derived(kind: ValueKindFact, shape: ShapeFact, source: &ValueFact) -> ValueFact {
    let storage = if shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else {
        match source.storage {
            StorageFact::Sparse => StorageFact::Sparse,
            StorageFact::Scalar | StorageFact::Dense => StorageFact::Dense,
            StorageFact::Unknown | StorageFact::Opaque => StorageFact::Unknown,
        }
    };
    let mut output = ValueFact::proven(kind, shape, storage);
    output.residency = source.residency.clone();
    output
}

fn unsupported(code: &str, message: &str) -> FactInference {
    FactInference {
        fact: ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
        diagnostics: vec![InferenceDiagnostic::error(code, message)],
    }
}
