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
    } else if let Some(recipe) = super::RealFloatingRecipe::for_facts(operator, left, right) {
        ValueKindFact::Numeric(NumericFact {
            class: recipe.result_class(),
            domain: crate::NumericDomain::Real,
        })
    } else if operator == OperatorKind::ElementwiseMultiply
        || (operator == OperatorKind::MatrixMultiply
            && [left, right]
                .iter()
                .any(|operand| operand.shape.element_count() == Some(1)))
    {
        // Scalar matrix multiplication delegates to elementwise times. Share
        // its floating class contract without asserting contraction arithmetic
        // or extending integer, complex, or unproven-carrier inference.
        real_floating_product_kind(left, right)
            .unwrap_or_else(|| numeric_result_kind(numeric_fact(left), numeric_fact(right)))
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

/// Exact represented arithmetic for real floating `+`. Mixed precision uses
/// a Double addition followed by one Single result conversion, as the shared
/// host operation does; rounding both operands to Single first is different.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealFloatingAddMode {
    Single,
    Double,
    MixedDoubleThenSingle,
}

/// Class projection shared with catalog arithmetic. This does not by itself
/// admit a computation or establish its intermediate precision.
#[inline]
pub const fn floating_binary_result_class(
    left: crate::NumericClass,
    right: crate::NumericClass,
) -> Option<crate::NumericClass> {
    match (left, right) {
        (crate::NumericClass::Single, crate::NumericClass::Single)
        | (crate::NumericClass::Single, crate::NumericClass::Double)
        | (crate::NumericClass::Double, crate::NumericClass::Single) => {
            Some(crate::NumericClass::Single)
        }
        (crate::NumericClass::Double, crate::NumericClass::Double) => {
            Some(crate::NumericClass::Double)
        }
        _ => None,
    }
}

fn real_floating_product_kind(left: &ValueFact, right: &ValueFact) -> Option<ValueKindFact> {
    if !matches!(left.storage, StorageFact::Scalar | StorageFact::Dense)
        || !matches!(right.storage, StorageFact::Scalar | StorageFact::Dense)
    {
        return None;
    }
    let left = left.numeric()?;
    let right = right.numeric()?;
    if left.domain != crate::NumericDomain::Real || right.domain != crate::NumericDomain::Real {
        return None;
    }
    Some(ValueKindFact::Numeric(NumericFact {
        class: floating_binary_result_class(left.class, right.class)?,
        domain: crate::NumericDomain::Real,
    }))
}

impl RealFloatingAddMode {
    /// Project a canonical represented Add recipe onto the provider ABI.
    /// The caller must separately establish that the operands have a real
    /// scalar/dense physical representation before using this mode.
    pub fn from_recipe(recipe: super::RealFloatingRecipe) -> Option<Self> {
        if recipe.operator() != OperatorKind::Add {
            return None;
        }
        match (recipe.computation_class(), recipe.result_class()) {
            (crate::NumericClass::Single, crate::NumericClass::Single) => Some(Self::Single),
            (crate::NumericClass::Double, crate::NumericClass::Double) => Some(Self::Double),
            (crate::NumericClass::Double, crate::NumericClass::Single) => {
                Some(Self::MixedDoubleThenSingle)
            }
            _ => None,
        }
    }

    pub const fn result_class(self) -> crate::NumericClass {
        match self {
            Self::Single | Self::MixedDoubleThenSingle => crate::NumericClass::Single,
            Self::Double => crate::NumericClass::Double,
        }
    }

    #[inline]
    pub fn add_represented(self, left: f64, right: f64) -> f64 {
        let (left_class, right_class) = match self {
            Self::Single => (crate::NumericClass::Single, crate::NumericClass::Single),
            Self::Double => (crate::NumericClass::Double, crate::NumericClass::Double),
            Self::MixedDoubleThenSingle => {
                (crate::NumericClass::Single, crate::NumericClass::Double)
            }
        };
        super::RealFloatingRecipe::for_classes(OperatorKind::Add, left_class, right_class)
            .and_then(|recipe| recipe.evaluate(left, right))
            .expect("real Add mode has a represented recipe")
    }
}

pub fn real_floating_add_mode(left: &ValueFact, right: &ValueFact) -> Option<RealFloatingAddMode> {
    // Retain the established provider ABI modes only as a projection of the
    // canonical represented arithmetic owner, never a second classifier.
    let recipe = super::RealFloatingRecipe::for_facts(OperatorKind::Add, left, right)?;
    RealFloatingAddMode::from_recipe(recipe)
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

#[cfg(test)]
mod mixed_add_tests {
    use super::*;

    fn real(class: crate::NumericClass, shape: &[Option<usize>]) -> ValueFact {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: crate::NumericDomain::Real,
            }),
            ShapeFact::from(shape.to_vec()),
            StorageFact::Dense,
        )
    }

    #[test]
    fn mixed_real_add_proves_single_and_shared_arithmetic_contracts() {
        let single = real(crate::NumericClass::Single, &[Some(2), Some(1)]);
        let double_scalar = real(crate::NumericClass::Double, &[Some(1), Some(1)]);
        let inferred = infer_binary(OperatorKind::Add, &single, &double_scalar);
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(inferred.fact.shape, ShapeFact::from(vec![Some(2), Some(1)]));
        assert_eq!(
            inferred.fact.numeric().map(|numeric| numeric.class),
            Some(crate::NumericClass::Single)
        );
        assert_eq!(
            real_floating_add_mode(&single, &double_scalar),
            Some(RealFloatingAddMode::MixedDoubleThenSingle)
        );
        for storage in [
            StorageFact::Sparse,
            StorageFact::Unknown,
            StorageFact::Opaque,
        ] {
            let mut unproved = single.clone();
            unproved.storage = storage;
            assert_eq!(real_floating_add_mode(&unproved, &double_scalar), None);
            assert!(infer_binary(OperatorKind::Add, &unproved, &double_scalar)
                .fact
                .numeric()
                .is_none());
        }
        let double_array = real(crate::NumericClass::Double, &[Some(1), Some(3)]);
        assert_eq!(
            infer_binary(OperatorKind::Add, &single, &double_array)
                .fact
                .shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
        assert_eq!(
            infer_binary(OperatorKind::Subtract, &single, &double_scalar)
                .fact
                .kind,
            ValueKindFact::Numeric(NumericFact {
                class: crate::NumericClass::Single,
                domain: crate::NumericDomain::Real,
            })
        );
        let mismatch = real(crate::NumericClass::Double, &[Some(3), Some(2)]);
        assert!(!infer_binary(OperatorKind::Add, &single, &mismatch)
            .diagnostics
            .is_empty());
    }

    #[test]
    fn mixed_real_products_share_class_contract_for_scalar_matrix_and_elementwise_times() {
        let single = real(crate::NumericClass::Single, &[Some(2), Some(1)]);
        let mut double_scalar = real(crate::NumericClass::Double, &[Some(1), Some(1)]);
        double_scalar.storage = StorageFact::Scalar;
        for operator in [
            OperatorKind::MatrixMultiply,
            OperatorKind::ElementwiseMultiply,
        ] {
            for (left, right) in [(&single, &double_scalar), (&double_scalar, &single)] {
                let inferred = infer_binary(operator, left, right);
                assert!(inferred.diagnostics.is_empty());
                assert_eq!(inferred.fact.shape, single.shape);
                assert_eq!(inferred.fact.storage, StorageFact::Dense);
                assert_eq!(
                    inferred.fact.numeric(),
                    Some(NumericFact {
                        class: crate::NumericClass::Single,
                        domain: crate::NumericDomain::Real,
                    })
                );
            }
            for storage in [
                StorageFact::Sparse,
                StorageFact::Unknown,
                StorageFact::Opaque,
            ] {
                let mut unproved = single.clone();
                unproved.storage = storage;
                assert!(infer_binary(operator, &unproved, &double_scalar)
                    .fact
                    .numeric()
                    .is_none());
            }
            let mut complex = single.clone();
            complex.kind = ValueKindFact::Numeric(NumericFact {
                class: crate::NumericClass::Single,
                domain: crate::NumericDomain::Complex,
            });
            assert!(infer_binary(operator, &complex, &double_scalar)
                .fact
                .numeric()
                .is_none());
            let integer = real(crate::NumericClass::Int8, &[Some(2), Some(1)]);
            assert!(infer_binary(operator, &integer, &double_scalar)
                .fact
                .numeric()
                .is_none());
        }
        let double_row = real(crate::NumericClass::Double, &[Some(1), Some(3)]);
        assert!(
            infer_binary(OperatorKind::MatrixMultiply, &single, &double_row)
                .fact
                .numeric()
                .is_none()
        );
        let broadcast = infer_binary(OperatorKind::ElementwiseMultiply, &single, &double_row);
        assert_eq!(
            broadcast.fact.shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
        assert_eq!(
            broadcast.fact.numeric().map(|numeric| numeric.class),
            Some(crate::NumericClass::Single)
        );
        for operator in [OperatorKind::Subtract, OperatorKind::ElementwiseDivide] {
            let inferred = infer_binary(operator, &single, &double_scalar);
            assert!(inferred.diagnostics.is_empty());
            assert_eq!(inferred.fact.shape, single.shape);
            assert_eq!(
                inferred.fact.kind,
                ValueKindFact::Numeric(NumericFact {
                    class: crate::NumericClass::Single,
                    domain: crate::NumericDomain::Real,
                })
            );
        }
        assert!(
            infer_binary(OperatorKind::ElementwisePower, &single, &double_scalar)
                .fact
                .numeric()
                .is_none()
        );
    }

    #[test]
    fn mixed_add_rounds_sum_once_and_preserves_unknown_exceptional_bounds() {
        let below_one = f64::from(f32::from_bits(0x3f7f_ffff));
        let half_ulp = f64::from(f32::EPSILON) / 4.0;
        let mode = RealFloatingAddMode::MixedDoubleThenSingle;
        assert_eq!(
            mode.add_represented(below_one, half_ulp - f64::EPSILON),
            below_one
        );
        assert_eq!(
            mode.add_represented(half_ulp + f64::EPSILON, below_one),
            1.0
        );
        assert!(mode
            .add_represented(f64::INFINITY, f64::NEG_INFINITY)
            .is_nan());
    }
}
