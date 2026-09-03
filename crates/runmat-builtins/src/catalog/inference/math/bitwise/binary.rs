use crate::catalog::inference::{argument_error, finish_fixed};
use crate::{BinaryBitwiseOperator, BuiltinCatalogEntry};
use runmat_types::{
    broadcast_shape, AliasFact, CallInference, CallRequest, DynamicReason, MutationFact,
    NumericClass, NumericDomain, NumericFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(
    _operator: BinaryBitwiseOperator,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-BITWISE-BINARY-ARITY",
            "binary bitwise functions require two inputs and accept one optional assumed type",
            request.arguments.len().min(2),
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    }

    let left = &request.arguments[0];
    let right = &request.arguments[1];
    let shape = match broadcast_shape(&left.shape, &right.shape) {
        Ok(shape) => shape,
        Err(diagnostic) => {
            diagnostics.push(diagnostic);
            return finish_fixed(
                entry,
                request,
                ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
                diagnostics,
            );
        }
    };

    let kind = match (&left.kind, &right.kind) {
        (ValueKindFact::Logical, ValueKindFact::Logical) => ValueKindFact::Logical,
        (ValueKindFact::Numeric(left_numeric), ValueKindFact::Numeric(right_numeric))
            if left_numeric.domain == NumericDomain::Real
                && right_numeric.domain == NumericDomain::Real =>
        {
            match binary_numeric_class(left, *left_numeric, right, *right_numeric) {
                Some(class) => ValueKindFact::Numeric(NumericFact {
                    class,
                    domain: NumericDomain::Real,
                }),
                None => ValueKindFact::Unknown,
            }
        }
        (ValueKindFact::Unknown, _) | (_, ValueKindFact::Unknown) => ValueKindFact::Unknown,
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-BITWISE-BINARY-INPUT",
                "binary bitwise inputs must both be real numeric values or both be logical values",
                0,
            ));
            ValueKindFact::Unknown
        }
    };

    let mut output = if matches!(kind, ValueKindFact::Unknown) {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    } else {
        ValueFact::proven(kind, shape.clone(), storage_for(&shape, left, right))
    };
    output.alias = AliasFact::Unique;
    output.mutation = MutationFact::ValueSemantics;
    output.residency = if left.residency == right.residency {
        left.residency.clone()
    } else {
        runmat_types::ResidencyFact::Unknown
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn binary_numeric_class(
    left: &ValueFact,
    left_numeric: NumericFact,
    right: &ValueFact,
    right_numeric: NumericFact,
) -> Option<NumericClass> {
    if left_numeric.class == right_numeric.class {
        return Some(left_numeric.class);
    }
    if left_numeric.class == NumericClass::Double
        && left.is_scalar()
        && right_numeric.class.integer_class().is_some()
    {
        return Some(right_numeric.class);
    }
    if right_numeric.class == NumericClass::Double
        && right.is_scalar()
        && left_numeric.class.integer_class().is_some()
    {
        return Some(left_numeric.class);
    }
    None
}

fn storage_for(
    shape: &runmat_types::ShapeFact,
    left: &ValueFact,
    right: &ValueFact,
) -> StorageFact {
    if shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else if matches!(left.storage, StorageFact::Sparse)
        && matches!(right.storage, StorageFact::Sparse | StorageFact::Scalar)
        || matches!(right.storage, StorageFact::Sparse)
            && matches!(left.storage, StorageFact::Sparse | StorageFact::Scalar)
    {
        // A zero-preserving binary operator can retain sparse storage for these
        // operand forms, but the catalog intentionally shares this conservative
        // rule across AND, OR, and XOR until value facts can prove the scalar.
        StorageFact::Unknown
    } else {
        StorageFact::Dense
    }
}
