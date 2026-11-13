use super::{CHOL_ERROR_INVALID_ARGUMENT, CHOL_ERROR_INVALID_INPUT};
use crate::catalog::entries::math::linalg::factor::inference::{
    double_scalar, factor_output, matrix_input,
};
use crate::catalog::inference::finish_fixed_outputs;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, LiteralValue, NumericClass, NumericDomain,
    NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = matrix_input(
        request,
        3,
        true,
        CHOL_ERROR_INVALID_ARGUMENT.code,
        CHOL_ERROR_INVALID_INPUT.code,
    );
    let matrix = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    let mut factor = factor_output(&matrix, matrix.shape.clone(), false, false, false);
    if matrix.storage == StorageFact::Sparse {
        factor.storage = StorageFact::Sparse;
        factor.residency = ResidencyFact::Host;
    }
    let mut permutation = ValueFact::unknown(DynamicReason::RuntimeValue);
    permutation.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    });
    let vector_form = request.literals.literal_args.iter().skip(1).any(|literal| {
        matches!(literal, LiteralValue::Character(text) | LiteralValue::String(text) | LiteralValue::Keyword(text) if text.eq_ignore_ascii_case("vector"))
    });
    permutation.storage = if matrix.storage == StorageFact::Sparse {
        if vector_form {
            StorageFact::Dense
        } else {
            StorageFact::Sparse
        }
    } else {
        StorageFact::Unknown
    };
    permutation.residency = ResidencyFact::Host;
    permutation.shape = match matrix.shape.known_dims().as_deref() {
        Some([Some(n), Some(_)]) if vector_form => ShapeFact::from(vec![Some(1), Some(*n)]),
        Some([Some(n), Some(_)]) => ShapeFact::from(vec![Some(*n), Some(*n)]),
        _ => ShapeFact::Ranked { rank: 2 },
    };
    finish_fixed_outputs(
        entry,
        request,
        vec![factor, double_scalar(), permutation],
        diagnostics,
    )
}
