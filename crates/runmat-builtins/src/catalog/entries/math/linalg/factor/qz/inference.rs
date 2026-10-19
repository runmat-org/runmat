use crate::catalog::entries::math::linalg::factor::inference::{
    dimensions, extent, matrix_shape, schur_operand,
};
use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, LiteralValue, NumericClass, NumericDomain,
    NumericFact, ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM.QZ.INVALID_INPUT",
            "qz expects A, B, and optional mode",
            0,
        ));
    }
    for index in 0..request.arguments.len().min(2) {
        diagnostics.extend(schur_operand(
            &request.arguments[index],
            index,
            "RM.QZ.INVALID_INPUT",
        ));
    }
    let a = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    let b = request
        .arguments
        .get(1)
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    if let (Some((Some(a_rows), Some(a_cols))), Some((Some(b_rows), Some(b_cols)))) =
        (dimensions(&a), dimensions(&b))
    {
        if a_rows != b_rows || a_cols != b_cols {
            diagnostics.push(argument_error(
                "RM.QZ.INVALID_INPUT",
                "qz coefficient matrices must have the same order",
                1,
            ));
        }
    }
    let mode = if request.arguments.len() < 3 {
        Some(NumericDomain::Complex)
    } else {
        match request.literals.literal_args.get(2) {
            Some(LiteralValue::Character(value))
            | Some(LiteralValue::String(value))
            | Some(LiteralValue::Keyword(value)) => {
                if value.trim().eq_ignore_ascii_case("complex") {
                    Some(NumericDomain::Complex)
                } else if value.trim().eq_ignore_ascii_case("real") {
                    Some(NumericDomain::Real)
                } else {
                    diagnostics.push(argument_error(
                        "RM.QZ.INVALID_INPUT",
                        "qz mode must be real or complex",
                        2,
                    ));
                    None
                }
            }
            _ => None,
        }
    };
    if mode == Some(NumericDomain::Real)
        && [&a, &b].iter().any(|input| {
            matches!(&input.kind, ValueKindFact::Numeric(fact) if fact.domain == NumericDomain::Complex)
        })
    {
        diagnostics.push(argument_error(
            "RM.QZ.INVALID_INPUT",
            "real QZ mode requires real coefficient matrices",
            2,
        ));
    }
    let class = match (&a.kind, &b.kind) {
        (ValueKindFact::Numeric(first), ValueKindFact::Numeric(second))
            if matches!(first.class, NumericClass::Single | NumericClass::Double)
                && matches!(second.class, NumericClass::Single | NumericClass::Double) =>
        {
            if first.class == NumericClass::Double || second.class == NumericClass::Double {
                Some(NumericClass::Double)
            } else if first.class == NumericClass::Single && second.class == NumericClass::Single {
                Some(NumericClass::Single)
            } else {
                None
            }
        }
        _ => None,
    };
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.shape = matrix_shape(extent(&a), extent(&a));
    output.storage = StorageFact::Dense;
    output.residency = ResidencyFact::Host;
    if let (Some(class), Some(domain)) = (class, mode) {
        output.kind = ValueKindFact::Numeric(NumericFact { class, domain });
    }
    finish_fixed_outputs(entry, request, vec![output; 6], diagnostics)
}
