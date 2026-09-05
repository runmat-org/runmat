use crate::NumericLimitRule;
use runmat_types::{
    DynamicReason, InferenceDiagnostic, NumericClass, NumericDomain, NumericFact, StorageFact,
    ValueFact, ValueKindFact,
};

use super::super::super::argument_error;

pub(super) fn default(rule: NumericLimitRule) -> ValueFact {
    let class = match rule {
        NumericLimitRule::Integer(_) => NumericClass::Int32,
        NumericLimitRule::Floating(_) => NumericClass::Double,
    };
    scalar(class, NumericDomain::Real)
}

pub(super) fn accepts(rule: NumericLimitRule, class: NumericClass) -> bool {
    match rule {
        NumericLimitRule::Integer(_) => class.integer_class().is_some(),
        NumericLimitRule::Floating(_) => {
            matches!(class, NumericClass::Double | NumericClass::Single)
        }
    }
}

pub(super) fn scalar(class: NumericClass, domain: NumericDomain) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact { class, domain }))
}

pub(super) fn like(
    rule: NumericLimitRule,
    prototype: &ValueFact,
    builtin: &str,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ValueFact {
    let ValueKindFact::Numeric(numeric) = prototype.kind else {
        if !matches!(prototype.kind, ValueKindFact::Unknown) {
            diagnostics.push(argument_error(
                "RM-CATALOG-NUMERIC-LIMIT-PROTOTYPE",
                format!("{builtin} requires a numeric prototype"),
                1,
            ));
        }
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    if !accepts(rule, numeric.class)
        || matches!(rule, NumericLimitRule::Integer(_))
            && matches!(prototype.storage, StorageFact::Sparse)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-NUMERIC-LIMIT-PROTOTYPE",
            format!("{builtin} does not support this prototype representation"),
            1,
        ));
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }

    let mut output = scalar(numeric.class, numeric.domain);
    output.storage = if matches!(prototype.storage, StorageFact::Sparse) {
        StorageFact::Sparse
    } else {
        StorageFact::Scalar
    };
    output.residency = prototype.residency.clone();
    output
}
