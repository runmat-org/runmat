use crate::{AngleConversionInferenceRule, BuiltinCatalogEntry};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, ValueFact,
    ValueKindFact,
};

use super::super::super::support::facts::{materialize, preserve_shape_as_dynamic};
use super::super::super::{argument_error, finish_fixed, numeric_kind};

pub(super) fn infer(
    _rule: AngleConversionInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let name = entry.identity.name;
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-ANGLE-CONVERSION-ARITY",
            format!("{name} requires exactly one input"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ANGLE-CONVERSION-ARITY",
            format!("{name} accepts exactly one input"),
            1,
        ));
    }

    let mut output = input.clone();
    match &mut output.kind {
        ValueKindFact::Numeric(numeric)
            if numeric.domain == NumericDomain::Complex
                && !matches!(numeric.class, NumericClass::Double | NumericClass::Single) =>
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ANGLE-CONVERSION-COMPLEX-INTEGER",
                format!("{name} does not accept complex fixed-width integer input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric) => {
            if !matches!(numeric.class, NumericClass::Double | NumericClass::Single) {
                numeric.class = NumericClass::Double;
            }
            materialize(&mut output);
        }
        ValueKindFact::Logical => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize(&mut output);
        }
        ValueKindFact::Unknown => preserve_shape_as_dynamic(&mut output),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-ANGLE-CONVERSION-INPUT",
                format!("{name} requires real or complex floating-point input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }

    finish_fixed(entry, request, output, diagnostics)
}
