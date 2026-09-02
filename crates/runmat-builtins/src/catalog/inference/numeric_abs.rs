use super::{argument_error, finish_fixed, numeric_kind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, ValueFact,
    ValueKindFact,
};

pub(super) fn infer_abs(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut output = request.arguments.first().cloned().unwrap_or_else(|| {
        diagnostics.push(argument_error(
            "RM-CATALOG-ABS-ARITY",
            "abs requires exactly one input",
            0,
        ));
        ValueFact::unknown(DynamicReason::RuntimeValue)
    });
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ABS-ARITY",
            "abs accepts exactly one input",
            1,
        ));
    }
    match &mut output.kind {
        ValueKindFact::Numeric(numeric) => numeric.domain = NumericDomain::Real,
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
        }
        ValueKindFact::Symbolic | ValueKindFact::Unknown => {}
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-ABS-INPUT",
                "abs requires numeric, logical, character, or symbolic input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}
