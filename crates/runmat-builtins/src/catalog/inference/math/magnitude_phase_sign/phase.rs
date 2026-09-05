use super::super::super::support::facts::{materialize, preserve_shape_as_dynamic};
use super::super::super::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-PHASE-ANGLE-ARITY",
            "angle requires exactly one input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-PHASE-ANGLE-ARITY",
            "angle accepts exactly one input",
            1,
        ));
    }
    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-PHASE-ANGLE-SPARSE",
            "angle does not currently accept sparse input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let mut output = input.clone();
    match &mut output.kind {
        ValueKindFact::Numeric(numeric)
            if matches!(numeric.class, NumericClass::Double | NumericClass::Single) =>
        {
            numeric.domain = NumericDomain::Real;
            materialize(&mut output);
        }
        ValueKindFact::Unknown => preserve_shape_as_dynamic(&mut output),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-PHASE-ANGLE-INPUT",
                "angle requires real or complex single- or double-precision input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}
