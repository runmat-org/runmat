use super::math_fact_transforms::{materialize_output, preserve_shape_on_dynamic_input};
use super::{argument_error, finish_fixed, numeric_kind};
use crate::{BuiltinCatalogEntry, HyperbolicFunction};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer_hyperbolic(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    function: HyperbolicFunction,
) -> CallInference {
    let name = match function {
        HyperbolicFunction::Sine => "sinh",
        HyperbolicFunction::Cosine => "cosh",
        HyperbolicFunction::Tangent => "tanh",
    };
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-HYPERBOLIC-ARITY",
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
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-HYPERBOLIC-ARITY",
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
                "RM-CATALOG-HYPERBOLIC-COMPLEX-INTEGER",
                format!("{name} does not accept complex fixed-width integer input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric) => {
            if !matches!(numeric.class, NumericClass::Double | NumericClass::Single) {
                numeric.class = NumericClass::Double;
            }
            if matches!(output.storage, StorageFact::Sparse) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-HYPERBOLIC-SPARSE",
                    format!("{name} does not currently accept sparse input"),
                    0,
                ));
                output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
            } else {
                materialize_output(&mut output);
            }
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize_output(&mut output);
        }
        ValueKindFact::Unknown => preserve_shape_on_dynamic_input(&mut output),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-HYPERBOLIC-INPUT",
                format!("{name} requires numeric, logical, or character input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}
