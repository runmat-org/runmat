use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, ShapeFact, ValueFact, ValueKindFact,
};

use super::super::super::{argument_error, finish_fixed, support};
use super::coefficient_class;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-NCHOOSEK-ARITY",
            "nchoosek requires exactly two inputs",
            request.arguments.len().min(1),
        ));
    }
    let Some(input) = request.arguments.first() else {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };

    let selection = request.arguments.get(1);
    let mut output = input.clone();
    match &output.kind {
        ValueKindFact::Numeric(numeric)
            if input.is_scalar() && numeric.domain == runmat_types::NumericDomain::Real =>
        {
            let class = match selection {
                Some(value) if value.is_scalar() => {
                    coefficient_class::resolve(input, value, &mut diagnostics)
                }
                Some(value) if !matches!(value.kind, ValueKindFact::Unknown) => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NCHOOSEK-SELECTION-SCALAR",
                        "nchoosek selection must be a numeric scalar",
                        1,
                    ));
                    None
                }
                _ => None,
            };
            output = match class {
                Some(class) => {
                    ValueFact::scalar(ValueKindFact::Numeric(runmat_types::NumericFact {
                        class,
                        domain: runmat_types::NumericDomain::Real,
                    }))
                }
                None => ValueFact::unknown(DynamicReason::RuntimeValue),
            };
        }
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Character => {
            output.shape = ShapeFact::Ranked { rank: 2 };
            support::facts::materialize(&mut output);
            output.residency = runmat_types::ResidencyFact::Host;
        }
        ValueKindFact::Unknown => {
            output = ValueFact::unknown(DynamicReason::RuntimeValue);
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-NCHOOSEK-INPUT",
                "nchoosek requires a supported numeric, logical, or character input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}
