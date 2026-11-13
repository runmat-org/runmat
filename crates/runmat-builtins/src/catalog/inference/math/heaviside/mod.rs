use super::super::support::facts::materialize;
use super::super::{argument_error, finish_fixed, numeric_kind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, ResidencyFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub(in crate::catalog::inference) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-HEAVISIDE-ARITY",
            "heaviside requires exactly one input",
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

    let mut output = input.clone();
    match input.kind {
        ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Complex => {
            diagnostics.push(argument_error(
                "RM-CATALOG-HEAVISIDE-INPUT",
                "heaviside requires real numeric, logical, character, or symbolic input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric) => {
            let class = if matches!(numeric.class, NumericClass::Double | NumericClass::Single) {
                numeric.class
            } else {
                NumericClass::Double
            };
            output.kind = numeric_kind(class, NumericDomain::Real);
            if matches!(input.storage, StorageFact::Sparse) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-HEAVISIDE-SPARSE",
                    "heaviside does not currently accept sparse input",
                    0,
                ));
                output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
            } else {
                if class != numeric.class && matches!(input.residency, ResidencyFact::Device { .. })
                {
                    output.residency = ResidencyFact::Unknown;
                }
                materialize(&mut output);
            }
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            if matches!(input.residency, ResidencyFact::Device { .. }) {
                output.residency = ResidencyFact::Unknown;
            }
            materialize(&mut output);
        }
        ValueKindFact::Symbolic => {}
        ValueKindFact::Unknown => {
            let shape = output.shape.clone();
            output = ValueFact::unknown(DynamicReason::RuntimeValue);
            output.shape = shape;
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-HEAVISIDE-INPUT",
                "heaviside requires real numeric, logical, character, or symbolic input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests;
