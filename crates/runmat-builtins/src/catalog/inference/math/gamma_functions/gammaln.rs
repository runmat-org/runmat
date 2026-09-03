use super::common::finish_materialized_real;
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    AliasFact, CallInference, CallRequest, DynamicReason, MutationFact, NumericClass,
    NumericDomain, StorageFact, ValueFact, ValueKindFact, ViewFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-GAMMALN-ARITY",
            "gammaln requires exactly one input",
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
    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-GAMMALN-SPARSE",
            "gammaln does not accept sparse input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let output_class = match input.kind {
        ValueKindFact::Numeric(numeric)
            if numeric.domain == NumericDomain::Real
                && matches!(numeric.class, NumericClass::Single | NumericClass::Double) =>
        {
            numeric.class
        }
        ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real => {
            NumericClass::Double
        }
        ValueKindFact::Logical | ValueKindFact::Character => NumericClass::Double,
        ValueKindFact::Unknown => {
            let mut output = input.clone();
            output.kind = ValueKindFact::Unknown;
            output.storage = StorageFact::Unknown;
            output.alias = AliasFact::Unique;
            output.view = ViewFact::Materialized;
            output.mutation = MutationFact::ValueSemantics;
            return finish_fixed(entry, request, output, diagnostics);
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-GAMMALN-INPUT",
                "gammaln requires real numeric input; RunMat mode also accepts logical and character input",
                0,
            ));
            return finish_fixed(
                entry,
                request,
                ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
                diagnostics,
            );
        }
    };

    finish_materialized_real(entry, request, input, output_class, diagnostics)
}
