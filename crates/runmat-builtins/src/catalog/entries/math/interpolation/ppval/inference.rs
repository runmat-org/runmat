use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, CertaintyFact, DynamicReason,
    NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-PPVAL-ARITY",
            "ppval requires a piecewise-polynomial struct and query points",
            0,
        ));
    }
    if let Some(pp) = request.arguments.first() {
        if !matches!(pp.kind, ValueKindFact::Struct(_) | ValueKindFact::Unknown) {
            diagnostics.push(argument_error(
                "RM-CATALOG-PPVAL-PP",
                "ppval requires a piecewise-polynomial struct",
                0,
            ));
        }
    }
    if let Some(query) = request.arguments.get(1) {
        if !matches!(
            query.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Logical
                | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-PPVAL-QUERY",
                "ppval query points must be real numeric or logical",
                1,
            ));
        }
    }
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::Unknown,
        StorageFact::Dense,
    );
    // The pp.dim field is runtime data. A vector of query points can therefore
    // yield a higher-rank array, and its shape is not proved by Xq alone.
    output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(diagnostics);
    inferred
}
