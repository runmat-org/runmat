//! Conservative output and rejection facts for admitted Q-Q plot forms.
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn handles() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::Unknown,
        StorageFact::Unknown,
    )
}

fn admissible_sample(fact: &ValueFact, optional_empty_y: bool) -> Result<(), &'static str> {
    match &fact.kind {
        ValueKindFact::Numeric(number) if number.domain == NumericDomain::Real => {}
        ValueKindFact::Logical | ValueKindFact::Unknown => {}
        _ => return Err("qqplot requires real numeric sample data"),
    }
    if let Some(count) = fact.shape.element_count() {
        if count == 0 && !optional_empty_y {
            return Err("qqplot requires a nonempty sample");
        }
    }
    if let Some(dims) = fact.shape.known_dims() {
        if dims.len() > 2
            && dims
                .iter()
                .filter(|extent| extent.is_some_and(|n| n > 1))
                .count()
                > 1
        {
            return Err("qqplot samples must be vectors or 2-D matrices");
        }
    }
    Ok(())
}

fn admissible_percentages(fact: &ValueFact) -> Result<(), &'static str> {
    match &fact.kind {
        ValueKindFact::Numeric(number) if number.domain == NumericDomain::Real => {}
        ValueKindFact::Logical | ValueKindFact::Unknown => {}
        _ => return Err("qqplot percentages must be real numeric values"),
    }
    if fact.shape.element_count() == Some(0) {
        return Err("qqplot percentages must be nonempty");
    }
    // Runtime accepts an arbitrary numeric percentage shape and linearizes
    // it; do not prove a narrower vector-only contract here.
    Ok(())
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let args = &request.arguments;
    let mut diagnostics = Vec::new();
    if args.is_empty() || args.len() > 4 {
        diagnostics.push(InferenceDiagnostic::error(
            "ARITY",
            "qqplot requires a sample and accepts at most three sample arguments plus axes",
        ));
    }
    if let Some(first) = args.first() {
        // Scalar numeric values may be axes handles. A known non-scalar first
        // operand cannot be axes, so positional sample roles are unambiguous.
        let first_is_sample = first.shape.element_count().is_some_and(|n| n != 1);
        if let Err(message) = admissible_sample(first, false) {
            diagnostics.push(InferenceDiagnostic::error("SAMPLE", message));
        }
        if first_is_sample {
            if args.len() > 3 {
                diagnostics.push(InferenceDiagnostic::error(
                    "ARITY",
                    "qqplot accepts at most x, y and pvec without an axes handle",
                ));
            }
            if let Some(second) = args.get(1) {
                if let Err(message) = admissible_sample(second, args.len() == 3) {
                    diagnostics.push(InferenceDiagnostic::error("SAMPLE", message));
                }
            }
            if let Some(third) = args.get(2) {
                if let Err(message) = admissible_percentages(third) {
                    diagnostics.push(InferenceDiagnostic::error("PVEC", message));
                }
            }
        }
    }
    let mut contract = CallContract::fixed(vec![if diagnostics.is_empty() {
        handles()
    } else {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    }]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
