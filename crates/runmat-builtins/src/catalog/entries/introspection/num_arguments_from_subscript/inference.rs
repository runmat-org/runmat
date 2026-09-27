use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 3 {
        diagnostics.push(argument_error(
            "RM-INTROSPECTION-ARITY",
            "expected 3 inputs",
            0,
        ));
    }

    let default_route = request.arguments.first().is_some_and(|target| {
        !matches!(
            target.kind,
            ValueKindFact::Object(_) | ValueKindFact::Unknown
        )
    });
    if let Some(path) = request.arguments.get(1).filter(|_| default_route) {
        if !matches!(path.kind, ValueKindFact::Unknown | ValueKindFact::Struct(_))
            || path.shape.element_count() == Some(0)
        {
            diagnostics.push(argument_error(
                "RM-INDEXING-PATH",
                "expected a nonempty standard substruct path",
                1,
            ));
        }
    }

    let output = match request.arguments.first().map(|a| &a.kind) {
        Some(ValueKindFact::Object(_) | ValueKindFact::Unknown) | None => {
            ValueFact::unknown(DynamicReason::DynamicDispatch)
        }
        _ => ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
    };
    if let Some(context) = request.literals.literal_string_at(2) {
        if !matches!(
            context
                .trim()
                .rsplit('.')
                .next()
                .unwrap_or("")
                .to_ascii_lowercase()
                .as_str(),
            "statement" | "expression" | "assignment"
        ) {
            diagnostics.push(argument_error(
                "RM-INDEXING-CONTEXT",
                "expected Statement, Expression or Assignment",
                2,
            ));
        }
    }
    let mut contract = CallContract::fixed(vec![output]);

    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
