use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry, EquationProblemProperty,
};
use runmat_types::*;
pub(super) fn problem(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    finish_fixed(
        entry,
        request,
        ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(crate::EQUATION_PROBLEM_CLASS.owned()),
                properties: Default::default(),
                properties_complete: false,
                handle_semantics: Some(false),
            }),
            ShapeFact::Scalar,
            StorageFact::Opaque,
        ),
        Vec::new(),
    )
}
pub(super) fn read(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    finish_fixed(
        entry,
        request,
        ValueFact::unknown(DynamicReason::RuntimeValue),
        Vec::new(),
    )
}

pub(super) fn constructor(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut out = problem(request, entry);
    if !request.arguments.len().is_multiple_of(2) {
        out.diagnostics.push(argument_error(
            "RM-CATALOG-EQNPROBLEM-PAIRS",
            "eqnproblem expects property name/value pairs",
            request.arguments.len().saturating_sub(1),
        ));
        return out;
    }
    let mut seen = [false; EquationProblemProperty::ALL.len()];
    for index in (0..request.arguments.len()).step_by(2) {
        let Some(name) = request.literals.literal_string_at(index) else {
            continue;
        };
        let Some(property) = EquationProblemProperty::parse_name_value(&name) else {
            out.diagnostics.push(argument_error(
                "RM-CATALOG-EQNPROBLEM-PROPERTY",
                "unknown equation problem property",
                index,
            ));
            continue;
        };
        let slot = EquationProblemProperty::ALL
            .iter()
            .position(|known| *known == property)
            .expect("canonical property");
        if seen[slot] || property == EquationProblemProperty::Variables {
            out.diagnostics.push(argument_error(
                "RM-CATALOG-EQNPROBLEM-PROPERTY",
                "duplicate or read-only equation problem property",
                index,
            ));
        }
        seen[slot] = true;
    }
    out
}
