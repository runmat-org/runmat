use super::super::program_facts as facts;
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(3..=7).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-LINPROG-ARITY",
            "linprog requires f, A, b and up to four equality/bound operands",
            0,
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        if matches!(argument.kind, ValueKindFact::Logical) {
            diagnostics.push(argument_error(
                "RM-CATALOG-LINPROG-CLASS",
                "linprog operands must be real numeric, not logical",
                index,
            ));
        }
    }
    facts::numeric(
        request,
        0,
        facts::OptimNumericInput::NonemptyVector,
        &mut diagnostics,
    );
    for index in [1, 3] {
        facts::numeric(
            request,
            index,
            facts::OptimNumericInput::OptionalMatrix,
            &mut diagnostics,
        );
    }
    for index in [2, 4, 5, 6] {
        facts::numeric(
            request,
            index,
            facts::OptimNumericInput::OptionalVector,
            &mut diagnostics,
        );
    }
    facts::constraint_geometry(request, 0, &[(1, 2), (3, 4)], &[5, 6], &mut diagnostics);
    if request.arguments.len() == 4
        && request.arguments[3]
            .shape
            .element_count()
            .is_some_and(|count| count != 0)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-LINPROG-EQUALITY",
            "Aeq requires a matching beq",
            3,
        ));
    }
    facts::finish(entry, request, facts::solver_outputs(false), diagnostics)
}
