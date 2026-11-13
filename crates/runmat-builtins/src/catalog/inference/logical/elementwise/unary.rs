use crate::{BuiltinCatalogEntry, LogicalUnaryOperator};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    operator: LogicalUnaryOperator,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(super::super::super::argument_error(
            "RM-CATALOG-LOGICAL-UNARY-ARITY",
            format!("{} requires exactly one input", operator.name()),
            request.arguments.len().min(1),
        ));
    }
    let Some(input) = request.arguments.first() else {
        return super::super::super::finish_fixed(
            entry,
            request,
            super::output::unknown(),
            diagnostics,
        );
    };
    if super::output::invalid(input) {
        diagnostics.push(super::super::super::argument_error(
            "RM-CATALOG-LOGICAL-INPUT",
            "unsupported logical operand",
            0,
        ));
    }
    if super::output::may_be_tabular(input) {
        return super::super::super::finish_fixed(
            entry,
            request,
            super::output::unknown(),
            diagnostics,
        );
    }
    let output = super::output::tabular(input).unwrap_or_else(|| {
        let mut output = super::output::logical(input.shape.clone());
        output.residency = super::output::residency(input);
        output
    });
    super::super::super::finish_fixed(entry, request, output, diagnostics)
}
