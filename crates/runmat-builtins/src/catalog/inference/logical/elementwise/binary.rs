use crate::{BuiltinCatalogEntry, LogicalBinaryOperator};
use runmat_types::{broadcast_shape, CallInference, CallRequest, ShapeFact};

pub(super) fn infer(
    operator: LogicalBinaryOperator,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(super::super::super::argument_error(
            "RM-CATALOG-LOGICAL-BINARY-ARITY",
            format!("{} requires exactly two inputs", operator.name()),
            request.arguments.len().min(1),
        ));
    }
    let (Some(left), Some(right)) = (request.arguments.first(), request.arguments.get(1)) else {
        return super::super::super::finish_fixed(
            entry,
            request,
            super::output::unknown(),
            diagnostics,
        );
    };

    for (index, input) in [left, right].into_iter().enumerate() {
        if super::output::invalid(input) {
            diagnostics.push(super::super::super::argument_error(
                "RM-CATALOG-LOGICAL-INPUT",
                "unsupported logical operand",
                index,
            ));
        }
    }
    let left_tabular = super::output::tabular(left);
    let right_tabular = super::output::tabular(right);
    if let (Some(left_class), Some(right_class)) = (
        super::output::tabular_class(left),
        super::output::tabular_class(right),
    ) {
        if left_class != right_class {
            diagnostics.push(super::super::super::argument_error(
                "RM-CATALOG-LOGICAL-TABULAR-CLASS",
                "tabular logical operands must use the same container class",
                1,
            ));
        }
    }
    if let Some(output) = left_tabular.or(right_tabular) {
        return super::super::super::finish_fixed(entry, request, output, diagnostics);
    }
    if super::output::may_be_tabular(left) || super::output::may_be_tabular(right) {
        return super::super::super::finish_fixed(
            entry,
            request,
            super::output::unknown(),
            diagnostics,
        );
    }

    let shape = match broadcast_shape(&left.shape, &right.shape) {
        Ok(shape) => shape,
        Err(diagnostic) => {
            diagnostics.push(diagnostic);
            ShapeFact::Unknown
        }
    };
    let mut output = super::output::logical(shape);
    output.residency = if super::output::residency(left) == runmat_types::ResidencyFact::Host
        && super::output::residency(right) == runmat_types::ResidencyFact::Host
    {
        runmat_types::ResidencyFact::Host
    } else {
        runmat_types::ResidencyFact::Unknown
    };
    super::super::super::finish_fixed(entry, request, output, diagnostics)
}
