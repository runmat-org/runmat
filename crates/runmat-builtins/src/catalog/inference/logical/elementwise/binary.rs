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

    let shape = match broadcast_shape(&left.shape, &right.shape) {
        Ok(shape) => shape,
        Err(diagnostic) => {
            diagnostics.push(diagnostic);
            ShapeFact::Unknown
        }
    };
    let mut output = super::output::logical(shape);
    output.residency =
        super::super::super::preserved_binary_residency(&left.residency, &right.residency);
    super::super::super::finish_fixed(entry, request, output, diagnostics)
}
