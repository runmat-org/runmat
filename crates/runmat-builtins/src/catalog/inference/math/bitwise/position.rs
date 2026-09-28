use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    AliasFact, CallInference, CallRequest, DynamicReason, MutationFact, ShapeFact, ValueFact,
    ValueKindFact,
};

mod operation;

use operation::PositionOperation;

pub(super) fn infer_get(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, PositionOperation::Get)
}

pub(super) fn infer_set(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, PositionOperation::Set)
}

fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    operation: PositionOperation,
) -> CallInference {
    let mut diagnostics = Vec::new();
    diagnostics.extend(super::options::numeric_control(request, 1));
    match operation {
        PositionOperation::Get => {
            diagnostics.extend(super::options::assumed_type(request, 2, &[0]))
        }
        PositionOperation::Set => {
            if request.arguments.len() == 3
                && matches!(
                    request.arguments[2].kind,
                    ValueKindFact::String | ValueKindFact::Character
                )
            {
                diagnostics.extend(super::options::assumed_type(request, 2, &[0]));
            } else {
                diagnostics.extend(super::options::numeric_control(request, 2));
                diagnostics.extend(super::options::assumed_type(request, 3, &[0]));
            }
        }
    }
    let name = operation.name();
    let max = operation.maximum_arity();
    if !(2..=max).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-BIT-POSITION-ARITY",
            format!("{name} accepts 2 to {max} inputs"),
            request.arguments.len().min(2),
        ));
    }
    let Some(value) = request.arguments.first() else {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    let Some(kind) = super::facts::result_kind(value) else {
        if !matches!(value.kind, ValueKindFact::Unknown) {
            diagnostics.push(argument_error(
                "RM-CATALOG-BIT-POSITION-INPUT",
                format!("{name} requires a real integer-valued numeric data input"),
                0,
            ));
        }
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    };
    let values = if matches!(operation, PositionOperation::Set)
        && request.arguments.len() == 3
        && matches!(
            request.arguments[2].kind,
            ValueKindFact::String | ValueKindFact::Character
        ) {
        2
    } else {
        operation.value_argument_count()
    };
    let mut shape = scalar_or_equal_shape(
        request
            .arguments
            .iter()
            .take(values)
            .map(|argument| &argument.shape),
    );
    if matches!(operation, PositionOperation::Set)
        && request.arguments.len() == 3
        && matches!(request.arguments[2].kind, ValueKindFact::Unknown)
    {
        // The third argument may be an assumed type or a shape-bearing V.
        shape = ShapeFact::Unknown;
    }
    let mut output = ValueFact::proven(kind, shape.clone(), super::facts::storage(value, &shape));
    output.residency = value.residency.clone();
    output.alias = AliasFact::Unique;
    output.mutation = MutationFact::ValueSemantics;
    finish_fixed(entry, request, output, diagnostics)
}

fn scalar_or_equal_shape<'a>(shapes: impl Iterator<Item = &'a ShapeFact>) -> ShapeFact {
    let mut output = ShapeFact::Scalar;
    for shape in shapes {
        output = super::facts::scalar_or_equal(&output, shape);
    }
    output
}
