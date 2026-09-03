use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    AliasFact, CallInference, CallRequest, DynamicReason, MutationFact, NumericDomain, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer_get(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, PositionOperation::Get)
}

pub(super) fn infer_set(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, PositionOperation::Set)
}

#[derive(Clone, Copy)]
enum PositionOperation {
    Get,
    Set,
}

impl PositionOperation {
    const fn name(self) -> &'static str {
        match self {
            Self::Get => "bitget",
            Self::Set => "bitset",
        }
    }
    const fn maximum_arity(self) -> usize {
        match self {
            Self::Get => 3,
            Self::Set => 4,
        }
    }
    const fn value_argument_count(self) -> usize {
        match self {
            Self::Get => 2,
            Self::Set => 3,
        }
    }
}

fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    operation: PositionOperation,
) -> CallInference {
    let mut diagnostics = Vec::new();
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
    if !matches!(value.kind, ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real)
    {
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
    }
    let shape = scalar_or_equal_shape(
        request
            .arguments
            .iter()
            .take(operation.value_argument_count())
            .map(|argument| &argument.shape),
    );
    let mut output = value.clone();
    output.shape = shape;
    output.storage = if output.shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else {
        StorageFact::Unknown
    };
    output.alias = AliasFact::Unique;
    output.mutation = MutationFact::ValueSemantics;
    finish_fixed(entry, request, output, diagnostics)
}

fn scalar_or_equal_shape<'a>(shapes: impl Iterator<Item = &'a ShapeFact>) -> ShapeFact {
    let mut output = ShapeFact::Scalar;
    for shape in shapes {
        if shape.element_count() == Some(1) {
            continue;
        }
        if matches!(output, ShapeFact::Scalar) {
            output = shape.clone();
        } else if output != *shape {
            return ShapeFact::Unknown;
        }
    }
    output
}
