use crate::catalog::inference::argument_error;
use runmat_types::{CallRequest, InferenceDiagnostic, ShapeFact, ValueKindFact};

use super::options::Options;

pub(super) struct Plan {
    pub(super) cell_indices: Vec<usize>,
    pub(super) extra_indices: Vec<usize>,
    pub(super) output_shape: ShapeFact,
    pub(super) uniform_output: Option<bool>,
    pub(super) diagnostics: Vec<InferenceDiagnostic>,
}

impl Plan {
    pub(super) fn from_request(request: &CallRequest) -> Self {
        let options = Options::parse(request);
        let mut diagnostics = options.diagnostics;
        if request.arguments.len() < 2 {
            diagnostics.push(argument_error(
                "RM-CATALOG-CELLFUN-ARITY",
                "cellfun requires a callable and at least one cell array",
                request.arguments.len().min(1),
            ));
        }
        validate_callable(request, &mut diagnostics);
        let mut cell_indices = Vec::new();
        let mut extra_indices = Vec::new();
        let mut constants_started = false;
        for index in 1..options.start {
            let Some(argument) = request.arguments.get(index) else {
                continue;
            };
            match argument.kind {
                ValueKindFact::Cell(_) if constants_started => diagnostics.push(argument_error(
                    "RM-CATALOG-CELLFUN-CELL-ORDER",
                    "cellfun cell arrays must precede constant callback arguments",
                    index,
                )),
                ValueKindFact::Cell(_) => cell_indices.push(index),
                ValueKindFact::Unknown if !constants_started => cell_indices.push(index),
                _ => {
                    constants_started = true;
                    extra_indices.push(index);
                }
            }
        }
        if cell_indices.is_empty() && request.arguments.len() >= 2 {
            diagnostics.push(argument_error(
                "RM-CATALOG-CELLFUN-INPUT",
                "cellfun requires at least one cell array before constant arguments and options",
                1,
            ));
        }
        let output_shape = output_shape(request, &cell_indices, &mut diagnostics);
        Self {
            cell_indices,
            extra_indices,
            output_shape,
            uniform_output: options.uniform_output,
            diagnostics,
        }
    }
}

fn validate_callable(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let Some(value) = request.arguments.first() else {
        return;
    };
    if matches!(
        value.kind,
        ValueKindFact::Callable(_)
            | ValueKindFact::String
            | ValueKindFact::Character
            | ValueKindFact::Unknown
    ) {
        return;
    }
    diagnostics.push(argument_error(
        "RM-CATALOG-CELLFUN-CALLABLE",
        "cellfun requires a function handle or function name",
        0,
    ));
}

fn output_shape(
    request: &CallRequest,
    indices: &[usize],
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ShapeFact {
    let Some(first) = indices
        .first()
        .and_then(|index| request.arguments.get(*index))
    else {
        return ShapeFact::Unknown;
    };
    for index in indices.iter().skip(1) {
        let Some(next) = request.arguments.get(*index) else {
            continue;
        };
        if known_mismatch(&first.shape, &next.shape) {
            diagnostics.push(argument_error(
                "RM-CATALOG-CELLFUN-SIZE",
                "cellfun cell arrays must have the same size",
                *index,
            ));
        }
    }
    first.shape.clone()
}

fn known_mismatch(left: &ShapeFact, right: &ShapeFact) -> bool {
    match (left, right) {
        (ShapeFact::Scalar, ShapeFact::Scalar) => false,
        (ShapeFact::Shaped { dims: left }, ShapeFact::Shaped { dims: right }) => {
            left.len() != right.len()
                || left.iter().zip(right).any(|(left, right)| {
                    matches!(
                        (left, right),
                        (
                            runmat_types::DimensionFact::Known(left),
                            runmat_types::DimensionFact::Known(right)
                        ) if left != right
                    )
                })
        }
        (ShapeFact::Unknown, _) | (_, ShapeFact::Unknown) => false,
        _ => true,
    }
}
