use crate::catalog::inference::argument_error;
use runmat_types::{
    broadcast_shape, CallRequest, InferenceDiagnostic, ResidencyFact, ShapeFact, ValueKindFact,
};

use super::options::Options;

pub(super) struct Plan {
    pub array_indices: Vec<usize>,
    pub output_shape: ShapeFact,
    pub uniform_output: Option<bool>,
    pub has_device_input: bool,
    pub diagnostics: Vec<InferenceDiagnostic>,
}

impl Plan {
    pub fn from_request(request: &CallRequest) -> Self {
        let mut diagnostics = Vec::new();
        if request.arguments.len() < 2 {
            diagnostics.push(argument_error(
                "RM-CATALOG-ARRAYFUN-ARITY",
                "arrayfun requires a callable and at least one array input",
                request.arguments.len().min(1),
            ));
        }
        if let Some(function) = request.arguments.first() {
            if !matches!(
                function.kind,
                ValueKindFact::Callable(_) | ValueKindFact::Unknown
            ) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ARRAYFUN-FUNCTION",
                    "arrayfun requires a callable first argument",
                    0,
                ));
            }
        }

        let options = Options::parse(request);
        diagnostics.extend(options.diagnostics);
        let array_indices = (1..options.start).collect::<Vec<_>>();
        if array_indices.is_empty() && request.arguments.len() >= 2 {
            diagnostics.push(argument_error(
                "RM-CATALOG-ARRAYFUN-INPUT",
                "arrayfun requires at least one array before its options",
                1,
            ));
        }
        let output_shape = output_shape(request, &array_indices, &mut diagnostics);
        let has_device_input = array_indices.iter().any(|index| {
            request
                .arguments
                .get(*index)
                .is_some_and(|argument| matches!(argument.residency, ResidencyFact::Device { .. }))
        });
        Self {
            array_indices,
            output_shape,
            uniform_output: options.uniform_output,
            has_device_input,
            diagnostics,
        }
    }
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
    indices
        .iter()
        .skip(1)
        .fold(first.shape.clone(), |shape, index| {
            let Some(next) = request.arguments.get(*index) else {
                return ShapeFact::Unknown;
            };
            match broadcast_shape(&shape, &next.shape) {
                Ok(shape) => shape,
                Err(_) => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-ARRAYFUN-SIZE",
                        "arrayfun inputs do not have compatible sizes",
                        *index,
                    ));
                    ShapeFact::Unknown
                }
            }
        })
}
