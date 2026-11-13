//! One-hot decoding output facts.
use super::common::{
    error, finish, literal_dim, output_selection, selected_fact, EncodingInferenceFlavor,
    OutputSelection,
};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, ShapeFact, ValueKindFact};

pub(in super::super) fn decode(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 3 {
        diagnostics.push(error(
            "ARITY",
            "onehotdecode requires encoded values, classes, and featureDim",
        ));
    }
    let axis = match literal_dim(request.literals.literal_args.get(2)) {
        Ok(axis) => axis,
        Err(()) => {
            diagnostics.push(error(
                "DIMENSION",
                "featureDim must be a positive integer scalar",
            ));
            None
        }
    };
    let selection = output_selection(request, 3, EncodingInferenceFlavor::Decode);
    if matches!(selection, OutputSelection::Invalid) {
        diagnostics.push(error(
            "OPTION",
            "invalid onehotdecode output type or option",
        ));
    }
    let shape = match (request.arguments.first(), axis) {
        (Some(input), Some(axis)) => {
            let admitted_shape = if matches!(
                input.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical
            ) {
                input.shape.known_dims()
            } else {
                None
            };
            admitted_shape
                .and_then(|mut dims| {
                    if axis > dims.len() {
                        diagnostics.push(error(
                            "DIMENSION",
                            "featureDim exceeds the encoded array rank",
                        ));
                        return None;
                    }
                    dims[axis - 1] = Some(1);
                    Some(ShapeFact::from(dims))
                })
                .unwrap_or(ShapeFact::Unknown)
        }
        _ => ShapeFact::Unknown,
    };
    finish(request, entry, selected_fact(selection, shape), diagnostics)
}
