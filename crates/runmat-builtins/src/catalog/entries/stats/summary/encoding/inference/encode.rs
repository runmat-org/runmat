//! One-hot encoding output facts.
use super::common::{
    error, finish, literal_dim, output_selection, selected_fact, EncodingInferenceFlavor,
    OutputSelection,
};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, ShapeFact, ValueKindFact};

pub(in super::super) fn encode(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(error(
            "ARITY",
            "onehotencode requires labels and featureDim",
        ));
    }
    let axis = match literal_dim(request.literals.literal_args.get(1)) {
        Ok(axis) => axis,
        Err(()) => {
            diagnostics.push(error(
                "DIMENSION",
                "featureDim must be a positive integer scalar",
            ));
            None
        }
    };
    let selection = output_selection(request, 2, EncodingInferenceFlavor::Encode);
    if matches!(selection, OutputSelection::Invalid) {
        diagnostics.push(error(
            "OPTION",
            "invalid onehotencode output type or option",
        ));
    }
    let shape = match (request.arguments.first(), axis) {
        (Some(input), Some(axis)) => {
            // Character arrays encode one label per row, and categorical
            // objects obtain their geometry from Codes. Their outer fact
            // shape is not the label shape.
            let mut dims = if matches!(
                input.kind,
                ValueKindFact::Numeric(_)
                    | ValueKindFact::Logical
                    | ValueKindFact::String
                    | ValueKindFact::Cell(_)
            ) {
                input.shape.known_dims()
            } else {
                None
            };
            if let Some(ref mut dims) = dims {
                if axis <= 32 {
                    dims.resize(axis.max(dims.len()), Some(1));
                    if dims[axis - 1] != Some(1) {
                        diagnostics.push(error(
                            "DIMENSION",
                            "featureDim must identify a singleton dimension",
                        ));
                    }
                    dims[axis - 1] = None; // ClassNames/category count is value-dependent.
                } else {
                    diagnostics.push(error("DIMENSION", "featureDim is too large"));
                    dims.clear();
                }
            }
            dims.filter(|dims| !dims.is_empty())
                .map_or(ShapeFact::Unknown, ShapeFact::from)
        }
        _ => ShapeFact::Unknown,
    };
    finish(request, entry, selected_fact(selection, shape), diagnostics)
}
