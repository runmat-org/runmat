//! Deep Learning Toolbox compatibility builtins.

mod arguments;
pub(crate) mod autodiff;
mod class_registration;
mod contracts;
mod errors;
pub(crate) mod graph;
mod layer_helpers;
mod layer_identity;
pub(crate) mod layers;
pub(crate) mod losses;
pub(crate) mod model;
pub(crate) mod onnx;
mod prelude;
pub(crate) mod sequences;
pub(crate) mod supervised;
pub(crate) mod training;

pub(super) use arguments::{
    logical_scalar, nonnegative_usize, numeric_scalar, numeric_values, numeric_vector, object,
    object_with_identity, positive_i64, positive_usize, scalar_text, string_array, tensor_value,
    text_or_missing,
};
pub(super) use class_registration::ensure_dlarray_class_registered;
pub use contracts::{
    ADAMUPDATE_DESCRIPTOR, ARRAY_DESCRIPTOR, DLFEVAL_DESCRIPTOR, DLGRADIENT_DESCRIPTOR,
    DLUPDATE_DESCRIPTOR, OBJECT_DESCRIPTOR,
};
pub(super) use contracts::{
    CLASSIFICATION_OUTPUT_LAYER_CLASS, DLARRAY_CLASS, ELU_LAYER_CLASS, FEATURE_INPUT_LAYER_CLASS,
    FULLY_CONNECTED_LAYER_CLASS, LAYER_GRAPH_CLASS, MAX_COMBVEC_COLUMNS, MAX_PAD_ELEMENTS,
    REGRESSION_OUTPUT_LAYER_CLASS, RELU_LAYER_CLASS, SOFTMAX_LAYER_CLASS, TRAINING_OPTIONS_CLASS,
};
pub(super) use errors::{any_type, deep_learning_error, gather_args, unsupported_error};
pub(super) use layer_helpers::{layer_names, layer_object, layers_from_value, parse_name_values};
use prelude::*;

#[cfg(test)]
mod tests;
