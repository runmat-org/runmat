mod parameters;
mod errors;
mod descriptors;
mod integers;
mod documentation;
pub(super) use descriptors::{moving_descriptor, moving_stdvar_descriptor};
pub(super) use documentation::{
    MOVING_WINDOW_CLASSES_TEXT, MOVING_WINDOW_CONTROLS_TEXT, MOVING_WINDOW_ERRORS_TEXT,
    MOVING_WINDOW_PERFORMANCE_TEXT,
};
pub use errors::*;
pub(super) use integers::{
    moving_gpu_sample_points_extension, moving_integer_metadata, moving_stdvar_integer_metadata,
};
pub(super) use parameters::*;
