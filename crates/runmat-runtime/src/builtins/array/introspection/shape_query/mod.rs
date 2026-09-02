mod metadata;
mod output;
mod selectors;

pub(super) use metadata::VisibleDimensions;
pub(super) use output::{exact_double, row_vector, StructuralOutputError};
pub(super) use selectors::{
    parse_dimension_arguments, DimensionSelectorError, EmptySelectorPolicy,
};
