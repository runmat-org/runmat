mod compute;
mod geometry;

pub(super) use compute::{covariance_dense, covariance_pairwise, filter_complete_rows};
pub(super) use geometry::{
    combine_tensors, covariance_column_count, covariance_element_count, covariance_input_shape,
    validate_covariance_pair_shapes,
};
