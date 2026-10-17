//! Domain-owned conversion between state-space models and control polynomials.

mod ss_conversion;
mod series;

pub(crate) use series::SeriesModel;
pub(crate) use ss_conversion::ss_object_coefficient_grid_async;
pub(crate) use ss_conversion::ss_object_to_tf_async;
pub(crate) use ss_conversion::{
    allocation_error, preallocate_polynomial_buffers, selected_input_polynomials,
    SsPolynomialGeometry,
};
