//! Domain-owned conversion between state-space models and control polynomials.

mod ss_conversion;

pub(crate) use ss_conversion::ss_object_to_tf_async;
