mod call;
mod residency;
#[cfg(test)]
mod tests;

pub(in crate::catalog::inference) use call::{
    infer_numeric_conversion_call, infer_numeric_conversion_with_like_call,
    infer_single_fusion_fact,
};
