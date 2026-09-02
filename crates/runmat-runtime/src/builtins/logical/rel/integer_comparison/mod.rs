//! Exact relational comparison substrate for native numerical storage.

mod exact;
mod gpu;

#[derive(Clone, Copy)]
pub enum IntegerComparisonOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug)]
pub enum IntegerComparisonError {
    SizeMismatch,
    Internal,
}

pub(crate) use exact::{
    compare_integer_values, compare_numeric_scalars_exact, integer_f64_order,
    matches_optional_relation, matches_relation, storage_value,
    try_complex_integer_equality_comparison, try_complex_ordering_comparison,
    try_integer_comparison, try_real_ordering_comparison,
};
pub(crate) use gpu::{
    restore_explicit_comparison_result, select_comparison_output_source,
    try_gpu_equality_comparison, try_gpu_ordering_comparison,
};

#[cfg(test)]
mod tests;
