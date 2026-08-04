//! Sorting and set-related array builtins.

use runmat_accelerate_api::{GpuTensorHandle, IntegerElementType};

pub mod argsort;
pub(super) mod float_order;
pub(super) mod integer_order;
pub mod intersect;
pub mod ismember;
pub mod ismembertol;
pub mod issorted;
pub mod issortedrows;
pub mod setdiff;
pub mod setxor;
pub mod sort;
pub mod sortrows;
pub(crate) mod type_resolvers;
pub mod union;
pub mod unique;

pub(super) fn is_unsupported_set_gpu_integer(handle: &GpuTensorHandle) -> bool {
    matches!(
        runmat_accelerate_api::handle_integer_type(handle),
        Some(IntegerElementType::I64 | IntegerElementType::U64)
    )
}
