use crate::aggregate::{shape_rows_cols, total_len};
use crate::*;
use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::fmt;

mod admission;
mod character;
mod complex;
mod conversion;
mod copy_accounting;
mod dense;
mod exact_host_numeric;
mod format;
mod host_allocation;
mod host_buffer;
mod host_complex_buffer;
mod host_index_buffer;
mod host_logical_buffer;
mod host_numeric_buffer;
mod logical;
mod sparse;
mod string;
mod symbolic;

pub(crate) use admission::BorrowedShapeAdmission;
pub use admission::{
    ComplexTensorAdmission, NumericTensorAdmission, ValidatedComplexTensorParts,
    ValidatedNumericTensorParts,
};
pub use character::CharArray;
pub use complex::{ComplexElement, ComplexStorage, ComplexTensor, IntegerComplexStorage};
pub use copy_accounting::{host_copy_metrics, record_host_copy, HostCopyMetrics, HostCopyReason};
pub use dense::Tensor;
pub use exact_host_numeric::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostAllocationReservation,
    ExactHostNumericAllocation, ExactHostNumericAllocationError, ExactHostTensor2dAllocation,
};
pub(crate) use format::{format_integer_complex_value, should_expand_nd_display, write_nd_pages};
pub use host_allocation::{AdoptedHostAllocation, HostAllocationProvenance, HostAllocationRelease};
pub use host_complex_buffer::HostComplexBuffer;
pub use host_index_buffer::HostIndexBuffer;
pub use host_logical_buffer::HostLogicalBuffer;
pub use host_numeric_buffer::HostNumericBuffer;
pub use logical::LogicalArray;
pub use sparse::{SparseCscValidation, SparseTensor, ValidatedSparseCscParts};
pub use string::StringArray;
pub use symbolic::SymbolicArray;
