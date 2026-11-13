use crate::aggregate::{shape_rows_cols, total_len};
use crate::*;
use std::convert::TryFrom;
use std::fmt;

mod admission;
mod character;
mod complex;
mod conversion;
mod copy_accounting;
mod dense;
mod exact_host_numeric;
mod exact_vec;
mod format;
mod host_allocation;
mod host_buffer;
mod host_complex_buffer;
mod host_index_buffer;
mod host_logical_buffer;
mod host_numeric_buffer;
mod logical;
mod shared_shape;
mod sparse;
mod string;
mod symbolic;

pub(in crate::array) use admission::{
    admit_shape, admit_shape_for_resize, shape_dimensions, ValidatedShape,
};
pub(crate) use admission::{BorrowedShapeAdmission, ShapeProgress};
pub use admission::{
    ComplexTensorAdmission, NumericTensorAdmission, ValidatedComplexTensorParts,
    ValidatedNumericTensorParts,
};
pub use character::{
    decode_utf16_code_units, encode_utf16_code_units, CharArray, CharArrayAdmission, Utf16CodeUnit,
    Utf16DecodeError,
};
pub use complex::{
    ComplexElement, ComplexStorage, ComplexTensor, IntegerComplexStorage, UniqueHostComplexMutation,
};
pub use copy_accounting::{host_copy_metrics, record_host_copy, HostCopyMetrics, HostCopyReason};
pub use dense::Tensor;
pub use exact_host_numeric::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostAllocationReservation,
    ExactHostComplexTensorAllocation, ExactHostIndexBufferAllocation,
    ExactHostIntegerComplexTensorAllocation, ExactHostLogicalTensorAllocation,
    ExactHostNumericAllocation, ExactHostNumericAllocationError, ExactHostSparseClass,
    ExactHostSparsePublicationError, ExactHostSparseTensorAllocation, ExactHostTensor2dAllocation,
    ExactHostTensorAllocation, PreparedExactSparseTensor,
};
pub use exact_vec::{exact_capacity_vec, exact_capacity_vec_from_slice, exact_capacity_vec_with};
pub(crate) use format::{format_integer_complex_value, should_expand_nd_display, write_nd_pages};
pub use host_allocation::{AdoptedHostAllocation, HostAllocationProvenance, HostAllocationRelease};
pub use host_complex_buffer::HostComplexBuffer;
pub use host_index_buffer::HostIndexBuffer;
pub use host_logical_buffer::HostLogicalBuffer;
pub use host_numeric_buffer::{HostNumericBuffer, UniqueHostNumericMutation};
pub use logical::LogicalArray;
pub use shared_shape::{HostShapeInput, SharedShape, SharedShapeBuilder, SharedShapeIntoIter};
pub use sparse::{SparseCscMergeEvent, SparseCscValidation, SparseTensor, ValidatedSparseCscParts};
pub use string::StringArray;
pub use symbolic::SymbolicArray;
