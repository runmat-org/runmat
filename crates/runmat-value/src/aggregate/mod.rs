use crate::Value;

mod cell;
mod struct_array;
mod structure;

pub use cell::CellArray;
pub(crate) use cell::{shape_rows_cols, total_len};
pub use struct_array::{
    StructArray, StructArrayAdmission, StructArrayBuilder, StructArrayConstructionError,
    StructArrayFieldError, StructArrayOperand, StructArrayQuote, StructElementRef, StructFieldsRef,
    UnboundedStructArrayBuilder, ValidatedStructArrayParts,
};
pub use structure::{
    StructFieldInsertError, StructValue, StructValueBuilder, UnboundedStructValueBuilder,
};
