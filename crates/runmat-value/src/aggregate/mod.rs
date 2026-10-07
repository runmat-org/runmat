use crate::Value;

mod cell;
mod struct_array;
mod structure;

pub use cell::CellArray;
pub(crate) use cell::{shape_rows_cols, total_len};
pub use struct_array::{
    StructArray, StructArrayAdmission, StructArrayOperand, StructElementRef, StructFieldsRef,
    ValidatedStructArrayParts,
};
pub use structure::StructValue;
