use crate::Value;

mod cell;
mod structure;

pub use cell::CellArray;
pub(crate) use cell::{shape_rows_cols, total_len};
pub use structure::StructValue;
