use crate::aggregate::{shape_rows_cols, total_len};
use crate::*;
use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::fmt;

mod character;
mod complex;
mod conversion;
mod dense;
mod format;
mod logical;
mod sparse;
mod string;
mod symbolic;

pub use character::CharArray;
pub use complex::{ComplexStorage, ComplexTensor, IntegerComplexStorage};
pub use dense::Tensor;
pub(crate) use format::{format_integer_complex_value, should_expand_nd_display, write_nd_pages};
pub use logical::LogicalArray;
pub use sparse::SparseTensor;
pub use string::StringArray;
pub use symbolic::SymbolicArray;
