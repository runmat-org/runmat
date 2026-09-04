//! Grouping-variable normalization shared by grouping operations.

mod categorical;
mod column;
mod labels;
mod numeric;
mod observations;
mod temporal;
mod text;

pub(crate) use column::{columns_from_arguments, columns_from_value, GroupColumn};
pub(crate) use labels::{label_columns, select_group_rows};
pub(crate) use observations::build_index;

pub(crate) type VariableResult<T> = Result<T, String>;
