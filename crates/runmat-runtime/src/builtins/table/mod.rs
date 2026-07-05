//! MATLAB table, timetable, categorical, and tabular workflow builtins.

pub(crate) mod builtins;
mod containers;
mod display;
mod import;
mod metadata;
mod names;
mod object;
mod parsing;
mod prelude;
mod registry;

#[cfg(test)]
use builtins::*;
pub use metadata::*;
pub use prelude::{TABLE_CLASS, TIMETABLE_CLASS};
pub use registry::ensure_table_class_registered;

use containers::*;
pub(crate) use containers::{
    categorical_categories, categorical_compare, categorical_extrema_to_value,
    categorical_from_args, categorical_labels, categorical_max_evaluate, categorical_min_evaluate,
    CategoricalComparison,
};
pub(crate) use display::categorical_label_at;
use display::format_key_number;
pub use display::{table_display_text, table_summary_text};
use import::*;
use names::*;
use object::*;
pub use object::{
    is_table_value, is_tabular_object, sortrows_table, table_from_columns, table_height,
    table_variable_names_from_object, table_variables, table_width,
};
pub(crate) use object::{
    select_rows, selected_row_names, table_from_columns_like, value_row_count,
};
use parsing::*;
pub(in crate::builtins::table) use prelude::*;

#[cfg(test)]
mod tests;
