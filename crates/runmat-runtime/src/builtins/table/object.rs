use super::*;

mod analytics;
mod indexing;
mod properties;
mod rows;
mod selectors;
mod structure;

pub use analytics::sortrows_table;
pub use structure::{
    is_table_value, is_tabular_object, table_from_columns, table_height,
    table_replace_variables_like, table_variable_names_from_object, table_variables, table_width,
};

#[cfg(test)]
pub(super) use analytics::cell_key_string;
pub(super) use analytics::{groupsummary_impl, grpstats_impl, pivot_impl};
pub(super) use indexing::*;
pub(super) use properties::*;
use rows::{assign_rows, concatenate_numeric_columns, selected_row_times};
pub(crate) use rows::{select_rows, selected_row_names, value_row_count};
pub(super) use selectors::one_based_to_zero;
pub(crate) use selectors::parse_variable_selector_for_object;
pub(crate) use structure::table_from_columns_like;
pub(super) use structure::{
    into_table_object, into_timetable_object, is_tabular_class, table_from_columns_with_class,
    table_from_columns_with_properties, table_object,
};
