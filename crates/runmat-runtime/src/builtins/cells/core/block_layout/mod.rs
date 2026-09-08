mod shape;
mod traversal;

pub(super) use shape::{
    checked_element_count, column_major_strides, extend_with_ones, prefix_offsets,
};
pub(super) use traversal::column_major_coordinates;
