//! Shared diagnostic paths for encoding and reconstructing nested values.

pub(super) fn index_path(parent: &str, index: usize) -> String {
    format!("{parent}[{index}]")
}

pub(super) fn field_path(parent: &str, field: &str) -> String {
    format!("{parent}.{field}")
}
