use super::*;
pub(super) use crate::context::document_test_support::{
    controlled_context, poll_pending_until_host_starts, poll_ready,
};

pub(super) fn args(path: &std::path::Path, line: f64) -> Vec<Value> {
    vec![
        Value::String(path.to_string_lossy().into_owned().into()),
        Value::Num(line),
    ]
}
