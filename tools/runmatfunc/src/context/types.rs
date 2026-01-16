use std::path::PathBuf;

use serde::Serialize;

use crate::builtin::metadata::BuiltinRecord;

#[derive(Debug, Serialize)]
pub struct AuthoringContext {
    pub builtin: BuiltinRecord,
    pub prompt: String,
    pub source_paths: Vec<PathBuf>,
}
