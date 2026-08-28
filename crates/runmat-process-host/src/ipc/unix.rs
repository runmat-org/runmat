use std::path::{Path, PathBuf};

use crate::{ProcessHostError, ProcessHostResult};

pub fn validate_socket_path(path: &Path) -> ProcessHostResult<PathBuf> {
    if !path.is_absolute() {
        return Err(ProcessHostError::Configuration(
            "Unix IPC socket path must be absolute".into(),
        ));
    }
    Ok(path.to_path_buf())
}
