use crate::{ProcessHostError, ProcessHostResult};

pub fn validate_pipe_name(name: &str) -> ProcessHostResult<String> {
    if !name.starts_with(r"\\.\pipe\") {
        return Err(ProcessHostError::Configuration(
            "Windows IPC pipe must use the local named-pipe namespace".into(),
        ));
    }
    Ok(name.to_owned())
}
