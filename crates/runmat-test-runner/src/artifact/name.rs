use crate::{RunnerError, RunnerResult};

pub fn safe_artifact_name(name: &str) -> RunnerResult<String> {
    let normalized = name.trim().replace('\\', "/");
    if normalized.is_empty()
        || normalized.starts_with('/')
        || normalized.split('/').any(|segment| segment == "..")
        || normalized.contains('\0')
    {
        return Err(RunnerError::Artifact(format!(
            "unsafe artifact name '{name}'"
        )));
    }
    Ok(normalized)
}
