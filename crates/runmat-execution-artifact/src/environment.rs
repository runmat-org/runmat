//! Canonical, bounded, explicitly public environment carried by an execution bundle.
use crate::{ArtifactError, ArtifactResult};
pub const MAX_PORTABLE_ENVIRONMENT_ENTRIES: usize = 256;
pub const MAX_PORTABLE_ENVIRONMENT_BYTES: usize = 1024 * 1024;

pub(crate) fn validate(entries: &[(String, String)]) -> ArtifactResult<()> {
    let bytes = entries.iter().try_fold(0_usize, |total, (name, value)| {
        total.checked_add(name.len())?.checked_add(value.len())
    });
    if entries.len() > MAX_PORTABLE_ENVIRONMENT_ENTRIES
        || bytes.is_none_or(|bytes| bytes > MAX_PORTABLE_ENVIRONMENT_BYTES)
        || entries.windows(2).any(|pair| pair[0].0 >= pair[1].0)
    {
        return Err(ArtifactError::Limit(
            "portable environment is unordered, duplicated or exceeds its bound".into(),
        ));
    }
    for (name, value) in entries {
        let upper = name.to_ascii_uppercase();
        if name.is_empty()
            || name.len() > 128
            || value.len() > 4096
            || !name.is_ascii()
            || !value.is_ascii()
            || name.contains('=')
            || name.chars().any(char::is_control)
            || value.chars().any(char::is_control)
            || upper.starts_with("OTEL_")
            || upper.starts_with("RUNMAT_OTEL_")
            || upper.starts_with("RUNMAT_RUNTIME__LOGGING__OTEL__")
            || ["TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH"]
                .iter()
                .any(|marker| upper.contains(marker))
        {
            return Err(ArtifactError::Invalid(
                "portable environment contains an invalid or secret-bearing entry".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
