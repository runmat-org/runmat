use sha2::{Digest, Sha256};

use crate::version::TEST_IDENTITY_ALGORITHM_VERSION;
use crate::TestDomainError;

pub(crate) fn digest(kind: &str, parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    write_part(&mut hasher, "runmat-test");
    write_part(&mut hasher, &TEST_IDENTITY_ALGORITHM_VERSION.to_string());
    write_part(&mut hasher, kind);
    for part in parts {
        write_part(&mut hasher, part);
    }
    format!(
        "v{}:sha256:{}",
        TEST_IDENTITY_ALGORITHM_VERSION,
        hex::encode(hasher.finalize()),
    )
}

/// Check only the current canonical identity presentation. This does not
/// rederive an ID from discovery facts, and transparent serde remains unchanged.
pub(crate) fn validate_current(value: &str, field: &'static str) -> Result<(), TestDomainError> {
    let prefix = format!("v{}:", TEST_IDENTITY_ALGORITHM_VERSION);
    if value
        .strip_prefix(&prefix)
        .and_then(runmat_execution::Digest::parse_canonical)
        .is_none()
    {
        return Err(TestDomainError::InvalidField {
            field,
            reason: "expected current canonical test identity digest".into(),
        });
    }
    Ok(())
}

fn write_part(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}
