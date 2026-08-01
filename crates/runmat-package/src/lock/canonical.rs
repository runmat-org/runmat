use super::model::PackageLock;
use super::validate::{canonicalized, validate_lock};
use crate::LockError;

pub fn encode_lock(lock: &PackageLock) -> Result<String, LockError> {
    validate_lock(lock)?;
    let canonical = canonicalized(lock.clone());
    let mut encoded = toml::to_string(&canonical)?;
    if !encoded.ends_with('\n') {
        encoded.push('\n');
    }
    Ok(encoded)
}

pub fn decode_lock(input: &str) -> Result<PackageLock, LockError> {
    let lock: PackageLock = toml::from_str(input)?;
    validate_lock(&lock)?;
    if lock != canonicalized(lock.clone()) {
        return Err(LockError::Invalid(
            "package and edge records are not in canonical order".to_string(),
        ));
    }
    Ok(lock)
}
