use runmat_package_cache::{CacheError, CacheState};
use rusqlite::Connection;

pub(crate) fn validate_payload_closure(
    connection: &Connection,
    state: &CacheState,
) -> Result<(), CacheError> {
    let mut statement = connection
        .prepare("SELECT 1 FROM object_payloads WHERE digest = ?1")
        .map_err(failure)?;
    for (digest, object) in &state.objects {
        if object.stored_payload_bytes() == 0 {
            continue;
        }
        let present = statement.exists([digest.to_string()]).map_err(failure)?;
        if !present {
            return Err(CacheError::Miss(digest.clone()));
        }
    }
    Ok(())
}

fn failure(error: rusqlite::Error) -> CacheError {
    runmat_package_cache::BackendError::Failure(error.to_string()).into()
}
