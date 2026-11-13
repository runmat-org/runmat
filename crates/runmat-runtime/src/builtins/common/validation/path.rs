//! Path validation through the configured filesystem service, never raw host I/O.
use super::{value_texts, BuiltinResult, Value};

pub async fn value_is_file_async(value: &Value) -> BuiltinResult<bool> {
    paths_match(value, runmat_filesystem::FsMetadata::is_file).await
}

pub async fn value_is_folder_async(value: &Value) -> BuiltinResult<bool> {
    paths_match(value, runmat_filesystem::FsMetadata::is_dir).await
}

async fn paths_match(
    value: &Value,
    predicate: fn(&runmat_filesystem::FsMetadata) -> bool,
) -> BuiltinResult<bool> {
    // Admit all text before performing any filesystem access.
    for path in value_texts(value)? {
        if !runmat_filesystem::metadata_async(path.as_ref())
            .await
            .is_ok_and(|metadata| predicate(&metadata))
        {
            return Ok(false);
        }
    }
    Ok(true)
}
