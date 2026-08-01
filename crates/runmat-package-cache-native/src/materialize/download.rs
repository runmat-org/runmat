use crate::NativeCacheError;
use runmat_package::ContentDigest;
use runmat_package_cache::CacheBackend;

pub(super) async fn read_verified<B: CacheBackend>(
    backend: &B,
    digest: &ContentDigest,
    expected_bytes: u64,
) -> Result<Vec<u8>, NativeCacheError> {
    let bytes = backend
        .read_object_bytes(digest)
        .await
        .map_err(runmat_package_cache::CacheError::from)?
        .ok_or_else(|| runmat_package_cache::CacheError::Miss(digest.clone()))?;
    if bytes.len() as u64 != expected_bytes || ContentDigest::sha256(&bytes) != *digest {
        return Err(NativeCacheError::CorruptTree {
            path: std::path::PathBuf::new(),
            reason: format!("blob {digest} failed size or digest verification"),
        });
    }
    Ok(bytes)
}
