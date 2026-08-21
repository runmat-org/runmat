use super::{BuiltinCatalogEntry, BUILTIN_CATALOG_SCHEMA_VERSION};
use sha2::{Digest, Sha256};

pub fn canonical_catalog_fingerprint(
    entries: &[&'static BuiltinCatalogEntry],
) -> Result<[u8; 32], serde_json::Error> {
    let mut ordered = entries.to_vec();
    ordered.sort_unstable_by_key(|entry| entry.identity);
    let mut hash = Sha256::new();
    hash.update(b"runmat-canonical-builtin-catalog");
    hash.update(BUILTIN_CATALOG_SCHEMA_VERSION.to_le_bytes());
    hash.update(serde_json::to_vec(&ordered)?);
    Ok(hash.finalize().into())
}
