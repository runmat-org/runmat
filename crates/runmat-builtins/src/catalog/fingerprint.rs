use super::{BuiltinCatalogAlias, BuiltinCatalogEntry, BUILTIN_CATALOG_SCHEMA_VERSION};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Serialize)]
struct SemanticAliasEdge<'a> {
    alias: &'a super::BuiltinCatalogIdentity,
    canonical: &'a super::BuiltinCatalogIdentity,
}

pub fn canonical_catalog_fingerprint(
    entries: &[&BuiltinCatalogEntry],
    aliases: &[&BuiltinCatalogAlias],
) -> Result<[u8; 32], serde_json::Error> {
    let mut ordered = entries.to_vec();
    ordered.sort_unstable_by_key(|entry| entry.identity);
    let mut ordered_aliases = aliases.to_vec();
    ordered_aliases.sort_unstable_by_key(|entry| entry.alias);
    let mut hash = Sha256::new();
    hash.update(b"runmat-canonical-builtin-catalog");
    hash.update(BUILTIN_CATALOG_SCHEMA_VERSION.to_le_bytes());
    hash.update(serde_json::to_vec(&ordered)?);
    let semantic_aliases = ordered_aliases
        .iter()
        .map(|entry| SemanticAliasEdge {
            alias: &entry.alias,
            canonical: &entry.canonical,
        })
        .collect::<Vec<_>>();
    hash.update(serde_json::to_vec(&semantic_aliases)?);
    Ok(hash.finalize().into())
}
