//! Stored record formats and the refusal for a record written in another one.
//!
//! A stored run record authenticates its content by re-serializing the typed run metadata and
//! comparing the digest it recorded. A record written before a persisted type gained, lost or
//! renamed a field therefore cannot be authenticated by a later runtime. Each such change bumps
//! `ARTIFACT_SCHEMA_VERSION` (and the frequency record versions when the study step record
//! changes), so the reader refuses the old record by its format instead of reporting a digest
//! mismatch. Records are never migrated: running the study again writes the current format.
//!
//! The `shape` test module pins the persisted shape each version names.

/// The refusal for a stored result whose record format this runtime does not read.
pub(in crate::analysis) fn stored_by_another_version(schema_version: &str) -> String {
    format!(
        "this result was saved by an earlier version of RunMat (record format \
         `{schema_version}`) and cannot be opened by this version; run the study again to \
         regenerate it"
    )
}

#[cfg(test)]
pub(in crate::analysis) mod shape;
