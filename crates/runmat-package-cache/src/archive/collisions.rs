use super::{ArchiveError, ValidatedArchiveEntry};
use std::collections::BTreeMap;
use unicode_casefold::UnicodeCaseFold;
use unicode_normalization::UnicodeNormalization;

pub(crate) fn reject_collisions(entries: &[ValidatedArchiveEntry]) -> Result<(), ArchiveError> {
    let mut portable_names = BTreeMap::new();
    for entry in entries {
        let normalized: String = entry.path.as_str().nfkc().collect();
        let key: String = normalized.case_fold().collect();
        if let Some(previous) = portable_names.insert(key, entry.path.clone()) {
            return Err(ArchiveError::Collision {
                first: previous.to_string(),
                second: entry.path.to_string(),
            });
        }
    }
    Ok(())
}
