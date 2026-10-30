//! Recovery owns its temporary/noclobber workflow; native filesystem owns
//! the one Windows owner/SYSTEM creation and opened-handle ACL mechanism.
use std::path::Path;

use anyhow::{Context as _, Result};

pub(super) fn create_temporary(parent: &Path) -> Result<tempfile::NamedTempFile> {
    let parent = parent
        .canonicalize()
        .context("resolve private recovery output directory")?;
    let path = parent.join(format!(".runmat-recovery-{}.tmp", uuid::Uuid::new_v4()));
    let file = runmat_filesystem::NativeFsProvider
        .create_owner_private_file_new(&path)
        .context("create owner-private recovery file")?;
    Ok(tempfile::NamedTempFile::from_parts(
        file,
        tempfile::TempPath::try_from_path(path)?,
    ))
}
