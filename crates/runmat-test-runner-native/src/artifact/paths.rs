use std::path::{Path, PathBuf};

use runmat_test::identity::RunId;
use sha2::{Digest, Sha256};

pub(super) fn run_directory(root: &Path, run_id: &RunId) -> PathBuf {
    root.join(format!("{:x}", Sha256::digest(run_id.as_str().as_bytes())))
}
