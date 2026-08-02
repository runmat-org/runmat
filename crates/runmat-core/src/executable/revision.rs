use sha2::{Digest, Sha256};

use super::ExecutableSource;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutableRevision {
    pub program_revision: Option<runmat_test::plan::ProgramRevision>,
    pub source_digest: String,
}

impl ExecutableRevision {
    pub fn derive(
        source: &ExecutableSource,
        program_revision: Option<runmat_test::plan::ProgramRevision>,
    ) -> Self {
        let mut digest = Sha256::new();
        frame(&mut digest, source.owner_identity.as_bytes());
        frame(&mut digest, source.relative_path.as_bytes());
        frame(&mut digest, source.text.as_bytes());
        Self {
            program_revision,
            source_digest: format!("sha256:{:x}", digest.finalize()),
        }
    }
}

fn frame(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_be_bytes());
    digest.update(bytes);
}
