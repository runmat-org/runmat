use sha2::{Digest as _, Sha256};

use super::ExecutableSource;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutableRevision {
    pub program_revision: runmat_execution::ProgramRevision,
    pub source_digest: String,
}

impl ExecutableRevision {
    pub fn derive(
        source: &ExecutableSource,
        program_revision: Option<runmat_execution::ProgramRevision>,
        environment: runmat_execution::ProgramEnvironment,
    ) -> Self {
        let mut digest = Sha256::new();
        frame(&mut digest, source.owner_identity.as_bytes());
        frame(&mut digest, source.relative_path.as_bytes());
        frame(&mut digest, source.text.as_bytes());
        let source_digest = runmat_execution::Digest::from_bytes(digest.finalize().into());
        let program_revision = program_revision.unwrap_or_else(|| {
            let mut graph = Sha256::new();
            frame(&mut graph, b"runmat-loose-source-graph-v1");
            frame(&mut graph, source.owner_identity.as_bytes());
            runmat_execution::ProgramRevision::new(
                runmat_execution::Digest::from_bytes(graph.finalize().into()),
                source_digest,
                environment,
            )
            .expect("Core source identity and program environment are valid")
        });
        Self {
            program_revision,
            source_digest: source_digest.to_string(),
        }
    }
}

fn frame(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_be_bytes());
    digest.update(bytes);
}
