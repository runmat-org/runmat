use runmat_test::descriptor::SourceDescriptor;
use runmat_test::discovery::FrozenTestRunSnapshot;

use crate::ExecutableSource;

#[derive(Clone, Debug)]
pub(super) struct TestSourceCatalog {
    revision: runmat_execution::ProgramRevision,
    snapshot: FrozenTestRunSnapshot,
}

impl TestSourceCatalog {
    pub fn from_snapshot(snapshot: &FrozenTestRunSnapshot) -> Self {
        Self {
            revision: snapshot.program_revision.clone(),
            snapshot: snapshot.clone(),
        }
    }

    pub fn executable_source(
        &self,
        descriptor: &SourceDescriptor,
    ) -> Result<ExecutableSource, String> {
        let source = self
            .snapshot
            .sources
            .iter()
            .find(|source| {
                source.owner_identity == descriptor.owner_identity
                    && source.relative_path == descriptor.relative_path
            })
            .ok_or_else(|| {
                format!(
                    "frozen test source '{}:{}' is unavailable",
                    descriptor.owner_identity, descriptor.relative_path
                )
            })?;
        Ok(ExecutableSource::new(
            descriptor.owner_identity.clone(),
            descriptor.relative_path.clone(),
            source.content.clone(),
        ))
    }

    pub fn revision(&self) -> runmat_execution::ProgramRevision {
        self.revision.clone()
    }

    pub fn snapshot(&self) -> &FrozenTestRunSnapshot {
        &self.snapshot
    }
}
