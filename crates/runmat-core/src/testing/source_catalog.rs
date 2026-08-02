use std::collections::BTreeMap;

use runmat_test::descriptor::SourceDescriptor;
use runmat_test::discovery::FrozenTestRunSnapshot;

use crate::ExecutableSource;

#[derive(Clone, Debug)]
pub(super) struct TestSourceCatalog {
    revision: runmat_test::plan::ProgramRevision,
    sources: BTreeMap<(String, String), String>,
}

impl TestSourceCatalog {
    pub fn from_snapshot(snapshot: &FrozenTestRunSnapshot) -> Self {
        Self {
            revision: snapshot.program_revision.clone(),
            sources: snapshot
                .sources
                .iter()
                .map(|source| {
                    (
                        (source.owner_identity.clone(), source.relative_path.clone()),
                        source.content.clone(),
                    )
                })
                .collect(),
        }
    }

    pub fn executable_source(
        &self,
        descriptor: &SourceDescriptor,
        span_only: bool,
    ) -> Result<ExecutableSource, String> {
        let content = self
            .sources
            .get(&(
                descriptor.owner_identity.clone(),
                descriptor.relative_path.clone(),
            ))
            .ok_or_else(|| {
                format!(
                    "frozen test source '{}:{}' is unavailable",
                    descriptor.owner_identity, descriptor.relative_path
                )
            })?;
        let text = if span_only {
            let start = descriptor.span.start_byte as usize;
            let end = descriptor.span.end_byte as usize;
            content
                .get(start..end)
                .ok_or_else(|| {
                    format!("procedure span {start}..{end} is not a valid UTF-8 source range")
                })?
                .to_owned()
        } else {
            content.clone()
        };
        Ok(ExecutableSource::new(
            descriptor.owner_identity.clone(),
            descriptor.relative_path.clone(),
            text,
        ))
    }

    pub fn revision(&self) -> runmat_test::plan::ProgramRevision {
        self.revision.clone()
    }
}
