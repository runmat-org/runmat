use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::TestDomainError;
use runmat_execution::{
    Digest as ExecutionDigest, DomainContribution, ProgramEnvironment, ProgramRevision,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SavedRunSource {
    pub owner_identity: String,
    pub relative_path: String,
    pub content: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct UnsavedRunBuffer {
    pub owner_identity: String,
    pub relative_path: String,
    pub content: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunSourceOrigin {
    Saved,
    UnsavedBuffer,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FrozenRunSource {
    pub owner_identity: String,
    pub relative_path: String,
    pub content: String,
    pub origin: RunSourceOrigin,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FrozenTestRunSnapshot {
    /// Source-catalog revision before intentionally selected editor overlays.
    pub base_source_digest: String,
    pub program_revision: ProgramRevision,
    pub sources: Vec<FrozenRunSource>,
}

impl FrozenTestRunSnapshot {
    /// Freeze one immutable input for discovery and execution. Unsaved buffers
    /// intentionally included by the caller replace the corresponding saved
    /// source. Editor state that is not supplied here cannot leak into the run.
    pub fn freeze(
        graph_digest: impl Into<String>,
        base_source_digest: impl Into<String>,
        environment: ProgramEnvironment,
        test_config_digest: impl Into<String>,
        saved_sources: Vec<SavedRunSource>,
        unsaved_buffers: Vec<UnsavedRunBuffer>,
    ) -> Result<Self, TestDomainError> {
        let mut sources = BTreeMap::<(String, String), FrozenRunSource>::new();
        for source in saved_sources {
            let relative_path = normalize_relative_path(&source.relative_path)?;
            let key = (source.owner_identity.clone(), relative_path.clone());
            if sources
                .insert(
                    key,
                    FrozenRunSource {
                        owner_identity: source.owner_identity,
                        relative_path,
                        content: source.content,
                        origin: RunSourceOrigin::Saved,
                    },
                )
                .is_some()
            {
                return Err(TestDomainError::InvalidField {
                    field: "saved_sources",
                    reason: "duplicate owner and relative source path".into(),
                });
            }
        }
        for buffer in unsaved_buffers {
            let relative_path = normalize_relative_path(&buffer.relative_path)?;
            sources.insert(
                (buffer.owner_identity.clone(), relative_path.clone()),
                FrozenRunSource {
                    owner_identity: buffer.owner_identity,
                    relative_path,
                    content: buffer.content,
                    origin: RunSourceOrigin::UnsavedBuffer,
                },
            );
        }
        let sources = sources.into_values().collect::<Vec<_>>();
        let base_source_digest = base_source_digest.into();
        let source_digest = source_digest(&base_source_digest, &sources);
        let graph_digest = parse_digest("graph_digest", graph_digest.into())?;
        let test_config_digest = parse_digest("test_config_digest", test_config_digest.into())?;
        let program_revision = ProgramRevision::new(graph_digest, source_digest, environment)
            .and_then(|revision| {
                revision.with_domain_contribution(DomainContribution::new(
                    "runmat.test.config",
                    test_config_digest,
                )?)
            })
            .map_err(|error| TestDomainError::InvalidField {
                field: "program_revision",
                reason: error.to_string(),
            })?;
        Ok(Self {
            base_source_digest,
            program_revision,
            sources,
        })
    }

    pub fn validate(&self) -> Result<(), TestDomainError> {
        let expected = source_digest(&self.base_source_digest, &self.sources);
        if expected != *self.program_revision.source_digest() {
            return Err(TestDomainError::InvalidField {
                field: "program_revision.source_digest",
                reason: "frozen run sources no longer match the program revision".into(),
            });
        }
        let mut previous: Option<(&str, &str)> = None;
        for source in &self.sources {
            let normalized = normalize_relative_path(&source.relative_path)?;
            if normalized != source.relative_path {
                return Err(TestDomainError::InvalidField {
                    field: "sources.relative_path",
                    reason: "source path is not normalized".into(),
                });
            }
            let current = (
                source.owner_identity.as_str(),
                source.relative_path.as_str(),
            );
            if previous.is_some_and(|value| value >= current) {
                return Err(TestDomainError::InvalidField {
                    field: "sources",
                    reason: "sources must be unique and canonically ordered".into(),
                });
            }
            previous = Some(current);
        }
        Ok(())
    }
}

fn normalize_relative_path(path: &str) -> Result<String, TestDomainError> {
    let normalized = path.replace('\\', "/");
    let invalid = normalized.is_empty()
        || normalized.starts_with('/')
        || normalized.split('/').any(|segment| {
            segment.is_empty() || segment == "." || segment == ".." || segment.contains('\0')
        })
        || normalized
            .as_bytes()
            .get(1)
            .is_some_and(|second| *second == b':');
    if invalid {
        return Err(TestDomainError::InvalidField {
            field: "sources.relative_path",
            reason: "source paths must be normalized, non-empty, and relative".into(),
        });
    }
    Ok(normalized)
}

fn source_digest(base_source_digest: &str, sources: &[FrozenRunSource]) -> ExecutionDigest {
    let mut hasher = Sha256::new();
    write_part(&mut hasher, "runmat-test-run-sources-v1");
    write_part(&mut hasher, base_source_digest);
    for source in sources {
        write_part(&mut hasher, &source.owner_identity);
        write_part(&mut hasher, &source.relative_path);
        write_part(&mut hasher, &source.content);
    }
    ExecutionDigest::from_bytes(hasher.finalize().into())
}

fn parse_digest(field: &'static str, value: String) -> Result<ExecutionDigest, TestDomainError> {
    value.parse().map_err(
        |error: runmat_execution::ContractError| TestDomainError::InvalidField {
            field,
            reason: error.to_string(),
        },
    )
}

fn write_part(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}
