//! Path-free portable identity retained by the exact issued source lease.

use runmat_execution::{Digest, ExecutableSourceRecordV2};

use super::{
    lease_source_context_in_context, source_info_in_context, RuntimeSourceLease, RuntimeSourceRef,
    SourceIdExhausted,
};
use runmat_types::ProgramSourceId;
use std::sync::Arc;

/// A complete source map admitted before any individual source is issued.
/// Individual records cannot be constructed as portable authority from their
/// public wire fields alone.
#[derive(Debug)]
pub struct ValidatedPortableSourceMap(Arc<runmat_execution::ExecutableSourceMapV2>);

impl ValidatedPortableSourceMap {
    pub fn new(
        map: Arc<runmat_execution::ExecutableSourceMapV2>,
    ) -> Result<Self, runmat_execution::ContractError> {
        map.validate()?;
        Ok(Self(map))
    }

    pub fn record(&self, ordinal: ProgramSourceId) -> Option<PortableSourceRecordAdmission> {
        self.0
            .entries
            .get(usize::try_from(ordinal.0).ok()?)
            .filter(|record| record.source_id == ordinal)
            .cloned()
            .map(PortableSourceRecordAdmission)
    }

    pub(super) fn map_arc(&self) -> Arc<runmat_execution::ExecutableSourceMapV2> {
        Arc::clone(&self.0)
    }
}

#[derive(Clone, Debug)]
pub struct PortableSourceRecordAdmission(ExecutableSourceRecordV2);

impl PortableSourceRecordAdmission {
    pub fn record(&self) -> &ExecutableSourceRecordV2 {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PortableSourceIdentity {
    pub owner_identity: String,
    pub relative_path: String,
    pub content_revision: Digest,
}

impl PortableSourceIdentity {
    pub(super) fn from_record(record: &ExecutableSourceRecordV2) -> Self {
        Self {
            owner_identity: record.owner_identity.clone(),
            relative_path: record.relative_path.clone(),
            content_revision: record.content_revision,
        }
    }
}

/// Resolve the immutable portable owner attached when this exact source was
/// issued. A source ID, matching display name, or matching text alone conveys
/// no portable owner authority.
pub fn portable_source_identity_in_context(
    runtime: &crate::context::RuntimeContext,
    source: RuntimeSourceRef,
) -> Option<PortableSourceIdentity> {
    let id = source.resolve_in(runtime)?;
    source_info_in_context(runtime, id)?.portable_identity
}

impl RuntimeSourceLease {
    /// Attach only the portable source-map record paired with this issued
    /// diagnostic source. Owner/path provenance never comes from a later
    /// compilation's primary source or a numeric source ID.
    fn bind_portable_identity(
        &self,
        record: &ExecutableSourceRecordV2,
    ) -> Result<(), PortableSourceIdentityError> {
        if self.0.info.name.as_ref() != record.display_name
            || self.0.info.text.as_ref() != record.text
            || record.content_revision != Digest::sha256(record.text.as_bytes())
        {
            return Err(PortableSourceIdentityError::RecordMismatch);
        }
        let identity = PortableSourceIdentity::from_record(record);
        if let Some(existing) = self.0.portable_identity.get() {
            return (existing == &identity)
                .then_some(())
                .ok_or(PortableSourceIdentityError::ConflictingOwner);
        }
        self.0
            .portable_identity
            .set(identity)
            .map_err(|_| PortableSourceIdentityError::ConflictingOwner)
    }
}

/// Issue one fresh source lease from an admitted map record. Dynamic source
/// evaluation may instead attach the same checked record to its exact live,
/// already-issued lease through `bind_portable_source_map_v2_with_existing_lease`;
/// a display name, matching text, or numeric source ID alone is insufficient.
pub fn lease_portable_source_context_in_context(
    runtime: &crate::context::RuntimeContext,
    admitted: &PortableSourceRecordAdmission,
    fullpath_name: Option<Arc<str>>,
) -> Result<RuntimeSourceLease, PortableSourceIdentityError> {
    let record = admitted.record();
    let lease = lease_source_context_in_context(
        runtime,
        Arc::<str>::from(record.display_name.as_str()),
        fullpath_name,
        Arc::<str>::from(record.text.as_str()),
    )
    .map_err(PortableSourceIdentityError::Exhausted)?;
    lease.bind_portable_identity(record)?;
    Ok(lease)
}

pub(super) fn attach_admitted_identity(
    lease: &RuntimeSourceLease,
    admitted: &PortableSourceRecordAdmission,
) -> Result<(), PortableSourceIdentityError> {
    lease.bind_portable_identity(admitted.record())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PortableSourceIdentityError {
    RecordMismatch,
    ConflictingOwner,
    Exhausted(SourceIdExhausted),
}

impl std::fmt::Display for PortableSourceIdentityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RecordMismatch => {
                formatter.write_str("portable source record differs from issued source text")
            }
            Self::ConflictingOwner => {
                formatter.write_str("issued source already has another portable owner")
            }
            Self::Exhausted(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for PortableSourceIdentityError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Exhausted(error) => Some(error),
            Self::RecordMismatch | Self::ConflictingOwner => None,
        }
    }
}

#[cfg(test)]
#[path = "portable_provenance/tests.rs"]
mod tests;
