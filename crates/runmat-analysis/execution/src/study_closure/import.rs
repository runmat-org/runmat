//! Caller-bounded reads preserve the original producer policy and receipt identities.
use super::{check_inventory, invalid};
use crate::AnalysisBridgeResult;
use runmat_execution::{value::ValueRef, Digest};
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ArtifactAccess, ArtifactError,
    ArtifactResult,
};

// A borrowed trait object keeps nested imports at one source type without boxing the provider.
pub(crate) struct CallerBoundedImport<'a> {
    source: &'a dyn CacheImport,
    limits: ObjectInventoryLimits,
}

impl<'a> CallerBoundedImport<'a> {
    pub(crate) fn new(
        source: &'a dyn CacheImport,
        limits: ObjectInventoryLimits,
    ) -> AnalysisBridgeResult<Self> {
        if limits.max_objects == 0 || limits.max_object_bytes == 0 || limits.max_total_bytes == 0 {
            return Err(invalid(
                "study import needs a nonzero caller inventory budget",
            ));
        }
        Ok(Self { source, limits })
    }

    pub(crate) fn close<'r>(
        self,
        references: &'r [ValueRef],
        access: &'r ArtifactAccess,
    ) -> AnalysisBridgeResult<ClosedStudyImport<'a, 'r>> {
        check_inventory(references, access, self.limits)?;
        if references.windows(2).any(|pair| pair[0].id >= pair[1].id) {
            return Err(invalid(
                "study closure references need unique canonical identities",
            ));
        }
        Ok(ClosedStudyImport {
            source: self,
            references,
            access,
        })
    }

    fn maximum(&self, requested: u64) -> u64 {
        requested
            .min(self.limits.max_object_bytes)
            .min(self.limits.max_total_bytes)
    }
}

impl CacheImport for CallerBoundedImport<'_> {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.read_verified_bounded(digest, self.limits.max_object_bytes)
    }

    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        let maximum = self.maximum(maximum);
        verify(
            self.source.read_verified_bounded(digest, maximum)?,
            digest,
            maximum,
            None,
        )
    }

    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum: u64,
        admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        let maximum = self.maximum(maximum);
        verify(
            self.source
                .read_verified_bounded_with_admission(digest, maximum, admit)?,
            digest,
            maximum,
            None,
        )
    }
}

pub(crate) struct ClosedStudyImport<'s, 'r> {
    source: CallerBoundedImport<'s>,
    references: &'r [ValueRef],
    access: &'r ArtifactAccess,
}

impl ClosedStudyImport<'_, '_> {
    pub(crate) fn references(&self) -> &[ValueRef] {
        self.references
    }

    fn length(&self, digest: Digest) -> ArtifactResult<u64> {
        let id = self.access.value_id(digest);
        let index = self
            .references
            .binary_search_by_key(&id, |reference| reference.id)
            .map_err(|_| {
                ArtifactError::Identity("study replay read outside its admitted closure".into())
            })?;
        Ok(self.references[index].encoded_length)
    }
}

impl CacheImport for ClosedStudyImport<'_, '_> {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.read_verified_bounded(digest, u64::MAX)
    }

    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        let length = self.length(digest)?;
        let maximum = self.source.maximum(maximum.min(length));
        verify(
            self.source.source.read_verified_bounded(digest, maximum)?,
            digest,
            maximum,
            Some(length),
        )
    }

    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum: u64,
        admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        let length = self.length(digest)?;
        let maximum = self.source.maximum(maximum.min(length));
        verify(
            self.source
                .source
                .read_verified_bounded_with_admission(digest, maximum, admit)?,
            digest,
            maximum,
            Some(length),
        )
    }
}

fn verify(
    payload: Option<Vec<u8>>,
    digest: Digest,
    maximum: u64,
    length: Option<u64>,
) -> ArtifactResult<Option<Vec<u8>>> {
    if let Some(bytes) = payload.as_ref() {
        if bytes.len() as u64 > maximum {
            return Err(ArtifactError::Limit(
                "study cache exceeded its admitted read bound".into(),
            ));
        }
        if length.is_some_and(|length| bytes.len() as u64 != length)
            || Digest::sha256(bytes) != digest
        {
            return Err(ArtifactError::Identity(
                "study cache bytes differ from their admitted identity".into(),
            ));
        }
    }
    Ok(payload)
}

#[cfg(test)]
mod tests;
