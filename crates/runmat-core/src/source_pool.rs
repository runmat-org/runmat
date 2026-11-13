use runmat_hir::SourceId;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(crate) struct SourceText {
    pub(crate) name: Arc<str>,
    pub(crate) fullpath_name: Option<Arc<str>>,
    pub(crate) text: Arc<str>,
    portable_record: Option<runmat_runtime::source_context::PortableSourceRecordAdmission>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct SourceKey {
    pub(crate) name: Arc<str>,
    pub(crate) fullpath_name: Option<Arc<str>>,
    pub(crate) text: Arc<str>,
    portable_identity: Option<runmat_runtime::source_context::PortableSourceIdentity>,
}

#[derive(Default)]
pub(crate) struct SourcePool {
    sources: Vec<SourceText>,
    index: HashMap<SourceKey, SourceId>,
    issued: HashMap<SourceId, runmat_runtime::source_context::RuntimeSourceLease>,
}

#[derive(Debug)]
pub(crate) enum SourcePoolBindingError {
    Missing(SourceId),
    ForeignOwner(SourceId),
    Exhausted(runmat_runtime::source_context::SourceIdExhausted),
    Portable(runmat_runtime::source_context::PortableSourceIdentityError),
}

impl std::fmt::Display for SourcePoolBindingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing(id) => write!(formatter, "source pool has no record {}", id.0),
            Self::ForeignOwner(id) => write!(
                formatter,
                "source pool record {} belongs to another runtime source owner",
                id.0
            ),
            Self::Exhausted(error) => error.fmt(formatter),
            Self::Portable(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SourcePoolBindingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Missing(_) | Self::ForeignOwner(_) => None,
            Self::Exhausted(error) => Some(error),
            Self::Portable(error) => Some(error),
        }
    }
}

impl SourcePool {
    pub(crate) fn intern_with_fullpath(
        &mut self,
        name: &str,
        fullpath_name: Option<&str>,
        text: &str,
    ) -> SourceId {
        let name: Arc<str> = Arc::from(name);
        let fullpath_name = fullpath_name.map(Arc::<str>::from);
        let text: Arc<str> = Arc::from(text);
        let key = SourceKey {
            name: Arc::clone(&name),
            fullpath_name: fullpath_name.as_ref().map(Arc::clone),
            text: Arc::clone(&text),
            portable_identity: None,
        };
        if let Some(id) = self.index.get(&key) {
            return *id;
        }
        let id = SourceId(self.sources.len());
        self.sources.push(SourceText {
            name,
            fullpath_name,
            text,
            portable_record: None,
        });
        self.index.insert(key, id);
        id
    }

    /// Compiler-issued source identity is keyed by the source map's original
    /// owner as well as diagnostic text. Equal text from distinct owners must
    /// never reuse one runtime lease.
    pub(crate) fn intern_portable_entry(
        &mut self,
        entry: &crate::SourceMapEntry,
        admitted: runmat_runtime::source_context::PortableSourceRecordAdmission,
    ) -> Result<SourceId, SourcePoolBindingError> {
        let name: Arc<str> = Arc::from(entry.display_name.as_str());
        let fullpath_name = entry.full_path.as_deref().map(Arc::<str>::from);
        let text: Arc<str> = Arc::from(entry.text.as_str());
        let record = admitted.record();
        if record.source_id != entry.source_id
            || record.owner_identity != entry.owner_identity
            || record.relative_path != entry.relative_path
            || record.content_revision != entry.content_revision
            || record.display_name != entry.display_name
            || record.text != entry.text
        {
            return Err(SourcePoolBindingError::Portable(
                runmat_runtime::source_context::PortableSourceIdentityError::RecordMismatch,
            ));
        }
        let key = SourceKey {
            name: Arc::clone(&name),
            fullpath_name: fullpath_name.as_ref().map(Arc::clone),
            text: Arc::clone(&text),
            portable_identity: Some(runmat_runtime::source_context::PortableSourceIdentity {
                owner_identity: entry.owner_identity.clone(),
                relative_path: entry.relative_path.clone(),
                content_revision: entry.content_revision,
            }),
        };
        if let Some(id) = self.index.get(&key) {
            return Ok(*id);
        }
        let id = SourceId(self.sources.len());
        self.sources.push(SourceText {
            name,
            fullpath_name,
            text,
            portable_record: Some(admitted),
        });
        self.index.insert(key, id);
        Ok(id)
    }

    pub(crate) fn issued_source_ref(
        &mut self,
        id: SourceId,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<runmat_runtime::source_context::RuntimeSourceRef, SourcePoolBindingError> {
        if let Some(lease) = self.issued.get(&id) {
            let reference = lease.source_ref();
            return reference
                .resolve_in(runtime)
                .map(|_| reference)
                .ok_or(SourcePoolBindingError::ForeignOwner(id));
        }
        let source = self
            .sources
            .get(id.0)
            .ok_or(SourcePoolBindingError::Missing(id))?;
        let lease = if let Some(record) = &source.portable_record {
            runmat_runtime::source_context::lease_portable_source_context_in_context(
                runtime,
                record,
                source.fullpath_name.as_ref().map(Arc::clone),
            )
            .map_err(SourcePoolBindingError::Portable)?
        } else {
            runmat_runtime::source_context::lease_source_context_in_context(
                runtime,
                Arc::clone(&source.name),
                source.fullpath_name.as_ref().map(Arc::clone),
                Arc::clone(&source.text),
            )
            .map_err(SourcePoolBindingError::Exhausted)?
        };
        let reference = lease.source_ref();
        self.issued.insert(id, lease);
        Ok(reference)
    }

    #[cfg(test)]
    pub(crate) fn entries(&self) -> impl Iterator<Item = (SourceId, &SourceText)> {
        self.sources
            .iter()
            .enumerate()
            .map(|(index, source)| (SourceId(index), source))
    }
}

#[cfg(test)]
#[path = "source_pool/tests.rs"]
mod tests;
