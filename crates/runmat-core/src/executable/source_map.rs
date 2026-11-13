#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExecutableSourceMap {
    entries: Vec<SourceMapEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceMapEntry {
    pub source_id: runmat_types::ProgramSourceId,
    pub owner_identity: String,
    pub relative_path: String,
    pub content_revision: runmat_execution::Digest,
    pub display_name: String,
    pub full_path: Option<String>,
    pub text: String,
}

impl ExecutableSourceMap {
    pub(crate) fn new(mut entries: Vec<SourceMapEntry>) -> Self {
        entries.sort_by_key(|entry| entry.source_id);
        Self { entries }
    }

    pub fn entries(&self) -> &[SourceMapEntry] {
        &self.entries
    }

    /// Path-free private-owner key for one authenticated source ordinal.
    pub(crate) fn private_source_alias(
        &self,
        source: runmat_types::ProgramSourceId,
    ) -> Option<runmat_vm::PrivateSourceAlias> {
        let entry = self.entries.get(usize::try_from(source.0).ok()?)?;
        if entry.source_id != source {
            return None;
        }
        Some(runmat_vm::PrivateSourceAlias::from_source_path(
            &entry.owner_identity,
            &entry.relative_path,
        ))
    }

    pub(crate) fn validate_private_source_owners(
        &self,
        registry: &runmat_vm::FunctionRegistry,
    ) -> Result<(), String> {
        let known = self
            .entries
            .iter()
            .filter_map(|entry| self.private_source_alias(entry.source_id))
            .collect::<std::collections::HashSet<_>>();
        for body in registry.functions.values() {
            let Some(owner) = &body.private_source_owner else {
                continue;
            };
            let own = body
                .source_id
                .and_then(runmat_vm::BytecodeSourceRef::program)
                .and_then(|id| self.private_source_alias(id));
            let aliases = owner.aliases();
            if own.as_ref().is_none_or(|own| !aliases.contains(own))
                || aliases.iter().any(|alias| !known.contains(alias))
                || aliases.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err("function private owner aliases are outside its source map".into());
            }
        }
        let mut partition_by_alias = std::collections::HashMap::new();
        let mut partition_by_source = std::collections::HashMap::new();
        for body in registry.functions.values() {
            let partition = body
                .private_source_owner
                .as_ref()
                .map(|owner| owner.aliases());
            if let Some(source) = body
                .source_id
                .and_then(runmat_vm::BytecodeSourceRef::program)
            {
                if let Some(previous) = partition_by_source.insert(source, partition) {
                    if previous != partition {
                        return Err("one source has conflicting private owner partitions".into());
                    }
                }
            }
            if let Some(aliases) = partition {
                for alias in aliases {
                    if let Some(previous) = partition_by_alias.insert(alias, aliases) {
                        if previous != aliases {
                            return Err("private owner partitions overlap asymmetrically".into());
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn bind_runtime_sources(
        &self,
        runtime: runmat_runtime::context::RuntimeContext,
    ) -> Result<runmat_runtime::context::RuntimeContext, String> {
        self.validate_content_revisions()?;
        let records = self.runtime_records();
        runmat_runtime::source_context::bind_portable_source_map_v2_with_records(
            runtime,
            &self.portable_v2()?,
            records,
        )
        .map_err(|error| error.to_string())
    }

    pub(crate) fn runtime_records(
        &self,
    ) -> Vec<runmat_runtime::source_context::ProgramSourceRecord> {
        self.entries
            .iter()
            .map(
                |entry| runmat_runtime::source_context::ProgramSourceRecord {
                    ordinal: entry.source_id,
                    name: std::sync::Arc::from(entry.display_name.as_str()),
                    fullpath_name: entry.full_path.as_deref().map(std::sync::Arc::<str>::from),
                    text: std::sync::Arc::from(entry.text.as_str()),
                },
            )
            .collect()
    }

    pub(crate) fn portable_v2(&self) -> Result<runmat_execution::ExecutableSourceMapV2, String> {
        runmat_execution::ExecutableSourceMapV2::new(
            self.entries
                .iter()
                .map(|entry| runmat_execution::ExecutableSourceRecordV2 {
                    source_id: entry.source_id,
                    owner_identity: entry.owner_identity.clone(),
                    relative_path: entry.relative_path.clone(),
                    content_revision: entry.content_revision,
                    display_name: entry.display_name.clone(),
                    text: entry.text.clone(),
                })
                .collect(),
        )
        .map_err(|error| error.to_string())
    }

    pub(crate) fn validate_content_revisions(&self) -> Result<(), String> {
        for (ordinal, entry) in self.entries.iter().enumerate() {
            if u32::try_from(ordinal)
                .ok()
                .map(runmat_types::ProgramSourceId)
                != Some(entry.source_id)
            {
                return Err(format!(
                    "source map ordinal {} is missing or duplicated",
                    ordinal
                ));
            }
            if entry.content_revision != runmat_execution::Digest::sha256(entry.text.as_bytes()) {
                return Err(format!(
                    "source {} has a content revision that does not match its text",
                    entry.source_id.0
                ));
            }
        }
        if self.entries.is_empty() {
            return Err("source map has no primary source".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_revision_is_checked_against_exact_text() {
        let entry = SourceMapEntry {
            source_id: runmat_types::ProgramSourceId(0),
            owner_identity: "package".into(),
            relative_path: "main.m".into(),
            content_revision: runmat_execution::Digest::sha256(b"x = 1;"),
            display_name: "main.m".into(),
            full_path: None,
            text: "x = 1;".into(),
        };
        assert!(ExecutableSourceMap::new(vec![entry.clone()])
            .validate_content_revisions()
            .is_ok());
        let mut forged = entry;
        forged.text = "x = 2;".into();
        assert!(ExecutableSourceMap::new(vec![forged])
            .validate_content_revisions()
            .is_err());
    }
}
