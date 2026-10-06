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

    pub(crate) fn bind_runtime_sources(
        &self,
        runtime: runmat_runtime::context::RuntimeContext,
    ) -> Result<runmat_runtime::context::RuntimeContext, String> {
        self.validate_content_revisions()?;
        let records =
            self.entries.iter().map(
                |entry| runmat_runtime::source_context::ProgramSourceRecord {
                    ordinal: entry.source_id,
                    name: std::sync::Arc::from(entry.display_name.as_str()),
                    fullpath_name: entry.full_path.as_deref().map(std::sync::Arc::<str>::from),
                    text: std::sync::Arc::from(entry.text.as_str()),
                },
            );
        runmat_runtime::source_context::bind_portable_source_map_v2_with_records(
            runtime,
            &self.portable_v2()?,
            records,
        )
        .map_err(|error| error.to_string())
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
