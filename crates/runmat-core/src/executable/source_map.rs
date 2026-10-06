#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExecutableSourceMap {
    entries: Vec<SourceMapEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceMapEntry {
    pub source_id: usize,
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

    pub(crate) fn validate_content_revisions(&self) -> Result<(), String> {
        for (ordinal, entry) in self.entries.iter().enumerate() {
            if entry.source_id != ordinal {
                return Err(format!(
                    "source map ordinal {} is missing or duplicated",
                    ordinal
                ));
            }
            if entry.content_revision != runmat_execution::Digest::sha256(entry.text.as_bytes()) {
                return Err(format!(
                    "source {} has a content revision that does not match its text",
                    entry.source_id
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
            source_id: 0,
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
