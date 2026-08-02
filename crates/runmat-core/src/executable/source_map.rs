#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExecutableSourceMap {
    entries: Vec<SourceMapEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceMapEntry {
    pub source_id: usize,
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
}
