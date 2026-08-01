use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TestSelector {
    #[serde(default)]
    pub names: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub source_prefixes: Vec<String>,
    #[serde(default)]
    pub excluded_tags: Vec<String>,
}

impl TestSelector {
    pub fn matches(&self, display_name: &str, tags: &[String], source_path: &str) -> bool {
        (self.names.is_empty()
            || self
                .names
                .iter()
                .any(|pattern| display_name.contains(pattern)))
            && self.tags.iter().all(|tag| tags.contains(tag))
            && self.excluded_tags.iter().all(|tag| !tags.contains(tag))
            && (self.source_prefixes.is_empty()
                || self
                    .source_prefixes
                    .iter()
                    .any(|prefix| source_path.starts_with(prefix)))
    }
}
