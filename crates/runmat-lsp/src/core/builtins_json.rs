use once_cell::sync::Lazy;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::sync::Mutex;

mod index;

static DOCUMENTS: Lazy<BTreeMap<String, String>> = Lazy::new(|| {
    index::index_presentation(include_str!(concat!(
        env!("OUT_DIR"),
        "/builtin_documentation.json"
    )))
    .expect("build-generated documentation must be a valid presentation-only export")
});

// Editor-only projection of the full canonical export retained in DOCUMENTS.
// Reference sections/FAQs remain in that export; hovers do not deserialize them.
#[derive(Debug, Clone, Deserialize)]
pub struct BuiltinDocExample {
    pub input: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BuiltinDocLink {
    pub label: String,
    pub url: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum BuiltinDocRelated {
    Name(String),
    Link(BuiltinDocLink),
}

#[derive(Debug, Clone, Deserialize)]
pub struct BuiltinDocJson {
    pub slug: Option<String>,
    pub summary: Option<String>,
    pub description: Option<String>,
    pub examples: Option<Vec<BuiltinDocExample>>,
    pub links: Option<Vec<BuiltinDocLink>>,
    pub related: Option<Vec<BuiltinDocRelated>>,
}

static DOC_CACHE: Lazy<Mutex<BTreeMap<String, BuiltinDocJson>>> =
    Lazy::new(|| Mutex::new(BTreeMap::new()));

pub fn builtin_json_str(name: &str) -> Option<&'static str> {
    DOCUMENTS
        .get(&name.to_ascii_lowercase())
        .map(String::as_str)
}

pub fn builtin_doc(name: &str) -> Option<BuiltinDocJson> {
    let key = name.to_ascii_lowercase();
    if let Some(document) = DOC_CACHE.lock().expect("documentation cache").get(&key) {
        return Some(document.clone());
    }
    let json = builtin_json_str(&key)?;
    let document: BuiltinDocJson = serde_json::from_str(json)
        .unwrap_or_else(|error| panic!("invalid build-generated documentation for {key}: {error}"));
    DOC_CACHE
        .lock()
        .expect("documentation cache")
        .insert(key, document.clone());
    Some(document)
}
