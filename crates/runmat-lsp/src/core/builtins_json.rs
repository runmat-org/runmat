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

#[derive(Debug, Clone, Deserialize)]
pub struct BuiltinDocExample {
    pub description: String,
    pub input: Option<String>,
    pub output: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BuiltinDocFaq {
    pub question: String,
    pub answer: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BuiltinDocLink {
    pub label: String,
    pub url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BuiltinDocGpuSupport {
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BuiltinDocJsonEncodeOption {
    pub name: String,
    #[serde(rename = "type")]
    pub type_name: String,
    pub default: String,
    pub description: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BuiltinDocSection {
    pub heading: String,
    pub paragraphs: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BuiltinDocJson {
    pub title: Option<String>,
    pub category: Option<String>,
    #[allow(dead_code)]
    pub keywords: Option<Vec<String>>,
    pub summary: Option<String>,
    pub description: Option<String>,
    pub behaviors: Option<Vec<String>>,
    pub sections: Option<Vec<BuiltinDocSection>>,
    pub examples: Option<Vec<BuiltinDocExample>>,
    pub faqs: Option<Vec<BuiltinDocFaq>>,
    pub links: Option<Vec<BuiltinDocLink>>,
    pub gpu_support: Option<BuiltinDocGpuSupport>,
    pub gpu_residency: Option<String>,
    pub gpu_behavior: Option<Vec<String>>,
    pub options: Option<Vec<String>>,
    pub jsonencode_options: Option<Vec<BuiltinDocJsonEncodeOption>>,
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
