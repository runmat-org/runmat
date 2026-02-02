use once_cell::sync::Lazy;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Mutex;

mod generated {
    include!(concat!(env!("OUT_DIR"), "/builtins_json_generated.rs"));
}

#[derive(Debug, Clone, Deserialize)]
pub struct BuiltinDocExample {
    pub description: String,
    pub input: String,
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
pub struct BuiltinDocJson {
    pub title: Option<String>,
    pub category: Option<String>,
    pub keywords: Option<Vec<String>>,
    pub summary: Option<String>,
    pub description: Option<String>,
    pub behaviors: Option<Vec<String>>,
    pub examples: Option<Vec<BuiltinDocExample>>,
    pub faqs: Option<Vec<BuiltinDocFaq>>,
    pub links: Option<Vec<BuiltinDocLink>>,
    pub gpu_support: Option<BuiltinDocGpuSupport>,
    pub gpu_residency: Option<String>,
    pub gpu_behavior: Option<Vec<String>>,
    pub options: Option<Vec<String>>,
    pub jsonencode_options: Option<Vec<BuiltinDocJsonEncodeOption>>,
}

static DOC_CACHE: Lazy<Mutex<HashMap<String, BuiltinDocJson>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub fn builtin_json_str(name: &str) -> Option<&'static str> {
    generated::builtin_json_by_name(name)
}

pub fn builtin_doc(name: &str) -> Option<BuiltinDocJson> {
    let key = name.to_ascii_lowercase();
    {
        let cache = DOC_CACHE.lock().ok()?;
        if let Some(doc) = cache.get(&key) {
            return Some(doc.clone());
        }
    }

    let json = builtin_json_str(&key)?;
    let parsed: BuiltinDocJson = serde_json::from_str(json).ok()?;

    if let Some(faqs) = &parsed.faqs {
        for faq in faqs {
            let _ = (&faq.question, &faq.answer);
        }
    }

    let mut cache = DOC_CACHE.lock().ok()?;
    cache.insert(key, parsed.clone());
    Some(parsed)
}
