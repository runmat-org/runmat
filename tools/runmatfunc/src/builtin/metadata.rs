use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct BuiltinManifest {
    pub builtins: Vec<BuiltinRecord>,
}

#[derive(Debug, Serialize)]
pub struct BuiltinRecord {
    pub name: String,
    pub category: Option<String>,
    pub summary: Option<String>,
    pub keywords: Vec<String>,
    pub accel_tags: Vec<String>,
    pub is_sink: bool,
    pub param_types: Vec<String>,
    pub return_type: String,
}
