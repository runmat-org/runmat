use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinDocumentation {
    pub summary: &'static str,
    pub keywords: &'static [&'static str],
    pub related: &'static [&'static str],
    pub introduced: Option<&'static str>,
    pub status: Option<&'static str>,
    pub examples: &'static [&'static str],
}
