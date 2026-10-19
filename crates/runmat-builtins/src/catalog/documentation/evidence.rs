use super::BuiltinDocumentationLink;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinEvidenceKind {
    UnitTest,
    IntegrationTest,
    BrowserTest,
    ProviderTest,
    WgpuTest,
    ConformanceTest,
    Validation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinEvidenceReference {
    pub kind: BuiltinEvidenceKind,
    pub label: &'static str,
    pub location: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinDocumentationEvidence {
    pub implementation: &'static [BuiltinDocumentationLink],
    pub verification: &'static [BuiltinEvidenceReference],
    pub notes: &'static [&'static str],
}

impl BuiltinDocumentationEvidence {
    pub const EMPTY: Self = Self {
        implementation: &[],
        verification: &[],
        notes: &[],
    };
}
