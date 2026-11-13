use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinDocumentationStatus {
    Stable,
    Experimental,
    Partial,
    Deprecated,
}

/// A deliberately named section of user-facing prose. The heading is content,
/// not a selector used by compiler or runtime behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinDocumentationSection {
    pub heading: &'static str,
    pub paragraphs: &'static [&'static str],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinDocumentationFaq {
    pub question: &'static str,
    pub answer: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinDocumentationLinkTarget {
    Builtin(&'static str),
    Documentation(&'static str),
    Source(&'static str),
    External(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinDocumentationLink {
    pub label: &'static str,
    pub target: BuiltinDocumentationLinkTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinDocumentationMediaKind {
    Image,
    Video,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinDocumentationMedia {
    pub kind: BuiltinDocumentationMediaKind,
    pub url: &'static str,
    pub alt: &'static str,
}
