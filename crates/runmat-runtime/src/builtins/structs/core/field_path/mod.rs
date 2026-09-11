//! Typed field paths shared by functional field reads and writes.

mod parse;
mod plan;

#[cfg(test)]
mod tests;

pub(super) use parse::parse;
pub(super) use plan::build_plan;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PathErrorKind {
    MissingPath,
    FieldName,
    EmptySelector,
    InvalidIndex,
}

#[derive(Debug)]
pub(super) struct PathError {
    pub kind: PathErrorKind,
    pub detail: String,
}

#[derive(Debug, Default)]
pub(super) struct FieldPath {
    pub leading_index: Option<IndexSelector>,
    pub fields: Vec<FieldStep>,
    pub uses_textual_index: bool,
}

#[derive(Debug)]
pub(super) struct FieldStep {
    pub name: String,
    pub index: Option<IndexSelector>,
}

#[derive(Debug, Clone)]
pub(super) struct IndexSelector {
    pub components: Vec<IndexComponent>,
}

#[derive(Debug, Clone)]
pub(super) enum IndexComponent {
    Scalar(usize),
    Vector(Vec<usize>, Vec<usize>),
    Logical(Vec<u8>),
    End,
}
