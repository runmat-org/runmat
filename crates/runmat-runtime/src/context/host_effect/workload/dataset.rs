//! Dataset namespace effects whose path and result facts must be certified together.

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum DatasetPathOperation {
    Create,
    Open,
    Exists,
    Delete,
    Copy,
    Move,
    Import,
    Export,
    List,
    Inspect,
}
