//! Post-open and post-scan facts for one numeric-text complete call.
//! The runtime derives these only from the retained bounded source and its
//! parsed result; a filename or pre-open metadata estimate is insufficient.

use runmat_types::NumericClass;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum ReadMatrixResultClass {
    Numeric(NumericClass),
    Logical,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct ReadMatrixTextResourceFacts {
    pub opened_extent: u64,
    pub source_resource_revision: [u8; 32],
    pub rows: u64,
    pub columns: u64,
    pub result_class: ReadMatrixResultClass,
}
