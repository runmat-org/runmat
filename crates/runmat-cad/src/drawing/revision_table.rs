//! A drawing's revision table (ASME Y14.35, ISO 7200): one row per released revision, oldest
//! first, written in the drawing itself:
//! `table revisions = revision(rows = [{ rev = "A", description = "Initial release", date = "2026-11-05", by = "RunMat" }], sheet = "1", at = [10, 287])`.
//! The title block's revision field reads the newest row when the drawn part sets no
//! `revision` property.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevisionRow {
    /// `A`, `B`, `01`.
    pub rev: String,
    pub description: String,
    /// As the drawing writes it (`2026-11-05`).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub date: String,
    /// Who approved the change.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub by: String,
}

/// The newest revision across the drawing's revision tables.
pub fn latest(tables: &[super::DrawingTable]) -> Option<&str> {
    tables.iter().rev().find_map(|t| match &t.kind {
        super::TableKind::Revision { rows } => rows.last().map(|r| r.rev.as_str()),
        _ => None,
    })
}
