//! Structured editing for the canonical `.fea` study schema.
//!
//! Authored values are deserialized into the same domain contracts used by execution before they
//! enter the YAML tree. This keeps the editor API from growing a parallel physics vocabulary.

mod contracts;
mod operations;
mod summary;
mod yaml;

use std::str::FromStr;

use serde_json::Value as JsonValue;

pub use contracts::*;
use operations::{create_source, mutate};
use summary::with_diff;
pub use summary::{summarize_fea_study_document, summarize_fea_study_document_contract};
use yaml::{decode, parse_document, serialize_document};

pub fn apply_fea_study_document_operation(
    operation: &str,
    path: &str,
    source: Option<&str>,
    input: JsonValue,
) -> Result<FeaStudyDocumentOperationOutput, String> {
    apply_fea_study_document_operation_typed(
        FeaStudyDocumentOperation::from_str(operation)?,
        path,
        source,
        input,
    )
}

pub fn apply_fea_study_document_operation_typed(
    operation: FeaStudyDocumentOperation,
    path: &str,
    source: Option<&str>,
    input: JsonValue,
) -> Result<FeaStudyDocumentOperationOutput, String> {
    if operation == FeaStudyDocumentOperation::Create {
        let next = create_source(path, decode(input, "create")?)?;
        return Ok(FeaStudyDocumentOperationOutput {
            result: with_diff(path, "", &next, &["document"])?,
            source: next,
            write: true,
        });
    }
    let current = source.unwrap_or_default();
    if operation == FeaStudyDocumentOperation::GetSummary {
        return Ok(FeaStudyDocumentOperationOutput {
            source: current.to_string(),
            result: summarize_fea_study_document_contract(path, current)?,
            write: false,
        });
    }
    let mut document = parse_document(current)?;
    let section = mutate(operation, &mut document, input)?;
    let next = serialize_document(&document)?;
    Ok(FeaStudyDocumentOperationOutput {
        result: with_diff(path, current, &next, &[section])?,
        source: next,
        write: true,
    })
}

#[cfg(test)]
mod tests;
