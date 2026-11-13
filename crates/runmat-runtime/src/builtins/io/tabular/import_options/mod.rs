//! Shared, typed import-options boundary for tabular readers and modifiers.

mod model;
mod object;
mod projection;
mod selectors;
mod variable;

pub(crate) use model::{ImportOptions, ImportSource};
pub(crate) use object::ensure_import_options_classes_registered;
pub(crate) use projection::ImportColumnProjection;
pub(crate) use variable::{
    normalize_numeric_token, EmptyFieldRule, VariableImportOptions, WhitespaceRule,
};

use crate::{build_runtime_error, RuntimeError};

pub(crate) const DELIMITED_IMPORT_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("matlab.io.text.DelimitedTextImportOptions");
pub(crate) const SPREADSHEET_IMPORT_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("matlab.io.spreadsheet.SpreadsheetImportOptions");
pub(crate) const VARIABLE_IMPORT_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("matlab.io.VariableImportOptions");

fn invalid(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier("RunMat:ImportOptions:InvalidArgument")
        .build()
}
