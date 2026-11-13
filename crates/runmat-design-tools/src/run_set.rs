//! A run's input values (`set`): the workspace files with one design's inputs set for this run,
//! through [`runmat_cad::run_inputs`]. The design file itself is never changed.

use crate::{kind_of, text_of, DesignKind, Files, ToolError};
use runmat_cad::run_inputs::{set_assembly_inputs, set_part_inputs, RunInputError};
use std::collections::BTreeMap;

fn tool_error(path: &str, e: RunInputError) -> ToolError {
    match e {
        RunInputError::Unknown { .. } => ToolError::new("unknown_input", e.to_string()),
        RunInputError::Value { .. } => ToolError::invalid(e.to_string()),
        RunInputError::Invalid(_) => ToolError::new("not_compiled", format!("{path}: {e}")),
    }
}

/// `files` with the part or assembly at `path` reading each input of `set` (name → value text,
/// `"3.5 mm"`) as its value. An empty `set` borrows nothing new.
pub fn with_inputs(
    path: &str,
    files: &Files,
    set: &BTreeMap<String, String>,
) -> Result<Files, ToolError> {
    let text = text_of(files, path)?;
    let pairs: Vec<(String, String)> = set.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    let changed = match kind_of(path)? {
        DesignKind::Part => set_part_inputs(path, text, &pairs),
        DesignKind::Assembly => {
            let scope = runmat_cad_wasm::runtime::module_scope(path, text, files);
            set_assembly_inputs(path, text, &scope, &pairs)
        }
        other => {
            return Err(ToolError::invalid(format!(
                "`set` takes the inputs of a part or an assembly; {path} is a {other:?} file"
            )))
        }
    }
    .map_err(|e| tool_error(path, e))?;
    let mut out = files.clone();
    out.insert(path.to_string(), changed);
    Ok(out)
}
