//! Exact STEP AP242 from the installed kernel (runmat-cad-wasm's exchange): a part's exact
//! B-rep with its material and face names; an assembly with its hierarchy as authored, one
//! product per part configuration. `None` when the kernel cannot export (the grid test double),
//! so the caller falls back to the faceted writer.

use crate::{kind_of, text_of, DesignKind, Files, ToolError};
use runmat_cad_wasm::exchange::{assembly_step, part_step_in};

fn not_compiled(path: &str, message: String) -> ToolError {
    ToolError::new("not_compiled", format!("{path} doesn't compile (design.check lists why): {message}"))
}

/// The STEP AP242 text of the part (in its named configuration `configuration`, when given)
/// or assembly at `path`, or `None` when the installed kernel does not write STEP.
pub fn exact_step_in(path: &str, files: &Files, configuration: Option<&str>) -> Result<Option<Vec<u8>>, ToolError> {
    let text = text_of(files, path)?;
    match kind_of(path)? {
        DesignKind::Part => part_step_in(path, text, configuration).map_err(|m| not_compiled(path, m)),
        DesignKind::Assembly => assembly_step(path, text, files).map_err(|m| not_compiled(path, m)),
        _ => Ok(None),
    }
}
