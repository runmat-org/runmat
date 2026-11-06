//! Hover for a circuit component whose params are computed by `.m` functions (PCB-38):
//! `component RS = { part = "R0402-10K", value = rcResistor(fc = 160 Hz, c = 100 nF) as ohm }`.
//!
//! The editor doesn't run the call, so the hover reads it from the circuit source: the
//! function, the `.m` file it resolves to, and the result when the projection has it.

use super::DesignFiles;
use runmat_circuit::runtime::values::function_file;
use serde_json::Value;

/// Markdown lines for the computed params of the component `name` in `path`: the param, its
/// value when `row` (the component's index row) has one, the function and its file.
pub(super) fn computed_lines(path: &str, text: &str, files: &DesignFiles, name: &str, row: Option<&Value>) -> Vec<String> {
    let Some(design) = runmat_circuit::syntax::parse_circuit(path, text).design else {
        return Vec::new();
    };
    let Some(component) = design.sheets.values().flat_map(|s| &s.components).find(|c| c.name == name) else {
        return Vec::new();
    };
    component
        .params
        .iter()
        .filter_map(|(param, value)| {
            let call = value.call.as_ref()?;
            let file = function_file(path, &call.function, files)
                .map(|f| format!(" in `{}`", f.trim_start_matches('/')))
                .unwrap_or_default();
            let result = row
                .and_then(|r| r["params"][param.as_str()].as_str())
                .map(|v| format!(" = {v}"))
                .unwrap_or_default();
            Some(format!(
                "**{param}**{result}, computed by `{}`{file}. The Desktop runs it in the RunMat runtime when the circuit compiles; Properties shows the result.",
                call.function
            ))
        })
        .collect()
}
