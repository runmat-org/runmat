//! `assembly.set-property`: one field of the selected components, as Properties edits it:
//! Fixed, Flexible (a subassembly's), Reference, Representation and Configuration.

use super::{configure, find, model, representation, selected_names, Compiled};
use crate::edit::EditRequest;
use crate::runtime::Files;
use runmat_cad::assembly::model::{AssemblyDocument, DefinitionKind};
use runmat_cad::syntax::edit::EditFailure;

pub(super) fn set(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    files: &Files,
    request: &EditRequest,
) -> Result<String, EditFailure> {
    let names = selected_names(compiled, request)?;
    let [(field, value)] = request.values.iter().collect::<Vec<_>>()[..] else {
        return Err(model("set one property at a time"));
    };
    let flag = || {
        value
            .as_bool()
            .ok_or_else(|| model(format!("`{field}` is on or off")))
    };
    match field.as_str() {
        "fixed" | "flexible" | "reference" => {
            let on = flag()?;
            for name in &names {
                let at = find(document, name)?;
                let o = &mut document.occurrences[at];
                match field.as_str() {
                    "fixed" => o.fixed = on,
                    "flexible" => {
                        if o.source.definition.kind != DefinitionKind::Assembly {
                            return Err(model(format!("`{name}` isn't a subassembly")));
                        }
                        o.flexible = on
                    }
                    _ => o.reference = on,
                }
            }
            let word = match (field.as_str(), on) {
                ("reference", true) => "a reference",
                ("reference", false) => "not a reference",
                ("fixed", true) => "fixed",
                ("fixed", false) => "floating",
                (_, true) => "flexible",
                (_, false) => "rigid",
            };
            Ok(format!("Made {} {word}", names.join(", ")))
        }
        "representation" => representation(document, &names, value.as_str().unwrap_or("")),
        "configuration" => {
            configure::configure(document, files, &names, value.as_str().unwrap_or(""))
        }
        other => Err(model(format!(
            "`{other}` isn't a component property Properties can set"
        ))),
    }
}
