//! Assembly catalog commands from the canvas, as structured edits of the assembly document
//! written back as the smallest text change.
//!
//! - `assembly.delete`, `assembly.suppress`, `assembly.fix`: on the selected components.
//! - `assembly.insert`: a part, assembly or circuit board as a new component ([`place`]).
//! - `assembly.move`: the triad's translation and rotation added to the components' placement.
//! - `assembly.pattern`: a linear or circular component pattern.
//! - `assembly.mate`: coincident, concentric, distance, angle, parallel, perpendicular and
//!   tangent mates (and the connector-based joints) between faces, edges or whole components,
//!   each pick becoming a mate connector on its component ([`mate`], [`frames`]).
//! - `explode.new`, `explode.step`, `explode.radial`, `explode.lines`: exploded views, their
//!   steps and trail lines ([`explode`]).
//!
//! Picks are Outline IDs: `occurrence:<uuid>` for a component, and
//! `occurrence:<uuid>/face:<body>:<face>` or `occurrence:<uuid>/edge:<body>:<edge>` for a face
//! or edge of one, indexed into the compiled display of its definition.

mod explode;
mod frames;
mod mate;
mod place;
#[cfg(test)]
mod explode_tests;
#[cfg(test)]
mod tests;

use super::{model, text_value, EditRequest};
use crate::runtime::{compile_assembly, Files};
use runmat_cad::assembly::model::AssemblyDocument;
use runmat_cad::syntax::edit::{diff_source_in, EditFailure, SourceChange};
use runmat_cad::syntax::{parse_assembly_in, print_assembly, ASSEMBLY};
use runmat_design_syntax::edit::TextEdit;
use runmat_types::Span;
use serde_json::Value;

/// The compiled assembly an edit reads picks against: occurrence rows (uuid, path) and the
/// display (prototypes with their faces and edges, occurrence transforms).
pub(super) struct Compiled {
    rows: Vec<Value>,
    display: Value,
}

impl Compiled {
    fn new(path: &str, text: &str, files: &Files) -> Result<Self, EditFailure> {
        let (index, display, _) = compile_assembly(path, text, files)
            .map_err(|_| model("the assembly doesn't compile"))?;
        Ok(Self {
            rows: index["occurrences"].as_array().cloned().unwrap_or_default(),
            display,
        })
    }

    /// The authored name of the top-level component `uuid` names.
    fn occurrence_name(&self, uuid: &str) -> Result<String, EditFailure> {
        let row = self
            .rows
            .iter()
            .find(|r| r["uuid"].as_str() == Some(uuid))
            .ok_or_else(|| model("the selected component isn't in this assembly any more"))?;
        let path = row["path"].as_str().unwrap_or_default();
        if path.contains('/') {
            return Err(model(format!(
                "`{path}` is inside a sub-assembly; open that assembly to change it"
            )));
        }
        Ok(path.to_owned())
    }

    /// The display occurrence `uuid` names: its prototype index and world transform.
    fn placed(&self, uuid: &str) -> Option<(usize, Vec<f64>)> {
        self.display["occurrences"]
            .as_array()?
            .iter()
            .find(|o| o["uuid"].as_str() == Some(uuid))
            .map(|o| {
                let prototype = o["prototype"].as_u64().unwrap_or(0) as usize;
                let transform = o["transform"]
                    .as_array()
                    .map(|t| t.iter().filter_map(Value::as_f64).collect())
                    .unwrap_or_default();
                (prototype, transform)
            })
    }

    /// A body of prototype `prototype`'s part display.
    fn body(&self, prototype: usize, body: usize) -> Option<&Value> {
        self.display["prototypes"][prototype]["part"]["bodies"]
            .as_array()?
            .iter()
            .find(|b| b["index"].as_u64() == Some(body as u64))
    }
}

/// A pick split into its component and, for a face or edge pick, the entity on it.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Pick {
    pub uuid: String,
    pub entity: Option<Entity>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Entity {
    Face { body: usize, face: usize },
    Edge { body: usize, edge: usize },
}

pub(super) fn parse_pick(id: &str) -> Option<Pick> {
    let rest = id.strip_prefix("occurrence:")?;
    let (uuid, entity) = match rest.split_once('/') {
        None => (rest, None),
        Some((uuid, tail)) => {
            let (kind, at) = tail.split_once(':')?;
            let (body, index) = at.split_once(':')?;
            let (body, index) = (body.parse().ok()?, index.parse().ok()?);
            let entity = match kind {
                "face" => Entity::Face { body, face: index },
                "edge" => Entity::Edge { body, edge: index },
                _ => return None,
            };
            (uuid, Some(entity))
        }
    };
    Some(Pick {
        uuid: uuid.to_owned(),
        entity,
    })
}

/// The selected components' names (the command's `components` picks, else the selection).
fn selected_names(
    compiled: &Compiled,
    request: &EditRequest,
) -> Result<Vec<String>, EditFailure> {
    let mut ids = crate::features::picks_of(&request.values, "components");
    if ids.is_empty() {
        ids = request.selection.clone();
    }
    let mut names = Vec::new();
    for pick in ids.iter().filter_map(|id| parse_pick(id)) {
        let name = compiled.occurrence_name(&pick.uuid)?;
        if !names.contains(&name) {
            names.push(name);
        }
    }
    if names.is_empty() {
        return Err(model("select a component first"));
    }
    Ok(names)
}

fn find(document: &AssemblyDocument, name: &str) -> Result<usize, EditFailure> {
    document
        .occurrences
        .iter()
        .position(|o| o.name == name)
        .ok_or_else(|| model(format!("the assembly has no component `{name}`")))
}

/// Applies repair `values.repair` (an index, 0 when absent) of the assembly's diagnostic
/// `values.diagnostic` (its id). This file's own repairs apply here; a repair that edits
/// another file says which.
fn repair(
    path: &str,
    text: &str,
    files: &Files,
    document: &mut AssemblyDocument,
    values: &serde_json::Map<String, serde_json::Value>,
) -> Result<String, EditFailure> {
    use runmat_cad::diagnostics::Repair;
    let number = |id: &str| values.get(id).and_then(|v| v.as_u64().or_else(|| v.as_str()?.parse().ok())).map(|v| v as usize);
    let id = number("diagnostic").ok_or_else(|| model("name the problem to repair"))?;
    let at = number("repair").unwrap_or(0);
    let (index, _, _) = crate::runtime::compile_assembly(path, text, files)
        .map_err(|_| model("the assembly doesn't compile; fix the source first"))?;
    let diagnostic = index["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|d| d["id"].as_u64() == Some(id as u64))
        .ok_or_else(|| model("that problem is gone; the assembly changed since it was reported"))?;
    let repair: Repair = serde_json::from_value(diagnostic["repairs"][at].clone())
        .map_err(|_| model("that problem has no such repair"))?;
    match repair {
        Repair::SuppressMate { assembly, mate } if assembly.0 == path => {
            let found = document
                .mates
                .iter_mut()
                .find(|m| m.name == mate)
                .ok_or_else(|| model(format!("the assembly has no mate `{mate}`")))?;
            found.suppressed = true;
            Ok(format!("Suppressed mate {mate}"))
        }
        Repair::SuppressMate { assembly, .. } => Err(model(format!("this repair edits {}; open it to apply", assembly.0))),
        other => Err(model(format!("`{other:?}` applies from the source editor in this build"))),
    }
}

pub(super) fn edit(
    path: &str,
    text: &str,
    files: &Files,
    request: &EditRequest,
) -> Result<(SourceChange, String), EditFailure> {
    let scope = crate::runtime::module_scope(path, text, files);
    let parsed = parse_assembly_in(path, text, &scope);
    let (Some(mut document), true) = (parsed.document, parsed.diagnostics.is_empty()) else {
        return Err(EditFailure::Invalid(parsed.diagnostics));
    };
    let compiled = Compiled::new(path, text, files)?;
    let summary = match request.command.as_str() {
        "assembly.delete" => {
            let names = selected_names(&compiled, request)?;
            for name in &names {
                let at = find(&document, name)?;
                document.occurrences.remove(at);
            }
            format!("Deleted {}", names.join(", "))
        }
        "assembly.suppress" => {
            let names = selected_names(&compiled, request)?;
            for name in &names {
                let at = find(&document, name)?;
                document.occurrences[at].suppressed = true;
            }
            format!("Suppressed {}", names.join(", "))
        }
        "assembly.fix" => {
            let names = selected_names(&compiled, request)?;
            let fixed = text_value(&request.values, "state") != "float";
            for name in &names {
                let at = find(&document, name)?;
                document.occurrences[at].fixed = fixed;
            }
            format!(
                "{} {}",
                if fixed { "Fixed" } else { "Floated" },
                names.join(", ")
            )
        }
        "assembly.insert" => place::insert(&mut document, files, &request.values)?,
        "assembly.move" => {
            let names = selected_names(&compiled, request)?;
            place::move_components(&mut document, &names, &request.values)?
        }
        "assembly.pattern" => {
            let names = selected_names(&compiled, request)?;
            place::pattern(&mut document, &compiled, &names, &request.values)?
        }
        "assembly.mate" => mate::mate(&mut document, &compiled, &request.values)?,
        "repair.repair" => repair(path, text, files, &mut document, &request.values)?,
        "explode.new" => explode::new_view(&mut document, &request.values)?,
        "explode.step" | "explode.radial" => {
            let names = selected_names(&compiled, request)?;
            let radial = request.command == "explode.radial";
            explode::step(&mut document, &compiled, &names, &request.values, &request.selection, radial)?
        }
        "explode.lines" => explode::lines(&mut document, &request.values, &request.selection)?,
        other => {
            return Err(model(format!(
                "`{other}` edits assemblies through the source editor in this build"
            )))
        }
    };
    // The minimal diff can't yet place every occurrence statement (one with a body block):
    // then the file is reprinted whole and the change says so.
    let change = match diff_source_in(&ASSEMBLY, path, text, &document, &parsed.uses, &[], &scope) {
        Ok(change) => change,
        Err(EditFailure::Model(_)) => {
            let printed = print_assembly(&document, &parsed.uses).map_err(EditFailure::Print)?;
            SourceChange {
                edits: vec![TextEdit::replace(
                    Span {
                        start: 0,
                        end: text.len(),
                    },
                    printed.clone(),
                )],
                source: printed,
                reprinted: true,
            }
        }
        Err(other) => return Err(other),
    };
    Ok((change, summary))
}
