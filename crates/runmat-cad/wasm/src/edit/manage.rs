//! Manage commands on a part, run from the canvas.
//!
//! - `manage.material`: a library material (`material.steel_1018`) on the part's outputs, or on
//!   picked parts and bodies. A body that is an output's body sets that output's `material`;
//!   any other body gets a `material = { body = …, material = … }` line. A part with no output
//!   gets one for its body (or a line per body when it has several).
//! - `manage.appearance`: a display color (sRGB hex) on the part's outputs or the picked parts
//!   and bodies. A body takes a color through its output, so a body without one is promoted
//!   first ([`promote`]).
//! - `manage.equations`: sets an input's default, or a named value's expression, or adds a
//!   named value (`let width2 = input.width * 2`).
//! - `part.body-promote`: the body becomes a part output of this file, with its own name, part
//!   number and material (`output bracket = { body = feature.pad.body }`).
//! - `manage.split-to-files` ([`split`]): each body becomes its own `.part` beside this one and
//!   an assembly places them where they sit.

pub(super) mod split;

use super::{model, text_value, EditRequest};
use crate::features::picks_of;
use crate::picks::{lower_text, Picks};
use runmat_cad::ids::{LetName, MaterialRef, PartName};
use runmat_cad::model::{MaterialAssignment, PartDocument, PartOutput};
use runmat_cad::refs::Reference;
use runmat_cad::syntax::edit::{edit_part, EditFailure, SourceChange};
use runmat_cad::value::{LetDecl, ValueExpr};

/// What a material or appearance applies to.
enum Target {
    Output(String),
    Body(Reference, String),
}

/// The command's picks in `field`, else the selection's parts and bodies; each as an output or
/// a body. Empty: the whole part.
fn targets(request: &EditRequest, field: &str, picks: &Picks) -> Result<Vec<Target>, EditFailure> {
    let mut ids = picks_of(&request.values, field);
    if ids.is_empty() {
        ids = request
            .selection
            .iter()
            .filter(|s| s.starts_with("body:") || s.starts_with("part:"))
            .cloned()
            .collect();
    }
    ids.iter()
        .map(|id| match id.strip_prefix("part:") {
            Some(name) => Ok(Target::Output(name.to_owned())),
            None => {
                let value = picks.reference(id)?;
                let reference: Reference =
                    serde_json::from_value(value).map_err(|e| model(format!("`{id}`: {e}")))?;
                // The body's reference text (`feature.pad.body`) names an output made for it.
                let written = id
                    .strip_prefix("body:")
                    .and_then(|at| {
                        picks.bodies().iter().find(|b| {
                            b["index"].as_u64().map(|i| i.to_string()).as_deref() == Some(at)
                        })
                    })
                    .and_then(|b| b["reference"].as_str())
                    .unwrap_or(id)
                    .to_owned();
                Ok(Target::Body(reference, written))
            }
        })
        .collect()
}

pub(super) fn is_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The part's own name, from its file (`fan-mount.part` → `fan_mount`).
pub(super) fn part_stem(path: &str) -> String {
    let file = path.rsplit('/').next().unwrap_or(path);
    let stem: String = file
        .split('.')
        .next()
        .unwrap_or(file)
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if stem.starts_with(|c: char| c.is_ascii_digit()) {
        format!("p{stem}")
    } else {
        stem
    }
}

/// A name for a body's output: its feature (`feature.pad.body` → `pad`), else `body<n>`.
pub(super) fn body_name(reference_text: &str, at: usize) -> String {
    reference_text
        .strip_prefix("feature.")
        .and_then(|t| t.strip_suffix(".body"))
        .filter(|n| is_name(n))
        .map(str::to_owned)
        .unwrap_or_else(|| format!("body{}", at + 1))
}

fn output_mut<'a>(
    document: &'a mut PartDocument,
    name: &str,
) -> Result<&'a mut PartOutput, EditFailure> {
    document
        .parts
        .iter_mut()
        .find(|p| p.name.0 == name)
        .ok_or_else(|| model(format!("the part has no output `{name}`")))
}

fn new_output(name: &str, body: Reference) -> PartOutput {
    PartOutput {
        name: PartName::new(name),
        body,
        part_number: None,
        material: None,
        color: None,
        properties: Vec::new(),
        label: None,
    }
}

/// The output whose body is `body`, made when there is none (named after its feature).
fn output_for<'a>(
    document: &'a mut PartDocument,
    body: &Reference,
    text: &str,
    at: usize,
) -> &'a mut PartOutput {
    if let Some(i) = document.parts.iter().position(|p| &p.body == body) {
        return &mut document.parts[i];
    }
    let stem = body_name(text, at);
    let name = (1..)
        .map(|n| {
            if n == 1 {
                stem.clone()
            } else {
                format!("{stem}{n}")
            }
        })
        .find(|n| !document.parts.iter().any(|p| p.name.0 == *n))
        .expect("an unused name");
    document.parts.push(new_output(&name, body.clone()));
    document.parts.last_mut().expect("just pushed")
}

/// Every display body as (reference, reference text).
fn all_bodies(picks: &Picks) -> Result<Vec<(Reference, String)>, EditFailure> {
    picks
        .bodies()
        .iter()
        .filter_map(|b| b["reference"].as_str())
        .map(|text| {
            let reference = serde_json::from_value(lower_text(text)?)
                .map_err(|e| model(format!("`{text}`: {e}")))?;
            Ok((reference, text.to_owned()))
        })
        .collect()
}

fn library_material(request: &EditRequest) -> Result<String, EditFailure> {
    let material = text_value(&request.values, "material").trim();
    if material.is_empty() {
        return Err(model("choose a material"));
    }
    if runmat_cad::model::materials::library_label(material).is_none() {
        return Err(model(format!(
            "`{material}` isn't in the material library; choose one from the list"
        )));
    }
    Ok(if material.starts_with("material.") {
        material.to_owned()
    } else {
        format!("material.{material}")
    })
}

fn material(
    path: &str,
    text: &str,
    request: &EditRequest,
    picks: &Picks,
) -> Result<(SourceChange, String), EditFailure> {
    let material = library_material(request)?;
    let label = runmat_cad::model::materials::library_label(&material)
        .unwrap_or(&material)
        .to_owned();
    let targets = targets(request, "bodies", picks)?;
    let bodies = if targets.is_empty() && picks.document.parts.is_empty() {
        all_bodies(picks)?
    } else {
        Vec::new()
    };
    let stem = part_stem(path);
    let change = edit_part(text, path, |document| {
        let set = MaterialRef::new(&material);
        if targets.is_empty() {
            match (document.parts.is_empty(), bodies.as_slice()) {
                (false, _) => document
                    .parts
                    .iter_mut()
                    .for_each(|p| p.material = Some(set.clone())),
                (true, []) => return Err(model("the part has no body to take a material yet")),
                (true, [(body, _)]) => {
                    let mut output = new_output(&stem, body.clone());
                    output.material = Some(set);
                    document.parts.push(output);
                }
                (true, many) => {
                    document
                        .materials
                        .retain(|m| !many.iter().any(|(b, _)| *b == m.body));
                    document
                        .materials
                        .extend(many.iter().map(|(b, _)| MaterialAssignment {
                            body: b.clone(),
                            material: set.clone(),
                        }));
                }
            }
            return Ok(());
        }
        for target in &targets {
            match target {
                Target::Output(name) => output_mut(document, name)?.material = Some(set.clone()),
                Target::Body(body, _) => {
                    match document.parts.iter_mut().find(|p| &p.body == body) {
                        Some(output) => output.material = Some(set.clone()),
                        None => match document.materials.iter_mut().find(|m| &m.body == body) {
                            Some(line) => line.material = set.clone(),
                            None => document.materials.push(MaterialAssignment {
                                body: body.clone(),
                                material: set.clone(),
                            }),
                        },
                    }
                }
            }
        }
        Ok(())
    })?;
    let on = if targets.is_empty() {
        "the part".to_owned()
    } else {
        format!(
            "{} item{}",
            targets.len(),
            if targets.len() == 1 { "" } else { "s" }
        )
    };
    Ok((change, format!("Set {label} on {on}")))
}

/// `#rgb`, `#rrggbb` or `#rrggbbaa` (the `#` optional) as `#RRGGBB[AA]`.
fn color(request: &EditRequest) -> Result<String, EditFailure> {
    let typed = text_value(&request.values, "color").trim();
    let hex = typed.strip_prefix('#').unwrap_or(typed);
    let hex = if hex.len() == 3 {
        hex.chars().flat_map(|c| [c, c]).collect()
    } else {
        hex.to_owned()
    };
    let written = format!("#{}", hex.to_ascii_uppercase());
    if runmat_cad::appearance::parse_srgb_hex(&written).is_none() {
        return Err(model(format!(
            "`{typed}` isn't a color; write sRGB hex such as #2F6DB5"
        )));
    }
    Ok(written)
}

fn appearance(
    path: &str,
    text: &str,
    request: &EditRequest,
    picks: &Picks,
) -> Result<(SourceChange, String), EditFailure> {
    let color = color(request)?;
    let targets = targets(request, "targets", picks)?;
    let bodies = if targets.is_empty() && picks.document.parts.is_empty() {
        all_bodies(picks)?
    } else {
        Vec::new()
    };
    let change = edit_part(text, path, |document| {
        if targets.is_empty() {
            if document.parts.is_empty() {
                let [(body, _)] = bodies.as_slice() else {
                    return Err(model(if bodies.is_empty() {
                        "the part has no body to color yet"
                    } else {
                        "the part has several bodies; pick the ones to color"
                    }));
                };
                document
                    .parts
                    .push(new_output(&part_stem(path), body.clone()));
            }
            document
                .parts
                .iter_mut()
                .for_each(|p| p.color = Some(color.clone()));
            return Ok(());
        }
        for (at, target) in targets.iter().enumerate() {
            match target {
                Target::Output(name) => output_mut(document, name)?.color = Some(color.clone()),
                Target::Body(body, written) => {
                    output_for(document, body, written, at).color = Some(color.clone())
                }
            }
        }
        Ok(())
    })?;
    Ok((
        change,
        format!(
            "Colored {}",
            if targets.is_empty() {
                "the part".to_owned()
            } else {
                format!(
                    "{} item{}",
                    targets.len(),
                    if targets.len() == 1 { "" } else { "s" }
                )
            }
        ),
    ))
}

fn equations(
    path: &str,
    text: &str,
    request: &EditRequest,
) -> Result<(SourceChange, String), EditFailure> {
    let name = text_value(&request.values, "name").trim();
    let name = name
        .strip_prefix("input.")
        .or_else(|| name.strip_prefix("let."))
        .unwrap_or(name);
    if !is_name(name) {
        return Err(model(format!(
            "`{name}` isn't a name: use letters, digits and underscores, starting with a letter"
        )));
    }
    let typed = text_value(&request.values, "value").trim();
    if typed.is_empty() {
        return Err(model(format!("give {name} a value or an expression")));
    }
    let value: ValueExpr = serde_json::from_value(lower_text(typed)?)
        .map_err(|_| model(format!("`{typed}` isn't a value or an expression")))?;
    let mut summary = String::new();
    let change = edit_part(text, path, |document| {
        if let Some(input) = document.inputs.iter_mut().find(|i| i.name.0 == name) {
            input.default = value.clone();
            summary = format!("Set input {name} to {typed}");
        } else if let Some(line) = document.lets.iter_mut().find(|l| l.name.0 == name) {
            line.value = value.clone();
            summary = format!("Set {name} = {typed}");
        } else {
            document.lets.push(LetDecl {
                name: LetName::new(name),
                kind: None,
                value: value.clone(),
            });
            summary = format!("Added {name} = {typed}");
        }
        Ok(())
    })?;
    Ok((change, summary))
}

/// `part.body-promote`: the picked (or selected) body becomes output `name`.
fn promote(
    path: &str,
    text: &str,
    request: &EditRequest,
    picks: &Picks,
) -> Result<(SourceChange, String), EditFailure> {
    let mut ids = picks_of(&request.values, "body");
    if ids.is_empty() {
        ids = request
            .selection
            .iter()
            .filter(|s| s.starts_with("body:"))
            .cloned()
            .collect();
    }
    let [id] = ids.as_slice() else {
        return Err(model(if ids.is_empty() {
            "pick the body to promote"
        } else {
            "pick one body to promote"
        }));
    };
    let body: Reference =
        serde_json::from_value(picks.reference(id)?).map_err(|e| model(format!("`{id}`: {e}")))?;
    let name = text_value(&request.values, "name").trim().to_owned();
    if !is_name(&name) {
        return Err(model(format!("`{name}` isn't a part name: use letters, digits and underscores, starting with a letter")));
    }
    let change = edit_part(text, path, |document| {
        if let Some(output) = document.parts.iter().find(|p| p.body == body) {
            return Err(model(format!("that body is part {} already", output.name)));
        }
        if document.parts.iter().any(|p| p.name.0 == name) {
            return Err(model(format!(
                "the file has a part `{name}` already; pick another name"
            )));
        }
        document.parts.push(new_output(&name, body.clone()));
        Ok(())
    })?;
    Ok((change, format!("Promoted the body to part {name}")))
}

/// The part manage commands; None for any other command.
pub(super) fn edit(
    path: &str,
    text: &str,
    request: &EditRequest,
    picks: &Picks,
) -> Option<Result<(SourceChange, String), EditFailure>> {
    Some(match request.command.as_str() {
        "manage.material" => material(path, text, request, picks),
        "manage.appearance" => appearance(path, text, request, picks),
        "manage.equations" => equations(path, text, request),
        "part.body-promote" => promote(path, text, request, picks),
        _ => return None,
    })
}

#[cfg(test)]
mod tests;
