//! The part's library commands ([`runmat_cad::model::library`]):
//!
//! - `weldment.profile-library`: place structural members in a standard size
//!   (`standard(designation = "IPE 200")`), a sketch of this part
//!   (`sketch_region(sketch = z)`) or a project profile from `library/profiles/`, whose sketch is
//!   copied into the part; or save a sketch of this part as a project profile.
//! - `part.library`: insert a library feature on a picked face or plane, or a library part as a
//!   body, as a `library(def = …)` instance whose inputs stay editable as its `args`.

use super::{fresh_id, model, text_value, EditRequest};
use crate::features::picks_of;
use crate::picks::{lower_text, Picks};
use crate::runtime::{resolve_path, Files};
use runmat_cad::ids::{FeatureId, InputName};
use runmat_cad::model::datum::PlaneRef;
use runmat_cad::model::detail::LibraryInstance;
use runmat_cad::model::library;
use runmat_cad::model::weldment::{standard, SectionProperties};
use runmat_cad::model::{Feature, FeatureKind, PartDocument};
use runmat_cad::syntax::edit::{edit_part, EditFailure, SourceChange};
use runmat_cad::syntax::{parse_part, print_part};
use runmat_cad::value::ValueExpr;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

type Values = Map<String, Value>;

/// A library command's result: the part's change (if any), files it creates, and the summary.
pub(in crate::edit) struct LibraryEdit {
    pub own: Option<SourceChange>,
    pub created: Vec<(String, String)>,
    pub summary: String,
}

/// The commands this module answers.
pub(in crate::edit) fn edit(
    path: &str,
    text: &str,
    files: &Files,
    request: &EditRequest,
) -> Option<Result<LibraryEdit, EditFailure>> {
    Some(match request.command.as_str() {
        "weldment.profile-library" => match text_value(&request.values, "action") {
            "save" => save_profile(path, text, files, &request.values),
            _ => place_members(path, text, files, request),
        },
        "part.library" => insert(path, text, files, request),
        _ => return None,
    })
}

fn document(path: &str, text: &str) -> Result<PartDocument, EditFailure> {
    let parsed = parse_part(path, text);
    parsed
        .document
        .ok_or(EditFailure::Invalid(parsed.diagnostics))
}

fn sketch_of(document: &PartDocument, id: &str) -> Option<runmat_cad::model::sketch::Sketch> {
    match document
        .features
        .iter()
        .find(|f| f.id.0 == id)
        .map(|f| &f.kind)
    {
        Some(FeatureKind::Sketch(s)) => Some(s.clone()),
        _ => None,
    }
}

/// A sketch's region section, solved on its own on `origin.xy` (a section is in the sketch's
/// plane, wherever that plane is). Fails when the sketch needs other features or has no closed
/// region.
fn sketch_section(sketch: &runmat_cad::model::sketch::Sketch) -> Result<SectionProperties, String> {
    let mut alone = sketch.clone();
    alone.plane = PlaneRef::origin_xy();
    let mut doc = PartDocument::new("/section/profile.part");
    doc.features.push(Feature::new(
        library::PROFILE_SKETCH,
        FeatureKind::Sketch(alone),
    ));
    let text = print_part(&doc, &[]).map_err(|e| e.to_string())?;
    let part = crate::runtime::regenerate_part_in(
        "/section/profile.part",
        &text,
        None,
        &Default::default(),
    )
    .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
    if let Some(d) = part.result.diagnostics.errors().next() {
        return Err(d.message.clone());
    }
    let def = sketch_of(&part.document, library::PROFILE_SKETCH).ok_or("the sketch is gone")?;
    let (solved, frame) = part
        .result
        .state
        .sketches
        .get(&FeatureId::new(library::PROFILE_SKETCH))
        .ok_or("the sketch didn't solve")?;
    let s = runmat_cad::evaluate::section::sketch_section(&def, solved, frame)
        .ok_or("the sketch has no closed region")?;
    // In the sketch's own axes: the frame of origin.xy is the world's.
    Ok(SectionProperties {
        area: s.area,
        centroid: [s.centroid[0], s.centroid[1]],
        ixx: s.ixx,
        iyy: s.iyy,
        ixy: s.ixy,
    })
}

/// `z_rail`, `angle_40`: a name a profile file and a sketch id can both carry.
fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_lowercase() || c == '_')
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

fn profile_file(path: &str, files: &Files, name: &str) -> String {
    resolve_path(path, &format!("{}/{name}.part", library::PROFILES), files)
}

fn save_profile(
    path: &str,
    text: &str,
    files: &Files,
    values: &Values,
) -> Result<LibraryEdit, EditFailure> {
    let document = document(path, text)?;
    let picks = Picks::new(&document, path, text);
    let pick = picks_of(values, "sketch")
        .into_iter()
        .next()
        .ok_or_else(|| model("pick the sketch to save as a profile"))?;
    let id = picks.sketch(&pick)?;
    let name = text_value(values, "name").trim();
    let name = if name.is_empty() { id.as_str() } else { name };
    if !valid_name(name) {
        return Err(model(format!(
            "`{name}` isn't a profile name: use lowercase letters, digits and _"
        )));
    }
    let target = profile_file(path, files, name);
    if files.contains_key(&target) {
        return Err(model(format!(
            "the project already has a profile `{name}`; pick another name"
        )));
    }
    let sketch = sketch_of(&document, &id).expect("picked sketch");
    let section = sketch_section(&sketch)
        .map_err(|e| model(format!("sketch `{id}` can't be a profile: {e}")))?;
    let mut alone = sketch;
    alone.plane = PlaneRef::origin_xy();
    let mut profile = PartDocument::new(&target);
    profile.header = document.header.clone();
    profile.features.push(Feature::new(
        library::PROFILE_SKETCH,
        FeatureKind::Sketch(alone),
    ));
    let printed = print_part(&profile, &[]).map_err(EditFailure::Print)?;
    Ok(LibraryEdit {
        own: None,
        created: vec![(target.clone(), printed)],
        summary: format!(
            "Saved sketch {id} as profile {name} ({}): {}",
            target
                .rsplit("/library/")
                .next()
                .map_or(target.as_str(), |t| t),
            section.text()
        ),
    })
}

/// The member's profile model, a sketch to copy in, and the section.
type ChosenProfile = (
    Value,
    Option<(String, runmat_cad::model::sketch::Sketch)>,
    Option<SectionProperties>,
);

/// What the Profile field names, as the member's profile: the model JSON, a sketch to copy in
/// (a project profile), and the section for the summary.
fn chosen_profile(
    path: &str,
    files: &Files,
    document: &PartDocument,
    written: &str,
) -> Result<ChosenProfile, EditFailure> {
    let written = written.trim();
    if written.is_empty() {
        return Err(model(
            "choose a profile: a standard size such as IPE 200, a project profile, or sketch.<name>",
        ));
    }
    // A sketch of this part.
    let local = written.strip_prefix("sketch.").unwrap_or(written);
    if let Some(sketch) = sketch_of(document, local) {
        let section = sketch_section(&sketch).ok();
        return Ok((
            json!({ "profile": "sketch_region", "sketch": local }),
            None,
            section,
        ));
    }
    if let Some(size) = standard::lookup(written) {
        let profile = runmat_cad::model::weldment::MemberProfile::<f64>::Standard {
            designation: size.designation.clone(),
        };
        return Ok((
            json!({ "profile": "standard", "designation": size.designation }),
            None,
            profile.section(),
        ));
    }
    // A project profile: its sketch is copied in under the profile's name.
    let file = profile_file(path, files, written);
    if let Some(source) = files.get(&file) {
        let parsed = document_of_file(&file, source)?;
        let sketch = sketch_of(&parsed, library::PROFILE_SKETCH).ok_or_else(|| {
            model(format!(
                "{file} has no sketch `{}` to use as the profile",
                library::PROFILE_SKETCH
            ))
        })?;
        let section =
            sketch_section(&sketch).map_err(|e| model(format!("profile `{written}`: {e}")))?;
        let id = match sketch_of(document, written) {
            None => written.to_owned(),
            Some(existing) if existing.entities == sketch.entities => written.to_owned(),
            Some(_) => fresh_id(document, &format!("{written}_")),
        };
        return Ok((
            json!({ "profile": "sketch_region", "sketch": id }),
            Some((id, sketch)),
            Some(section),
        ));
    }
    Err(model(format!(
        "`{written}` isn't a standard size (IPE, HEA, HEB 100–600, NPS ½–12 Sch 40), a profile in \
         library/profiles/ or a sketch of this part"
    )))
}

fn document_of_file(file: &str, source: &str) -> Result<PartDocument, EditFailure> {
    parse_part(file, source)
        .document
        .ok_or_else(|| model(format!("{file} doesn't parse")))
}

fn place_members(
    path: &str,
    text: &str,
    files: &Files,
    request: &EditRequest,
) -> Result<LibraryEdit, EditFailure> {
    let values = crate::edit::sketch::with_selection(&request.values, &request.selection);
    let document = document(path, text)?;
    let picks = Picks::new(&document, path, text);
    let written = text_value(&values, "profile");
    let written = if written.trim().is_empty() && !values.contains_key("profile") {
        "IPE 200"
    } else {
        written
    };
    let (profile, copy, section) = chosen_profile(path, files, &document, written)?;
    let groups = crate::weldment_edits::segments(&values, &picks)?;
    let corner = match text_value(&values, "corner") {
        "" => "miter",
        c => c,
    };
    let material = match text_value(&values, "material") {
        "" => "s235",
        m => m,
    };
    let rotation = if values.contains_key("rotation") {
        crate::features::value(&values, "rotation", "deg")?
    } else {
        json!({ "form": "quantity", "value": 0.0, "unit": "deg" })
    };
    let count: usize = groups.iter().map(|(_, s)| s.len()).sum();
    let groups: Vec<Value> = groups
        .into_iter()
        .map(|(sketch, segments)| json!({ "sketch": sketch, "segments": segments, "profile": profile, "material": material, "corner": corner, "rotation": rotation }))
        .collect();
    let kind: FeatureKind = serde_json::from_value(
        json!({ "feature": "weldment", "weldment": "structural_member", "groups": groups }),
    )
    .map_err(|e| model(format!("the members: {e}")))?;
    let mut added = String::new();
    let change = edit_part(text, path, |document| {
        if let Some((id, sketch)) = &copy {
            if sketch_of(document, id).is_none() {
                // The profile sketch goes before the members' path sketches, at the front.
                document
                    .features
                    .insert(0, Feature::new(id, FeatureKind::Sketch(sketch.clone())));
            }
        }
        added = fresh_id(document, "members");
        document.features.push(Feature::new(&added, kind.clone()));
        Ok(())
    })?;
    let what = written.trim();
    let mut summary = format!(
        "Added {added}: {count} member{} in {what}",
        if count == 1 { "" } else { "s" }
    );
    if let Some(section) = section {
        summary.push_str(&format!(" ({})", section.text()));
    }
    Ok(LibraryEdit {
        own: Some(change),
        created: Vec::new(),
        summary,
    })
}

/// `width = 8 mm, depth = 3 mm` as the instance's arguments.
fn arguments(written: &str) -> Result<BTreeMap<InputName, ValueExpr>, EditFailure> {
    let mut out = BTreeMap::new();
    for item in written.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (name, value) = item
            .split_once('=')
            .ok_or_else(|| model(format!("`{item}` isn't an input value; write name = value")))?;
        let lowered = lower_text(value.trim())
            .map_err(|_| model(format!("`{}` isn't a value", value.trim())))?;
        let value: ValueExpr = serde_json::from_value(lowered)
            .map_err(|_| model(format!("`{}` isn't a value", value.trim())))?;
        out.insert(InputName::new(name.trim()), value);
    }
    Ok(out)
}

fn insert(
    path: &str,
    text: &str,
    files: &Files,
    request: &EditRequest,
) -> Result<LibraryEdit, EditFailure> {
    let values = crate::edit::sketch::with_selection(&request.values, &request.selection);
    let def = text_value(&values, "definition").trim().to_owned();
    if def.is_empty() {
        return Err(model(
            "choose a library item: a feature_def or a part in library/ beside this part",
        ));
    }
    let file = library::def_file(&def)
        .ok_or_else(|| model(format!("`{def}` doesn't name a .part file in library/")))?;
    let full = resolve_path(path, file, files);
    let source = files
        .get(&full)
        .ok_or_else(|| model(format!("{file} isn't in the project")))?;
    let definition = library::definition(&def, source).map_err(model)?;
    let is_part = !def.contains('#');
    let document = document(path, text)?;
    let picks = Picks::new(&document, path, text);
    let placement = picks_of(&values, "placement").into_iter().next();
    let on: Option<PlaneRef> = match placement.clone() {
        None => None,
        Some(_) if is_part => {
            return Err(model(
                "a library part inserts at the origin; move its body afterwards with Move body",
            ))
        }
        Some(pick) => Some(
            serde_json::from_value(picks.plane(&pick)?)
                .map_err(|e| model(format!("the placement: {e}")))?,
        ),
    };
    let args = arguments(text_value(&values, "args"))?;
    if let Some(unknown) = args
        .keys()
        .find(|k| !definition.params.iter().any(|p| &p.name == *k))
    {
        let known: Vec<&str> = definition
            .params
            .iter()
            .map(|p| p.name.0.as_str())
            .collect();
        return Err(model(format!(
            "{file} has no input `{}`; its inputs are {}",
            unknown.0,
            if known.is_empty() {
                "none".to_owned()
            } else {
                known.join(", ")
            }
        )));
    }
    // The instance is named after the definition (`slot1`) or the part's file (`pin1`).
    let stem = match def.split_once('#') {
        Some((_, name)) => name.strip_prefix("feature_def.").unwrap_or(name),
        None => file
            .rsplit('/')
            .next()
            .and_then(|f| f.strip_suffix(".part"))
            .unwrap_or("part"),
    }
    .to_owned();
    let instance = LibraryInstance {
        def: def.clone(),
        args,
        bindings: BTreeMap::new(),
        mode: Default::default(),
        plane: on.clone(),
    };
    let mut added = String::new();
    let change = edit_part(text, path, |document| {
        added = fresh_id(document, &stem);
        document
            .features
            .push(Feature::new(&added, FeatureKind::Library(instance.clone())));
        Ok(())
    })?;
    let summary = if is_part {
        format!("Inserted {file} as body {added} at the origin")
    } else {
        format!(
            "Inserted {} as {added}{}",
            definition
                .name
                .rsplit("feature_def.")
                .next()
                .unwrap_or(&def),
            placement.map_or(" on origin.xy".to_owned(), |p| format!(" on {p}"))
        )
    };
    Ok(LibraryEdit {
        own: Some(change),
        created: Vec::new(),
        summary,
    })
}

#[cfg(test)]
mod tests;
