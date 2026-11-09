//! Assembly structure from the canvas: replacing components, forming and dissolving
//! subassemblies, and a new part in context.
//!
//! - `assembly.replace`: the components use another file of the same kind. Their mate
//!   connectors keep their anchors in the component's coordinates, so mates follow where the
//!   new file has geometry there; with Re-attach mates off, the components' mates and their
//!   connectors are removed.
//! - `assembly.subassembly`:
//!   - **Form new** moves the components into a new `.assembly` beside this one, at its origin,
//!     with the mates and connectors among them. Mates to components left behind stay here and
//!     reach in through the subassembly, which is then flexible so they keep solving.
//!   - **Dissolve** puts a subassembly's components back here where they sit now, with its
//!     mates and connectors; connectors here that reached into it reach the components directly.
//!   - **Make flexible** and **Make rigid** set whether the subassembly's own components join
//!     this assembly's mate solve.
//! - `assembly.new-part`: a new empty `.part` beside the assembly, placed as a component at the
//!   picked plane or face (its origin on the face's centre, its Z along the face's normal), so its
//!   first sketch on the Top plane lands on that face.
//!
//! New files come back as file changes beside the assembly's own; a name that is taken is
//! refused rather than overwritten.

use super::frames::{dot, Frame, V3};
use super::mate::quantity;
use super::{find, model, parse_pick, Compiled};
use crate::features::picks_of;
use crate::runtime::{resolve_path, Files};
use crate::xref_repairs::FileEdit;
use runmat_cad::assembly::model::{
    AssemblyDocument, DefinitionKind, DefinitionRef, FrameRef, Occurrence, OffsetExpr, PathSegment,
    Placement, RelPath, RotationExpr,
};
use runmat_cad::model::PartDocument;
use runmat_cad::syntax::edit::{EditFailure, SourceChange};
use runmat_cad::syntax::{parse_assembly_in, print_assembly, print_part};
use runmat_design_syntax::edit::TextEdit;
use runmat_types::Span;
use serde_json::{Map, Value};
use std::collections::BTreeSet;

type Values = Map<String, Value>;

fn text<'a>(values: &'a Values, id: &str) -> &'a str {
    values.get(id).and_then(Value::as_str).unwrap_or("").trim()
}

/// A new file named `name` (with extension `ext`) beside `parent`, refused when it exists.
fn new_file(parent: &str, name: &str, ext: &str, files: &Files) -> Result<String, EditFailure> {
    let name = if name.is_empty() {
        return Err(model(format!("name the new {ext} file")));
    } else {
        name
    };
    let name = if name.ends_with(ext) {
        name.to_owned()
    } else {
        format!("{name}{ext}")
    };
    if name.contains('/') {
        return Err(model(
            "the new file goes beside this assembly; give a file name without a folder",
        ));
    }
    let folder = &parent[..parent.rfind('/').map_or(0, |i| i + 1)];
    let path = format!("{folder}{name}");
    if files.contains_key(&path) || path == parent {
        return Err(model(format!(
            "`{name}` already exists; choose another name"
        )));
    }
    Ok(path)
}

fn stem(path: &str) -> String {
    let file = path.rsplit('/').next().unwrap_or(path);
    file.split('.')
        .next()
        .unwrap_or(file)
        .replace(['-', ' '], "_")
}

fn created(path: String, text: String) -> FileEdit {
    let change = SourceChange {
        edits: vec![TextEdit::replace(Span { start: 0, end: 0 }, text.clone())],
        source: text,
        reprinted: true,
    };
    (path, String::new(), change)
}

fn unused(taken: impl Fn(&str) -> bool, stem: &str) -> String {
    (1..)
        .map(|n| {
            if n == 1 {
                stem.to_owned()
            } else {
                format!("{stem}{n}")
            }
        })
        .find(|n| !taken(n))
        .expect("an unused name")
}

fn first_segment(path: &RelPath) -> Option<&str> {
    path.0
        .first()
        .filter(|s| s.instance.is_none())
        .map(|s| s.name.as_str())
}

pub(super) fn replace(
    document: &mut AssemblyDocument,
    files: &Files,
    names: &[String],
    values: &Values,
) -> Result<String, EditFailure> {
    let with = text(values, "with");
    if with.is_empty() {
        return Err(model("name the file to replace them with"));
    }
    let file = resolve_path(&document.path.0, with, files);
    if !files.contains_key(&file) {
        return Err(model(format!("there is no `{with}` in the project")));
    }
    let definition = match () {
        _ if file.ends_with(".part") => DefinitionRef::part(&file, None),
        _ if file.ends_with(".assembly") => DefinitionRef::assembly(&file),
        _ if file.ends_with(".circuit") => DefinitionRef::circuit(&file),
        _ => {
            return Err(model(
                "replace components with a .part, .assembly or .circuit file",
            ))
        }
    };
    if file == document.path.0 {
        return Err(model("an assembly can't contain itself"));
    }
    for name in names {
        let at = find(document, name)?;
        let occurrence = &mut document.occurrences[at];
        if occurrence.source.definition.kind != definition.kind {
            let kind = match occurrence.source.definition.kind {
                DefinitionKind::Part => "part",
                DefinitionKind::Assembly => "subassembly",
                _ => "circuit board",
            };
            return Err(model(format!(
                "`{name}` is a {kind}; replace it with a file of the same kind"
            )));
        }
        occurrence.source.definition = definition.clone();
        // The old file's configurations and overrides name things the new file may not have.
        occurrence.configuration = Default::default();
        occurrence.overrides.clear();
    }
    let remate = values
        .get("remate")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    if !remate {
        let on: BTreeSet<String> = document
            .connectors
            .iter()
            .filter(|c| first_segment(&c.occurrence).is_some_and(|s| names.iter().any(|n| n == s)))
            .map(|c| c.name.clone())
            .collect();
        document
            .mates
            .retain(|m| !on.contains(&m.a) && !on.contains(&m.b));
        let used: BTreeSet<String> = document
            .mates
            .iter()
            .flat_map(|m| [m.a.clone(), m.b.clone()])
            .collect();
        document
            .connectors
            .retain(|c| !on.contains(&c.name) || used.contains(&c.name));
    }
    Ok(format!(
        "Replaced {} with {}",
        names.join(", "),
        file.rsplit('/').next().unwrap_or(&file)
    ))
}

/// The connectors an occurrence's placement attaches to.
fn attached_to(placement: &Placement) -> Option<&str> {
    match placement {
        Placement::Attach {
            to: FrameRef::Connector { connector },
            ..
        } => Some(connector),
        _ => None,
    }
}

fn form(
    document: &mut AssemblyDocument,
    names: &[String],
    values: &Values,
    files: &Files,
) -> Result<(String, Vec<FileEdit>), EditFailure> {
    let path = new_file(&document.path.0, text(values, "name"), ".assembly", files)?;
    let inside = |n: &str| names.iter().any(|m| m == n);
    let host = |doc: &AssemblyDocument, connector: &str| -> Option<String> {
        doc.connector(connector)
            .and_then(|c| first_segment(&c.occurrence))
            .map(str::to_owned)
    };
    for p in &document.patterns {
        if p.seeds.iter().any(|s| inside(s)) {
            return Err(model(format!("`{}` patterns a component you picked; dissolve the pattern or pick its seeds' pattern too", p.name)));
        }
    }
    for m in &document.mirrors {
        if m.seeds.iter().any(|s| inside(s)) {
            return Err(model(format!(
                "`{}` mirrors a component you picked; remove the mirror first",
                m.name
            )));
        }
    }
    // A placement attached across the boundary would lose its frame.
    for o in &document.occurrences {
        if let Some(c) = attached_to(&o.placement) {
            let other = host(document, c);
            if other
                .as_deref()
                .is_some_and(|h| inside(h) != inside(&o.name))
                || (other.is_none() && inside(&o.name))
            {
                return Err(model(format!("`{}` is attached to `{c}` across the new subassembly's boundary; mate it instead", o.name)));
            }
        }
    }
    let mut sub = AssemblyDocument::new(&path, &stem(&path));
    sub.header = document.header.clone();
    let mut kept = Vec::new();
    for o in std::mem::take(&mut document.occurrences) {
        if inside(&o.name) {
            sub.occurrences.push(o);
        } else {
            kept.push(o);
        }
    }
    document.occurrences = kept;
    let side = |doc: &AssemblyDocument, connector: &str| host(doc, connector).map(|h| inside(&h));
    let mut cross = false;
    let mut mates = Vec::new();
    for m in std::mem::take(&mut document.mates) {
        match (side(document, &m.a), side(document, &m.b)) {
            (Some(true), Some(true)) => sub.mates.push(m),
            (Some(true), _) | (_, Some(true)) => {
                cross = true;
                mates.push(m);
            }
            _ => mates.push(m),
        }
    }
    document.mates = mates;
    let name = unused(
        |n| document.occurrences.iter().any(|o| o.name == n),
        &sub.name,
    );
    let used_here: BTreeSet<&str> = document
        .mates
        .iter()
        .flat_map(|m| [m.a.as_str(), m.b.as_str()])
        .chain(
            document
                .occurrences
                .iter()
                .filter_map(|o| attached_to(&o.placement)),
        )
        .collect();
    let used_there: BTreeSet<&str> = sub
        .mates
        .iter()
        .flat_map(|m| [m.a.as_str(), m.b.as_str()])
        .chain(
            sub.occurrences
                .iter()
                .filter_map(|o| attached_to(&o.placement)),
        )
        .collect();
    let mut connectors = Vec::new();
    for c in std::mem::take(&mut document.connectors) {
        let on_inside = first_segment(&c.occurrence).is_some_and(inside);
        if !on_inside {
            connectors.push(c);
            continue;
        }
        let (here, there) = (
            used_here.contains(c.name.as_str()),
            used_there.contains(c.name.as_str()),
        );
        if there || !here {
            sub.connectors.push(c.clone());
        }
        if here {
            let mut through = c;
            through.occurrence.0.insert(
                0,
                PathSegment {
                    name: name.clone(),
                    instance: None,
                },
            );
            connectors.push(through);
        }
    }
    document.connectors = connectors;
    let moved = sub.occurrences.len();
    let printed = print_assembly(&sub, &[]).map_err(EditFailure::Print)?;
    let mut occurrence = Occurrence::new(&name, DefinitionRef::assembly(&path));
    occurrence.flexible = cross;
    document.occurrences.push(occurrence);
    let summary = format!(
        "Formed {} from {} component{}{}",
        path.rsplit('/').next().unwrap_or(&path),
        moved,
        if moved == 1 { "" } else { "s" },
        if cross {
            " (flexible: mates reach into it)"
        } else {
            ""
        }
    );
    Ok((summary, vec![created(path, printed)]))
}

/// Row-major rotation and translation (m) of the occurrence at `path` as solved now.
fn world(compiled: &Compiled, path: &str) -> Option<([[f64; 3]; 3], V3)> {
    let row = compiled
        .rows
        .iter()
        .find(|r| r["path"].as_str() == Some(path))?;
    let rotation: [[f64; 3]; 3] = serde_json::from_value(row["world"]["rotation"].clone()).ok()?;
    let translation: V3 = serde_json::from_value(row["world"]["translation"].clone()).ok()?;
    Some((rotation, translation))
}

fn round(v: f64, places: i32) -> f64 {
    let s = 10f64.powi(places);
    let r = (v * s).round() / s;
    if r == 0.0 {
        0.0
    } else {
        r
    }
}

/// A literal placement for a rotation (row-major) and translation (m).
fn placement_at(rotation: [[f64; 3]; 3], translation: V3) -> Placement {
    let column = |k: usize| [rotation[0][k], rotation[1][k], rotation[2][k]].map(|c| round(c, 9));
    let identity = (0..3)
        .all(|i| (0..3).all(|j| (rotation[i][j] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12));
    let offset = OffsetExpr {
        translation: translation.map(|t| quantity(round(t * 1e3, 6), "mm")),
        rotation: if identity {
            RotationExpr::Identity
        } else {
            RotationExpr::Basis {
                x: column(0),
                z: column(2),
            }
        },
    };
    if identity && translation.iter().all(|t| t.abs() < 1e-12) {
        Placement::Origin
    } else {
        Placement::Offset { offset }
    }
}

fn det(r: [[f64; 3]; 3]) -> f64 {
    r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
}

fn dissolve(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    files: &Files,
    sub_name: &str,
) -> Result<String, EditFailure> {
    let at = find(document, sub_name)?;
    let occurrence = document.occurrences[at].clone();
    if occurrence.source.definition.kind != DefinitionKind::Assembly {
        return Err(model(format!("`{sub_name}` isn't a subassembly")));
    }
    if !occurrence.overrides.is_empty() {
        return Err(model(format!(
            "`{sub_name}` has overrides; remove them before dissolving it"
        )));
    }
    let path = resolve_path(
        &document.path.0,
        &occurrence.source.definition.file.0,
        files,
    );
    let text = files
        .get(&path)
        .ok_or_else(|| model(format!("`{path}` isn't in the project")))?;
    let scope = crate::runtime::module_scope(&path, text, files);
    let parsed = parse_assembly_in(&path, text, &scope);
    let Some(sub) = parsed.document else {
        return Err(EditFailure::Invalid(parsed.diagnostics));
    };
    if !sub.patterns.is_empty() || !sub.mirrors.is_empty() || !sub.fasteners.is_empty() {
        return Err(model(format!(
            "`{sub_name}` has patterns, mirrors or fastener sets; dissolve it in the source editor"
        )));
    }
    if sub.connectors.iter().any(|c| c.occurrence.0.is_empty())
        || document
            .connectors
            .iter()
            .any(|c| c.occurrence.0.len() == 1 && first_segment(&c.occurrence) == Some(sub_name))
    {
        return Err(model(format!("a connector sits on `{sub_name}`'s own frame; mate to its components before dissolving it")));
    }
    // Names that clash with this assembly's take the subassembly's name in front.
    let rename = |taken: &dyn Fn(&str) -> bool, n: &str| {
        if taken(n) {
            unused(taken, &format!("{sub_name}_{n}"))
        } else {
            n.to_owned()
        }
    };
    let mut names = std::collections::BTreeMap::new();
    let mut children = Vec::new();
    for child in &sub.occurrences {
        let taken = |n: &str| {
            (n != sub_name && document.occurrences.iter().any(|o| o.name == n))
                || names.values().any(|v: &String| v == n)
        };
        let new = rename(&taken, &child.name);
        let mut moved = child.clone();
        if !child.suppressed {
            let (rotation, translation) = world(compiled, &format!("{sub_name}/{}", child.name))
                .ok_or_else(|| {
                    model(format!(
                        "`{sub_name}/{}` isn't placed; fix the subassembly first",
                        child.name
                    ))
                })?;
            if det(rotation) < 0.0 {
                return Err(model(format!(
                    "`{}` is a mirror image; dissolve it in the source editor",
                    child.name
                )));
            }
            moved.placement = placement_at(rotation, translation);
        } else if !matches!(occurrence.placement, Placement::Origin) {
            return Err(model(format!(
                "`{}` is suppressed, so where it sits isn't known",
                child.name
            )));
        }
        moved.name = new.clone();
        names.insert(child.name.clone(), new);
        children.push(moved);
    }
    let mut connector_names = std::collections::BTreeMap::new();
    for c in &sub.connectors {
        let taken = |n: &str| {
            document.connectors.iter().any(|d| d.name == n)
                || connector_names.values().any(|v: &String| v == n)
        };
        let new = rename(&taken, &c.name);
        let mut moved = c.clone();
        moved.name = new.clone();
        if let Some(first) = moved.occurrence.0.first_mut() {
            if let Some(n) = names.get(&first.name) {
                first.name = n.clone();
            }
        }
        connector_names.insert(c.name.clone(), new);
        document.connectors.push(moved);
    }
    for c in document.connectors.iter_mut() {
        if first_segment(&c.occurrence) == Some(sub_name) && c.occurrence.0.len() > 1 {
            c.occurrence.0.remove(0);
            if let Some(n) = names.get(&c.occurrence.0[0].name) {
                c.occurrence.0[0].name = n.clone();
            }
        }
    }
    for m in &sub.mates {
        let mut moved = m.clone();
        moved.name = unused(|n| document.mates.iter().any(|d| d.name == n), &m.name);
        for end in [&mut moved.a, &mut moved.b] {
            if let Some(n) = connector_names.get(end) {
                *end = n.clone();
            }
        }
        document.mates.push(moved);
    }
    let count = children.len();
    document.occurrences.splice(at..=at, children);
    Ok(format!(
        "Dissolved {sub_name} ({count} component{})",
        if count == 1 { "" } else { "s" }
    ))
}

pub(super) fn subassembly(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    files: &Files,
    names: &[String],
    values: &Values,
) -> Result<(String, Vec<FileEdit>), EditFailure> {
    match values
        .get("action")
        .and_then(Value::as_str)
        .unwrap_or("form")
    {
        "form" => form(document, names, values, files),
        "dissolve" => {
            let mut done = Vec::new();
            for name in names {
                done.push(dissolve(document, compiled, files, name)?);
            }
            Ok((done.join("; "), Vec::new()))
        }
        action @ ("flexible" | "rigid") => {
            let flexible = action == "flexible";
            for name in names {
                let at = find(document, name)?;
                let o = &mut document.occurrences[at];
                if o.source.definition.kind != DefinitionKind::Assembly {
                    return Err(model(format!("`{name}` isn't a subassembly")));
                }
                o.flexible = flexible;
            }
            Ok((format!("Made {} {action}", names.join(", ")), Vec::new()))
        }
        other => Err(model(format!(
            "a subassembly is formed, dissolved, made flexible or made rigid, not `{other}`"
        ))),
    }
}

/// The picked plane in assembly coordinates (m): an origin plane, or a component's flat face.
fn start_frame(compiled: &Compiled, pick: &str) -> Result<Frame, EditFailure> {
    use super::frames::{perpendicular, FrameKind};
    let plane = |z: V3, x: V3| Frame {
        kind: FrameKind::Plane,
        origin: [0.0; 3],
        z,
        x,
        radius: None,
    };
    match pick {
        "origin:top" | "origin.xy" | "origin:xy" => {
            return Ok(plane([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]))
        }
        "origin:front" | "origin.zx" | "origin:zx" | "origin:xz" => {
            return Ok(plane([0.0, -1.0, 0.0], [1.0, 0.0, 0.0]))
        }
        "origin:right" | "origin.yz" | "origin:yz" => {
            return Ok(plane([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
        }
        _ => {}
    }
    let parsed = parse_pick(pick).filter(|p| matches!(p.entity, Some(super::Entity::Face { .. })));
    let Some(parsed) = parsed else {
        return Err(model("start the part on an origin plane or a flat face"));
    };
    let (prototype, transform) = compiled
        .placed(&parsed.uuid)
        .ok_or_else(|| model("the picked face's component isn't drawn"))?;
    let Some(super::Entity::Face { body, face }) = parsed.entity else {
        unreachable!()
    };
    let frame = compiled
        .body(prototype, body)
        .and_then(|b| {
            b["faces"]
                .as_array()?
                .iter()
                .find(|f| f["index"].as_u64() == Some(face as u64))
        })
        .and_then(super::frames::face_frame)
        .map(|f| super::frames::to_world(f, &transform))
        .filter(|f| f.kind == FrameKind::Plane)
        .ok_or_else(|| model("start the part on a flat face"))?;
    // Keep the part's X along the assembly's X where the face allows it.
    let x = if dot(frame.z, [1.0, 0.0, 0.0]).abs() < 0.999 {
        super::frames::unit(super::frames::sub(
            [1.0, 0.0, 0.0],
            super::frames::scale(frame.z, frame.z[0]),
        ))
    } else {
        perpendicular(frame.z)
    };
    Ok(Frame { x, ..frame })
}

pub(super) fn new_part(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    files: &Files,
    values: &Values,
) -> Result<(String, Vec<FileEdit>), EditFailure> {
    let path = new_file(&document.path.0, text(values, "name"), ".part", files)?;
    let frame = match picks_of(values, "plane").into_iter().next() {
        Some(pick) => Some(start_frame(compiled, &pick)?),
        None => None,
    };
    let part = PartDocument::new(&path);
    let printed = print_part(&part, &[]).map_err(EditFailure::Print)?;
    let name = unused(
        |n| document.occurrences.iter().any(|o| o.name == n),
        &stem(&path),
    );
    let mut occurrence = Occurrence::new(&name, DefinitionRef::part(&path, None));
    if let Some(f) = frame {
        let y = super::frames::cross(f.z, f.x);
        let rotation = [
            [f.x[0], y[0], f.z[0]],
            [f.x[1], y[1], f.z[1]],
            [f.x[2], y[2], f.z[2]],
        ];
        occurrence.placement = placement_at(rotation, f.origin);
    }
    document.occurrences.push(occurrence);
    Ok((
        format!(
            "Created {} in place as {name}",
            path.rsplit('/').next().unwrap_or(&path)
        ),
        vec![created(path, printed)],
    ))
}
