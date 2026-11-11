//! `design.seat_part`: seats a component against another in one call. The tool writes the
//! connectors and the mate into the assembly, solves it, and answers with the staged change,
//! the pose the solve gives the component, its interference with every other component (only
//! the pairs it is in, as `design.inspect`'s `changed` check), and its clearance to named
//! neighbours.
//!
//! The contact comes from the picks:
//! - On the target: a named face (a plane, or a cylinder's axis), a point and a direction (the
//!   ray from the point meets the target's surface; a curved shell is seated on this way), or
//!   one of the assembly's connectors.
//! - On the component: a named face, a side of its box (`+z` is its top; the contact is the
//!   middle of the material lying on that side), or a frame its part publishes. Without one,
//!   the box side that faces the target is used.
//!
//! On a plane or a ray's surface the component turns face to face with the target (`flip`
//! seats it the other way round), its long axis lies along `along` (or where it points now),
//! turned by `rotation` about the contact normal. The long axis's positive direction is read in
//! the part's coordinates, whatever sign the kernel gives a bore's axis, so rotation 0 points it
//! along `along` and 180 points it the other way. It then travels toward the target until it
//! first touches it, by rays both ways between the two parts' tessellations, and stops
//! `offset` short of it. On an axis it lines up with the target's axis, `offset` along it.
//! With `levelWith` it then moves square to that component's round axis onto the axis, so a
//! set of clamps under a curved belly holds one straight tube; where the axis would press it
//! into the target it stays at first contact and the notes say how far off the axis it is.
//!
//! The mate is a fasten unless `mate` asks for `coincident` (a planar mate) or `concentric` (a
//! cylindrical mate). A fasten moves the component from where it is: the solver seeds a fasten
//! from the connector frames in closed form. The other mates leave freedoms, so the component's
//! starting placement is written at the seated pose. Seating the same component again replaces
//! its seat.
//!
//! Two picks build symmetric, aligned structures (a landing gear) by construction:
//! - `through` seats a straight member through two or more holders' bores at once, on their
//!   common axis and centred between them, and refuses when the bores aren't on one axis
//!   (seat/through.rs);
//! - `mirror` seats a component as the mirror image of one already seated, across the target's
//!   plane of symmetry (seat/mirror.rs).
//!
//! With `contact` the result also carries `design.inspect`'s stance section for the assembly
//! after the seating: the ground clearance, the centre of mass's margin and tip-over angle, and
//! the stance.

#[path = "seat/geometry.rs"]
mod geometry;
#[path = "seat/mirror.rs"]
mod mirror;
#[path = "seat/text.rs"]
mod text;
#[path = "seat/through.rs"]
mod through;
#[path = "seat/types.rs"]
mod types;

use crate::edit::stage;
use crate::{text_of, Files, ToolError};
use geometry::*;
use runmat_cad::display::PartDisplay;
use runmat_cad::regen::kernel::Transform;
use serde_json::Value;
use std::rc::Rc;
pub use types::*;

/// Neighbours within this distance are measured (m).
const MEASURED: f64 = 0.025;

fn round(v: f64, places: i32) -> f64 {
    let k = 10f64.powi(places);
    let r = (v * k).round() / k;
    if r == 0.0 {
        0.0
    } else {
        r
    }
}

fn mm3(p: V3) -> [f64; 3] {
    p.map(|v| round(v * 1000.0, 3))
}

fn vector(v: &Option<Vec<f64>>, what: &str) -> Result<Option<V3>, ToolError> {
    match v {
        None => Ok(None),
        Some(v) if v.len() == 3 && v.iter().all(|c| c.is_finite()) => Ok(Some([v[0], v[1], v[2]])),
        Some(v) => Err(ToolError::invalid(format!(
            "{what} takes three numbers, not {v:?}"
        ))),
    }
}

/// The pose the seating computed: `p ↦ rotation p + translation` (m).
#[derive(Clone, Copy)]
struct Seated {
    rotation: M3,
    translation: V3,
}

impl Seated {
    fn apply(&self, p: V3) -> V3 {
        add(geometry::apply(&self.rotation, p), self.translation)
    }
}

/// A connector frame in its part's coordinates: origin, Z and X (m).
struct Frame {
    at: V3,
    z: V3,
    x: V3,
}

struct Placed {
    world: Transform,
    display: Rc<PartDisplay>,
}

/// The part occurrences a seating works on, from one resolution of the assembly.
struct PlacedSet<'a> {
    path: &'a str,
    text: &'a str,
    files: &'a Files,
    found: Vec<runmat_cad_wasm::probe::PlacedPart>,
}

impl<'a> PlacedSet<'a> {
    fn new(
        path: &'a str,
        text: &'a str,
        files: &'a Files,
        names: &[&str],
    ) -> Result<Self, ToolError> {
        let names: Vec<String> = names.iter().map(|n| (*n).to_owned()).collect();
        let found = runmat_cad_wasm::probe::placed_parts(path, text, files, &names)
            .map_err(|e| ToolError::new("not_compiled", format!("{path} doesn't resolve: {e}")))?;
        Ok(Self {
            path,
            text,
            files,
            found,
        })
    }

    fn get(&self, name: &str, role: &str) -> Result<Placed, ToolError> {
        match self.found.iter().find(|p| p.occurrence == name) {
            Some(p) => Ok(Placed {
                world: p.world,
                display: p.display.clone(),
            }),
            None => Err(ToolError::invalid(format!(
                "{role} `{name}` isn't a part occurrence of {}; its part occurrences are {}",
                self.path,
                occurrence_names(self.path, self.text, self.files).join(", ")
            ))),
        }
    }
}

fn occurrence_names(path: &str, text: &str, files: &Files) -> Vec<String> {
    let Ok(resolution) = runmat_cad_wasm::runtime::resolve_assembly(path, text, files) else {
        return Vec::new();
    };
    let r = &resolution.resolved;
    (0..r.nodes.len())
        .filter(|&i| r.nodes[i].parent.is_some())
        .map(|i| r.path(i as u32))
        .take(60)
        .collect()
}

fn refuse_face(role: &str, component: &str, face: &str, display: &PartDisplay) -> ToolError {
    ToolError::invalid(format!(
        "{role} face `{face}` isn't a face of `{component}`; its faces include {}",
        face_names(display, 16).join(", ")
    ))
}

pub fn seat(input: &SeatInput, files: &Files) -> Result<SeatOutput, ToolError> {
    let path = input.path.as_str();
    if !path.ends_with(".assembly") {
        return Err(ToolError::invalid(format!(
            "{path} isn't an assembly; design.seat_part seats a component in a .assembly"
        )));
    }
    let original = text_of(files, path)?;
    let name = input.component.trim();
    let holes = match &input.through {
        Some(entries) => {
            if input.target.is_some() || input.mirror.is_some() || input.level_with.is_some() {
                return Err(ToolError::invalid("through fastens the member to the first holder on the bores' axis; leave out target, mirror and levelWith"));
            }
            Some(through::holes(entries, name)?)
        }
        None => None,
    };
    let target_in = input.target.as_ref();
    let target_name = match (&holes, target_in) {
        (Some(h), _) => h[0].holder.as_str(),
        (None, Some(t)) => t.component.trim(),
        (None, None) => return Err(ToolError::invalid("give target (the component it goes on and where), or through (the holders whose bores it passes through)")),
    };
    let mirror_of = input
        .mirror
        .as_deref()
        .map(str::trim)
        .filter(|m| !m.is_empty());
    if !text::is_name(name) {
        return Err(ToolError::invalid(format!("component `{name}` isn't an occurrence name (letters, digits and _); seat a top-level occurrence")));
    }
    if name == target_name {
        return Err(ToolError::invalid(format!(
            "component `{name}` can't be seated on itself; name another target component"
        )));
    }
    let kind = match input
        .mate
        .as_deref()
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .unwrap_or("fasten")
    {
        "fasten" => "fasten",
        "coincident" => "planar",
        "concentric" => "cylindrical",
        other => {
            return Err(ToolError::invalid(format!(
            "mate `{other}` isn't one design.seat_part writes: fasten, coincident or concentric"
        )))
        }
    };
    let along = vector(&input.along, "along")?;
    let flip = input.flip.unwrap_or(false);
    let offset = input.offset.unwrap_or(0.0) / 1000.0;
    let turn_by = input.rotation.unwrap_or(0.0).to_radians();
    let own = input.own.clone().unwrap_or_default();
    let level_with = input
        .level_with
        .as_deref()
        .map(str::trim)
        .filter(|l| !l.is_empty());
    if level_with == Some(name) {
        return Err(ToolError::invalid(format!(
            "component `{name}` can't be levelled with itself; name a component seated earlier"
        )));
    }

    // The assembly with the component in it and free to move, and its old seat taken out.
    let mut tentative = text::without_seat(original, name);
    tentative = match &input.part {
        Some(part) => {
            if text::declares_occurrence(&tentative, name) {
                return Err(ToolError::invalid(format!("component `{name}` is already in {path}; leave out `part` to seat it, or name the new occurrence differently")));
            }
            let part_text = text_of(files, part).map_err(|_| ToolError::new("file_missing", format!("part `{part}` isn't in the workspace; create it first (design.propose_edit), then seat it")))?;
            let symbol = text::part_symbol(part_text).ok_or_else(|| ToolError::invalid(format!("part `{part}` declares no `part <name> {{`")))?;
            let values: Vec<(String, String)> = input.inputs.iter().flatten().map(|v| (v.name.clone(), v.value.clone())).collect();
            text::insert_occurrence(&tentative, path, part, &symbol, name, &values)
        }
        None => text::free_occurrence(&tentative, name).ok_or_else(|| {
            ToolError::invalid(format!("component `{name}` isn't a top-level occurrence of {path}; give `part` to insert a new one"))
        })?,
    };
    let mut wanted = vec![name, target_name];
    wanted.extend(level_with);
    wanted.extend(mirror_of);
    wanted.extend(holes.iter().flatten().map(|h| h.holder.as_str()));
    let set = PlacedSet::new(path, &tentative, files, &wanted)?;
    let moving = set.get(name, "component")?;

    // The connectors: exact frames when either side names a connector, else from the geometry.
    let target_connector = target_in
        .and_then(|t| t.connector.as_deref())
        .map(str::trim)
        .filter(|c| !c.is_empty());
    let own_connector = own
        .connector
        .as_deref()
        .map(str::trim)
        .filter(|c| !c.is_empty());
    let mut notes = Vec::new();
    let mut through_rows = Vec::new();
    let (target_line, own_line, seated, contact) = if let Some(holes) = &holes {
        if kind == "planar" {
            return Err(ToolError::invalid(
                "through puts the member on the bores' axis: mate fasten or concentric",
            ));
        }
        let t = through::through(
            &set, &moving, name, holes, &own, flip, turn_by, along, offset, &mut notes,
        )?;
        through_rows = t.rows;
        let contact = ContactReport {
            point_mm: mm3(t.center),
            normal: t.axis.map(|v| round(v, 4)),
        };
        (
            text::TargetLine::New(frame_text(&t.target_frame)),
            text::own_explicit(name, &t.own_frame, false),
            Some(t.seated),
            Some(contact),
        )
    } else if let Some(of) = mirror_of {
        let t = target_in.filter(|t| t.face.is_none() && t.from.is_none() && t.connector.is_none());
        if t.is_none() || kind != "fasten" || level_with.is_some() {
            return Err(ToolError::invalid("mirror fastens the mirror image to target.component; give target.component only, and leave out mate and levelWith"));
        }
        let target = set.get(target_name, "target component")?;
        let normal = vector(&input.mirror_normal, "mirrorNormal")?;
        let m = mirror::mirror(&set, &moving, &target, name, of, normal, &mut notes)?;
        (
            text::TargetLine::New(frame_text(&m.target_frame)),
            text::own_explicit(name, &m.own_frame, false),
            Some(m.seated),
            None,
        )
    } else if target_connector.is_some() || own_connector.is_some() {
        if kind != "fasten" {
            return Err(ToolError::invalid(format!(
                "a connector pick seats with a fasten; mate `{}` needs a face, side or ray on both",
                input.mate.as_deref().unwrap_or("")
            )));
        }
        if level_with.is_some() {
            return Err(ToolError::invalid("levelWith lines the component up from a target face or ray; leave out the connectors"));
        }
        let own_line = match own_connector {
            Some(c) => text::own_published(name, c, !flip),
            None => {
                let (frame, is_axis) = own_alone(&moving, &own, name)?;
                text::own_explicit(name, &frame, is_axis == flip)
            }
        };
        let target_line = match target_connector {
            Some(c) if text::declares_connector(&tentative, c) => {
                text::TargetLine::Existing(c.to_owned())
            }
            Some(c) => {
                return Err(ToolError::invalid(format!(
                    "target connector `{c}` isn't a connector of {path}"
                )))
            }
            None => {
                let target = set.get(target_name, "target component")?;
                let (contact, _) = target_contact(target_in.expect("a target"), &target)?;
                let (point, z) = match contact {
                    Contact::Plane { origin, normal } => (origin, normal),
                    Contact::Axis { origin, axis } => (add(origin, scale(axis, offset)), axis),
                };
                let x = turn(
                    along
                        .and_then(|a| square_to(a, z))
                        .unwrap_or_else(|| perpendicular(z)),
                    z,
                    turn_by,
                );
                let inv = target.world.inverse();
                text::TargetLine::New(frame_text(&Frame {
                    at: inv.apply(point),
                    z: inv.apply_vector(z),
                    x: inv.apply_vector(x),
                }))
            }
        };
        notes.push("placed by the connector frames: the fasten puts the frames together, with no search for first contact".to_owned());
        (target_line, own_line, None, None)
    } else {
        let target = set.get(target_name, "target component")?;
        let level = match level_with {
            Some(l) => Some(set.get(l, "levelWith component")?),
            None => None,
        };
        let picks = Picks {
            own: &own,
            flip,
            turn_by,
            along,
            offset,
            level: level.as_ref().zip(level_with),
        };
        let t = target_in.expect("a target");
        let (target_frame, own_frame, seated, is_axis) =
            seat_frames(input, t, &target, &moving, &picks, &mut notes)?;
        match (kind, is_axis) {
            ("planar", true) => return Err(ToolError::invalid(format!("mate coincident joins planes; target face `{}` on `{target_name}` is a cylinder: use concentric", t.face.as_deref().unwrap_or_default()))),
            ("cylindrical", false) => return Err(ToolError::invalid(format!("mate concentric joins axes; pick cylindrical faces for target.face on `{target_name}` and own.face on `{name}`"))),
            _ => {}
        }
        let contact = ContactReport {
            point_mm: mm3(seated.apply(own_frame.at)),
            normal: target
                .world
                .apply_vector(target_frame.z)
                .map(|v| round(v, 4)),
        };
        (
            text::TargetLine::New(frame_text(&target_frame)),
            text::own_explicit(name, &own_frame, !is_axis),
            Some(seated),
            Some(contact),
        )
    };

    let mut after = text::add_seat(&tentative, name, target_name, kind, &target_line, &own_line);
    if kind != "fasten" {
        let Some(s) = seated else {
            unreachable!("non-fasten mates come from geometry")
        };
        let (axis, angle) = axis_angle(&s.rotation);
        after = text::set_placement(
            &after,
            name,
            mm3(s.translation),
            axis.map(|v| round(v, 6)),
            round(angle.to_degrees(), 4),
        )
        .ok_or_else(|| {
            ToolError::invalid(format!(
                "component `{name}`'s placement couldn't be written"
            ))
        })?;
    }

    // The solve: the component's pose and the mate's outcome.
    let mut after_files = files.clone();
    after_files.insert(path.to_owned(), after.clone());
    let resolution = runmat_cad_wasm::runtime::resolve_assembly(path, &after, &after_files)
        .map_err(|d| {
            ToolError::new(
                "not_compiled",
                format!(
                    "the seated assembly doesn't parse: {}",
                    d.first().map(|d| d.message.as_str()).unwrap_or("")
                ),
            )
        })?;
    let r = &resolution.resolved;
    let node = (0..r.nodes.len())
        .find(|&i| r.path(i as u32) == name)
        .map(|i| &r.nodes[i]);
    let mate_name = text::mate_name(name);
    // The solver reports a mate's status as `{ "status": word, ..details }`: the word goes in
    // `mate_status` and the details (the mates it clashes with, why it failed) in the notes.
    let (mate_status, mate_detail) = match r
        .mates
        .iter()
        .find(|m| m.name == mate_name)
        .map(|m| serde_json::to_value(&m.status))
    {
        Some(Ok(Value::Object(mut o))) => {
            let word = o
                .remove("status")
                .and_then(|v| v.as_str().map(str::to_owned))
                .unwrap_or_else(|| "not solved".to_owned());
            (
                word,
                if o.is_empty() {
                    None
                } else {
                    Some(Value::Object(o).to_string())
                },
            )
        }
        _ => ("not solved".to_owned(), None),
    };
    let pose = match node {
        Some(n) => {
            let (axis, angle) = axis_angle(&n.world.rotation);
            if let Some(s) = seated {
                let off = norm(sub(n.world.translation, s.translation));
                if off > 1e-5 {
                    notes.push(format!("the solve placed it {:.3} mm from the seated pose; check the mate's status and the component's other mates", off * 1000.0));
                }
            }
            Pose {
                translation_mm: mm3(n.world.translation),
                rotation_axis: axis.map(|v| round(v, 4)),
                rotation_deg: round(angle.to_degrees(), 3),
                status: serde_json::to_value(n.pose).unwrap_or(Value::Null),
                dof: n.dof,
            }
        }
        None => {
            return Err(ToolError::new(
                "not_compiled",
                format!("component `{name}` didn't resolve in the seated assembly"),
            ))
        }
    };
    for p in resolution
        .problems
        .iter()
        .filter(|p| p.message.contains(name))
    {
        notes.push(p.message.clone());
    }

    // Interference and clearance: only the component's pairs.
    let changed = [name.to_owned()];
    let found = runmat_cad_wasm::evaluate::assembly_clearance_of(
        path,
        &after,
        &after_files,
        None,
        MEASURED,
        Some(&changed),
    )
    .map_err(|e| {
        ToolError::new(
            "not_compiled",
            format!("the interference check failed: {e}"),
        )
    })?;
    let other = |a: &str, b: &str| {
        if a == name || a.starts_with(&format!("{name}/")) {
            b.to_owned()
        } else {
            a.to_owned()
        }
    };
    let interference: Vec<OverlapRow> = found
        .overlaps
        .iter()
        .map(|o| OverlapRow {
            with: other(&o.a, &o.b),
            volume_mm3: round(o.volume * 1e9, 1),
        })
        .collect();
    let mut near: Vec<GapRow> = found
        .near
        .iter()
        .map(|n| GapRow {
            with: other(&n.a, &n.b),
            distance_mm: Some(round(n.distance * 1000.0, 2)),
            beyond_mm: None,
            not_measured: false,
        })
        .collect();
    // Pairs the kernel couldn't measure: its errors read `<a> / <b>: <why>`.
    let mine = |p: &str| p == name || p.starts_with(&format!("{name}/"));
    let mut unmeasured: Vec<String> = Vec::new();
    for e in &found.errors {
        if let Some((a, b)) = e
            .split_once(": ")
            .and_then(|(pair, _)| pair.split_once(" / "))
        {
            let with = if mine(a) {
                b
            } else if mine(b) {
                a
            } else {
                continue;
            };
            if !unmeasured.iter().any(|u| u == with) {
                unmeasured.push(with.to_owned());
            }
        }
    }
    let unmeasured_with = |n: &str| {
        unmeasured
            .iter()
            .any(|u| u == n || u.starts_with(&format!("{n}/")))
    };
    near.sort_by(|a, b| {
        a.distance_mm
            .unwrap_or(0.0)
            .total_cmp(&b.distance_mm.unwrap_or(0.0))
    });
    let clearance: Vec<GapRow> = match &input.clearance {
        Some(names) if !names.is_empty() => names
            .iter()
            .map(|n| {
                near.iter()
                    .find(|g| g.with == *n || g.with.starts_with(&format!("{n}/")))
                    .cloned()
                    .unwrap_or_else(|| {
                        if unmeasured_with(n) {
                            GapRow {
                                with: n.clone(),
                                distance_mm: None,
                                beyond_mm: None,
                                not_measured: true,
                            }
                        } else {
                            GapRow {
                                with: n.clone(),
                                distance_mm: None,
                                beyond_mm: Some(MEASURED * 1000.0),
                                not_measured: false,
                            }
                        }
                    })
            })
            .collect(),
        _ => near
            .into_iter()
            .filter(|g| g.with != target_name)
            .take(3)
            .collect(),
    };
    notes.extend(found.errors.iter().cloned());

    if let Some(detail) = mate_detail {
        notes.push(format!("{mate_name} is {mate_status}: {detail}"));
    }
    let overlap_text = match (interference.is_empty(), unmeasured.is_empty()) {
        (true, true) => "no interference".to_owned(),
        (true, false) => format!(
            "interference with {} not measured (see notes)",
            unmeasured.join(", ")
        ),
        (false, _) => {
            let overlaps = interference
                .iter()
                .map(|o| format!("{} by {} mm³", o.with, o.volume_mm3))
                .collect::<Vec<_>>()
                .join(", ");
            if unmeasured.is_empty() {
                format!("overlaps {overlaps}")
            } else {
                format!(
                    "overlaps {overlaps}; not measured against {}",
                    unmeasured.join(", ")
                )
            }
        }
    };
    let gap_text = clearance
        .iter()
        .find_map(|g| g.distance_mm.map(|d| format!("; {} mm from {}", d, g.with)))
        .unwrap_or_default();
    let [x, y, z] = pose.translation_mm;
    let stance = match input.contact.as_deref().filter(|c| !c.is_empty()) {
        Some(contact) => Some(crate::inspect::stance::stance(
            path,
            &after,
            &after_files,
            Some(contact),
            None,
        )?),
        None => None,
    };
    let stance_text = match &stance {
        Some(s) if s["ok"] == true => {
            format!(" Stance ok: {}", s["summary"].as_str().unwrap_or_default())
        }
        Some(s) => format!(
            " Stance NOT ok, fix its findings: {}",
            s["summary"].as_str().unwrap_or_default()
        ),
        None => String::new(),
    };
    let summary = format!("Seated {name} on {target_name} ({kind}, {mate_status}) at [{x}, {y}, {z}] mm: {overlap_text}{gap_text}.{stance_text}");
    let mut change = stage(summary.clone(), vec![(path.to_owned(), after)], files, path);
    change.notes.extend(notes.iter().cloned());
    Ok(SeatOutput {
        change,
        seat: SeatReport {
            summary,
            component: name.to_owned(),
            target: target_name.to_owned(),
            mate: mate_name,
            mate_kind: kind.to_owned(),
            mate_status,
            pose,
            contact,
            interference,
            clearance,
            through: through_rows,
            stance,
            notes,
        },
    })
}

/// The target's contact in assembly coordinates (a named face's plane or axis, or where the ray
/// meets its surface) and the way the component approaches it (none along an axis).
fn target_contact(t: &SeatTarget, target: &Placed) -> Result<(Contact, Option<V3>), ToolError> {
    let tname = t.component.trim();
    let world = &target.world;
    match (&t.face, vector(&t.from, "target.from")?, vector(&t.direction, "target.direction")?) {
        (Some(face), None, _) => {
            let f = find_face(&target.display, face).ok_or_else(|| refuse_face("target", tname, face, &target.display))?;
            match face_contact(f).ok_or_else(|| ToolError::invalid(format!("target face `{face}` on `{tname}` is neither flat nor a cylinder")))? {
                Contact::Plane { origin, normal } => {
                    let n = unit(world.apply_vector(normal)).unwrap_or(normal);
                    Ok((Contact::Plane { origin: world.apply(origin), normal: n }, Some(scale(n, -1.0))))
                }
                Contact::Axis { origin, axis } => Ok((Contact::Axis { origin: world.apply(origin), axis: unit(world.apply_vector(axis)).unwrap_or(axis) }, None)),
            }
        }
        (None, Some(from), Some(direction)) => {
            let a = unit(direction).ok_or_else(|| ToolError::invalid("target.direction has no length"))?;
            let fixed = mesh(&target.display, |p| world.apply(p));
            let from_m = scale(from, 0.001);
            let probe = Mesh { points: vec![from_m], triangles: Vec::new() };
            let d = travel(&probe, &fixed, a).ok_or_else(|| {
                ToolError::invalid(format!("the ray from target.from {from:?} along {direction:?} doesn't meet `{tname}`; start it outside the body and point it at it"))
            })?;
            Ok((Contact::Plane { origin: add(from_m, scale(a, d)), normal: scale(a, -1.0) }, Some(a)))
        }
        (None, Some(_), None) => Err(ToolError::invalid(format!("target.from on `{tname}` needs a direction: the way the component approaches it"))),
        (Some(face), Some(_), _) => Err(ToolError::invalid(format!("target.face `{face}` came with target.from; give one of them for `{tname}`"))),
        _ => Err(ToolError::invalid(format!("pick where on `{tname}` the component goes: target.face, target.connector, or target.from with target.direction"))),
    }
}

/// The component's own frame without a target (it fastens to a named connector), and whether
/// it is an axis.
fn own_alone(moving: &Placed, own: &SeatOwn, name: &str) -> Result<(Frame, bool), ToolError> {
    let c = match (&own.face, &own.side) {
        (Some(face), _) => {
            let f = find_face(&moving.display, face)
                .ok_or_else(|| refuse_face("own", name, face, &moving.display))?;
            face_contact(f).ok_or_else(|| {
                ToolError::invalid(format!(
                    "own face `{face}` on `{name}` is neither flat nor a cylinder"
                ))
            })?
        }
        (None, side) => {
            let side = side.as_deref().unwrap_or("+z");
            side_contact(&moving.display, side).ok_or_else(|| ToolError::invalid(format!("own side `{side}` isn't one of +x, -x, +y, -y, +z, -z (or `{name}` has no body)")))?
        }
    };
    Ok(match c {
        Contact::Plane { origin, normal } => (
            Frame {
                at: origin,
                z: normal,
                x: long_axis(&moving.display, normal),
            },
            false,
        ),
        Contact::Axis { origin, axis } => (
            Frame {
                at: origin,
                z: axis,
                x: perpendicular(axis),
            },
            true,
        ),
    })
}

/// The component's picks besides the target.
struct Picks<'a> {
    own: &'a SeatOwn,
    flip: bool,
    turn_by: f64,
    along: Option<V3>,
    offset: f64,
    /// The component whose round axis this one goes on, and its name.
    level: Option<(&'a Placed, &'a str)>,
}

/// The world axis of `placed`'s largest round face across `normal` (world): a point and its
/// unit direction.
fn world_round_axis(placed: &Placed, normal: V3) -> Option<(V3, V3)> {
    let inv = placed.world.inverse();
    let (p, a) = round_axis(&placed.display, inv.apply_vector(normal))?;
    Some((placed.world.apply(p), unit(placed.world.apply_vector(a))?))
}

/// The target's connector frame, the component's, the seated pose, and whether they meet on an
/// axis.
fn seat_frames(
    input: &SeatInput,
    t: &SeatTarget,
    target: &Placed,
    moving: &Placed,
    picks: &Picks,
    notes: &mut Vec<String>,
) -> Result<(Frame, Frame, Seated, bool), ToolError> {
    let Picks {
        own,
        flip,
        turn_by,
        along,
        offset,
        level,
    } = *picks;
    let tname = t.component.trim();
    let name = input.component.trim();
    let world = &target.world;
    let (contact, approach) = target_contact(t, target)?;
    if let (Contact::Axis { .. }, Some((_, l))) = (contact, level) {
        return Err(ToolError::invalid(format!("levelWith `{l}` lines the component up from a flat target face or a ray; on a cylinder it already shares the target's axis")));
    }
    // The component's own contact, in its part's coordinates.
    let own_contact = match (&own.face, &own.side) {
        (Some(face), _) => {
            let f = find_face(&moving.display, face)
                .ok_or_else(|| refuse_face("own", name, face, &moving.display))?;
            face_contact(f).ok_or_else(|| {
                ToolError::invalid(format!(
                    "own face `{face}` on `{name}` is neither flat nor a cylinder"
                ))
            })?
        }
        (None, Some(side)) => side_contact(&moving.display, side).ok_or_else(|| {
            ToolError::invalid(format!(
                "own side `{side}` isn't one of +x, -x, +y, -y, +z, -z (or `{name}` has no body)"
            ))
        })?,
        (None, None) => {
            let toward = match contact {
                Contact::Plane { normal, .. } => scale(normal, -1.0),
                Contact::Axis { axis, .. } => axis,
            };
            side_contact(&moving.display, facing_side(&moving.world.rotation, toward)).ok_or_else(
                || ToolError::invalid(format!("component `{name}` has no body to seat")),
            )?
        }
    };
    let current = moving.world.rotation;
    match (contact, own_contact) {
        (Contact::Plane { origin, normal }, Contact::Plane { origin: own_at, normal: own_n }) => {
            let own_x = long_axis(&moving.display, own_n);
            let to_z = if flip { normal } else { scale(normal, -1.0) };
            // The axis to level on, and the component's own bore, before it turns.
            let level = match level {
                Some((l, lname)) => {
                    let line = world_round_axis(l, normal).ok_or_else(|| ToolError::invalid(format!("levelWith `{lname}` has no round face lying across the contact (a bore or a tube's outside) to line up with")))?;
                    let bore = round_axis(&moving.display, own_n).ok_or_else(|| ToolError::invalid(format!("component `{name}` has no round face lying across its contact (a bore) to put on `{lname}`'s axis")))?;
                    Some((line, bore, lname))
                }
                None => None,
            };
            let lean = along.or(level.map(|((_, v), _, _)| v)).unwrap_or_else(|| geometry::apply(&current, own_x));
            let to_x = turn(square_to(lean, normal).unwrap_or_else(|| perpendicular(normal)), normal, turn_by);
            let rotation = frame_rotation(own_x, own_n, to_x, to_z);
            // Face to face at the contact point, then pulled back and moved in until it touches.
            let start = sub(origin, geometry::apply(&rotation, own_at));
            let a = approach.unwrap_or(scale(normal, -1.0));
            let own_mesh = mesh(&moving.display, |p| geometry::apply(&rotation, p));
            let fixed = mesh(&target.display, |p| world.apply(p));
            let reach = bounds(&moving.display).map_or(0.0, |(l, h)| norm(sub(h, l))) + bounds(&target.display).map_or(0.0, |(l, h)| norm(sub(h, l)));
            let back = sub(start, scale(a, reach));
            let pulled = Mesh {
                points: own_mesh.points.iter().map(|p| add(*p, back)).collect(),
                triangles: own_mesh.triangles.iter().map(|t| t.map(|p| add(p, back))).collect(),
            };
            let d = travel(&pulled, &fixed, a).ok_or_else(|| {
                ToolError::invalid(format!("`{name}` never meets `{tname}` moving along {:?} from the picked point; pick a point under the target", a.map(|v| round(v, 3))))
            })?;
            let mut translation = add(back, scale(a, d - offset));
            if let Some(((at, v), (bore_at, bore_axis), lname)) = level {
                let u = geometry::apply(&rotation, bore_axis);
                if dot(u, v).abs() < 0.999 {
                    return Err(ToolError::invalid(format!(
                        "`{name}`'s bore would run along {:?}, `{lname}`'s axis along {:?}; leave out `along` or give it `{lname}`'s direction",
                        u.map(|c| round(c, 3)),
                        v.map(|c| round(c, 3))
                    )));
                }
                // Square to the axis, from the bore's line onto it.
                let p = add(geometry::apply(&rotation, bore_at), translation);
                let gap = sub(at, p);
                let shift = sub(gap, scale(v, dot(gap, v)));
                // How far the shift moves it on toward the target past first contact.
                let press = dot(shift, a) - offset;
                if press > 1e-6 {
                    notes.push(format!(
                        "on `{lname}`'s axis `{name}` would press {:.2} mm into `{tname}`, so it is seated at first contact, {:.2} mm off that axis; seat `{lname}` again with levelWith `{name}` to line both up here",
                        press * 1000.0,
                        norm(shift) * 1000.0
                    ));
                } else {
                    translation = add(translation, shift);
                    notes.push(format!("on `{lname}`'s axis: moved {:.2} mm from first contact, about {:.2} mm short of `{tname}`", norm(shift) * 1000.0, -press * 1000.0));
                }
            }
            let seated = Seated { rotation, translation };
            // The component's frame (flipped when written) and the target's, where they meet.
            let own_frame = Frame { at: own_at, z: own_n, x: own_x };
            let point = seated.apply(own_at);
            let inv = world.inverse();
            let target_frame = Frame { at: inv.apply(point), z: inv.apply_vector(scale(to_z, -1.0)), x: inv.apply_vector(to_x) };
            Ok((target_frame, own_frame, seated, false))
        }
        (Contact::Axis { origin, axis }, Contact::Axis { origin: own_at, axis: own_axis }) => {
            let now = geometry::apply(&current, own_axis);
            let same = (dot(now, axis) >= 0.0) != flip;
            let to_z = if same { axis } else { scale(axis, -1.0) };
            let own_x = perpendicular(own_axis);
            let lean = along.unwrap_or_else(|| geometry::apply(&current, own_x));
            let to_x = turn(square_to(lean, to_z).unwrap_or_else(|| perpendicular(to_z)), to_z, turn_by);
            let rotation = frame_rotation(own_x, own_axis, to_x, to_z);
            let point = add(origin, scale(axis, offset));
            let translation = sub(point, geometry::apply(&rotation, own_at));
            let inv = world.inverse();
            let own_frame = Frame { at: own_at, z: own_axis, x: own_x };
            let target_frame = Frame { at: inv.apply(point), z: inv.apply_vector(to_z), x: inv.apply_vector(to_x) };
            Ok((target_frame, own_frame, Seated { rotation, translation }, true))
        }
        (Contact::Axis { .. }, Contact::Plane { .. }) => Err(ToolError::invalid(format!(
            "target face `{}` on `{tname}` is an axis (a cylinder); pick a cylindrical own.face on `{name}` to seat on it",
            t.face.as_deref().unwrap_or_default()
        ))),
        (Contact::Plane { .. }, Contact::Axis { .. }) => Err(ToolError::invalid(format!(
            "own face `{}` on `{name}` is a cylinder; pick a cylindrical target.face on `{tname}`, or a flat own.face or own.side",
            own.face.as_deref().unwrap_or_default()
        ))),
    }
}

fn frame_text(frame: &Frame) -> text::FrameText {
    text::FrameText {
        at_mm: mm3(frame.at),
        z: frame.z.map(|v| round(v, 6)),
        x: frame.x.map(|v| round(v, 6)),
    }
}
