//! Placing components: `assembly.insert`, `assembly.move` (the triad's translation and
//! rotation) and `assembly.pattern` (linear and circular component patterns).

use super::frames::{cross, dot, edge_frame, face_frame, to_world, unit, V3};
use super::mate::{field, quantity};
use super::{model, parse_pick, Compiled, Entity};
use crate::features::picks_of;
use crate::runtime::{regenerate_part, resolve_path, Files};
use runmat_cad::assembly::model::{
    AssemblyDocument, DefinitionRef, Occurrence, OccurrenceLayout, OccurrencePattern, OffsetExpr,
    Placement, RotationExpr,
};
use runmat_cad::syntax::edit::EditFailure;
use runmat_cad::value::ValueExpr;
use serde_json::{json, Map, Value};

type Values = Map<String, Value>;
type M3 = [[f64; 3]; 3];

fn text<'a>(values: &'a Values, id: &str, default: &'a str) -> &'a str {
    values.get(id).and_then(Value::as_str).unwrap_or(default)
}

fn number(values: &Values, id: &str) -> Option<f64> {
    match values.get(id)? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        Value::Object(o) => o.get("value")?.get("value")?.as_f64(),
        _ => None,
    }
}

/// A literal length in millimetres (`12 mm`, `0.012 m`).
fn literal_mm(value: &ValueExpr) -> Option<f64> {
    match value {
        ValueExpr::Quantity { value, unit } => Some(match unit.as_str() {
            "mm" => *value,
            "cm" => value * 10.0,
            "m" => value * 1000.0,
            "in" => value * 25.4,
            _ => return None,
        }),
        ValueExpr::Number { value } if *value == 0.0 => Some(0.0),
        _ => None,
    }
}

/// A literal angle in radians.
fn literal_rad(value: &ValueExpr) -> Option<f64> {
    match value {
        ValueExpr::Quantity { value, unit } => match unit.as_str() {
            "deg" => Some(value.to_radians()),
            "rad" => Some(*value),
            _ => None,
        },
        ValueExpr::Number { value } if *value == 0.0 => Some(0.0),
        _ => None,
    }
}

/// A component name from a file name: `bracket-v2.part` → `bracket_v2`.
fn stem(file: &str) -> String {
    let base = file.rsplit('/').next().unwrap_or(file);
    let base = base.split('.').next().unwrap_or(base);
    let mut name: String = base
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
        .collect();
    if !name.starts_with(|c: char| c.is_ascii_alphabetic()) {
        name.insert(0, 'c');
    }
    name
}

/// The definition's width along X in millimetres, when the file is at hand and regenerates.
fn width_mm(document: &AssemblyDocument, files: &Files, file: &str) -> Option<f64> {
    let path = resolve_path(&document.path.0, file, files);
    let part = regenerate_part(&path, files.get(&path)?).ok()?;
    let bounds = serde_json::to_value(&part.display).ok()?["bounds"].clone();
    let (min, max) = (bounds["min"][0].as_f64()?, bounds["max"][0].as_f64()?);
    Some((max - min) * 1000.0)
}

/// `assembly.insert`: the file as new components. The first component of an assembly is fixed
/// (as SolidWorks fixes it); later ones float unless "Fix in place" is on. A click position
/// (`x`, `y`, `z` in mm) places the first instance; further instances stand beside it along X
/// with a fifth of the definition's width between them.
pub(super) fn insert(
    document: &mut AssemblyDocument,
    files: &Files,
    values: &Values,
) -> Result<String, EditFailure> {
    let file = text(values, "file", "").trim().to_owned();
    if file.is_empty() {
        return Err(model("name the file to insert"));
    }
    let definition = match text(values, "source", "part") {
        _ if file.ends_with(".assembly") => DefinitionRef::assembly(&file),
        _ if file.ends_with(".circuit") => DefinitionRef::circuit(&file),
        _ if file.ends_with(".part") => DefinitionRef::part(&file, None),
        "step" | "library" => {
            return Err(model(
                "STEP and library parts are inserted from the Files panel in this build",
            ))
        }
        _ => return Err(model("insert a .part, .assembly or .circuit file")),
    };
    let count = number(values, "quantity").unwrap_or(1.0).max(1.0) as usize;
    let configuration = text(values, "configuration", "").trim().to_owned();
    let fix = values.get("fix").and_then(Value::as_bool).unwrap_or(false);
    let at = [
        number(values, "x").unwrap_or(0.0),
        number(values, "y").unwrap_or(0.0),
        number(values, "z").unwrap_or(0.0),
    ];
    let step = width_mm(document, files, &file).map_or(50.0, |w| w * 1.2);
    let stem = stem(&file);
    let mut names = Vec::new();
    for i in 0..count {
        let name = (1..)
            .map(|n| if n == 1 { stem.clone() } else { format!("{stem}{n}") })
            .find(|n| !document.occurrences.iter().any(|o| &o.name == n))
            .expect("an unused name");
        let mut occurrence = Occurrence::new(&name, definition.clone());
        if !configuration.is_empty() && configuration != "default" {
            occurrence.configuration =
                serde_json::from_value(json!({ "select": "named", "name": configuration }))
                    .map_err(|e| model(format!("configuration `{configuration}`: {e}")))?;
        }
        let offset = [at[0] + step * i as f64, at[1], at[2]];
        if offset != [0.0; 3] {
            occurrence.placement = Placement::Offset {
                offset: OffsetExpr::translation(offset.map(|v| quantity(round3(v), "mm"))),
            };
        }
        occurrence.fixed = fix || document.occurrences.is_empty();
        document.occurrences.push(occurrence);
        names.push(name);
    }
    Ok(format!("Inserted {}", names.join(", ")))
}

fn round3(v: f64) -> f64 {
    let r = (v * 1e3).round() / 1e3;
    if r == 0.0 {
        0.0
    } else {
        r
    }
}

fn mul(a: M3, b: M3) -> M3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn about(axis: usize, angle: f64) -> M3 {
    let (s, c) = angle.sin_cos();
    match axis {
        0 => [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        1 => [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        _ => [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
    }
}

/// A written rotation as a matrix, when it is literal.
fn matrix(rotation: &RotationExpr) -> Option<M3> {
    Some(match rotation {
        RotationExpr::Identity => about(0, 0.0),
        RotationExpr::Euler { yaw, pitch, roll } => mul(
            mul(about(2, literal_rad(yaw)?), about(1, literal_rad(pitch)?)),
            about(0, literal_rad(roll)?),
        ),
        RotationExpr::AxisAngle { axis, angle } => {
            let k = unit(*axis);
            let (s, c) = literal_rad(angle)?.sin_cos();
            let t = 1.0 - c;
            [
                [t * k[0] * k[0] + c, t * k[0] * k[1] - s * k[2], t * k[0] * k[2] + s * k[1]],
                [t * k[0] * k[1] + s * k[2], t * k[1] * k[1] + c, t * k[1] * k[2] - s * k[0]],
                [t * k[0] * k[2] - s * k[1], t * k[1] * k[2] + s * k[0], t * k[2] * k[2] + c],
            ]
        }
        RotationExpr::Basis { x, z } => {
            let z = unit(*z);
            let x = unit([x[0] - z[0] * dot(*x, z), x[1] - z[1] * dot(*x, z), x[2] - z[2] * dot(*x, z)]);
            let y = cross(z, x);
            [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]
        }
    })
}

fn clean(v: V3) -> [f64; 3] {
    v.map(|c| {
        let r = (c * 1e9).round() / 1e9;
        if r == 0.0 {
            0.0
        } else {
            r
        }
    })
}

/// `assembly.move`: adds the triad's translation (`dx`, `dy`, `dz` mm) and rotation (`rx`, `ry`,
/// `rz` degrees about the assembly's axes through the component's origin, X then Y then Z) to
/// each component's placement. A floating component's placement is where mates start from.
pub(super) fn move_components(
    document: &mut AssemblyDocument,
    names: &[String],
    values: &Values,
) -> Result<String, EditFailure> {
    let delta = ["dx", "dy", "dz"].map(|id| number(values, id).unwrap_or(0.0));
    let turn = ["rx", "ry", "rz"].map(|id| number(values, id).unwrap_or(0.0).to_radians());
    if delta == [0.0; 3] && turn == [0.0; 3] {
        return Err(model("drag the triad or type a distance or angle"));
    }
    let spin = mul(about(2, turn[2]), mul(about(1, turn[1]), about(0, turn[0])));
    for name in names {
        place_component(document, name, delta, (turn != [0.0; 3]).then_some(spin))?;
    }
    Ok(format!("Moved {}", names.join(", ")))
}

/// Adds `delta` (mm) to component `name`'s placement and turns it by `spin` about the
/// assembly's axes through its origin.
fn place_component(
    document: &mut AssemblyDocument,
    name: &str,
    delta: [f64; 3],
    spin: Option<M3>,
) -> Result<(), EditFailure> {
    let occurrence = document
        .occurrences
        .iter_mut()
        .find(|o| o.name == name)
        .ok_or_else(|| model(format!("the assembly has no component `{name}`")))?;
    let current = match &occurrence.placement {
        Placement::Origin => OffsetExpr::translation(std::array::from_fn(|_| quantity(0.0, "mm"))),
        Placement::Offset { offset } => offset.clone(),
        Placement::Attach { .. } => {
            return Err(model(format!(
                "`{name}` is attached to a frame; move it by editing its attach offset or mate it"
            )))
        }
    };
    let mut translation = Vec::new();
    for (axis, value) in current.translation.iter().enumerate() {
        let mm = literal_mm(value).ok_or_else(|| {
            model(format!("`{name}`'s placement is an expression; change it in the source"))
        })?;
        translation.push(quantity(round3(mm + delta[axis]), "mm"));
    }
    let rotation = match spin {
        None => current.rotation.clone(),
        Some(spin) => {
            let m = mul(
                spin,
                matrix(&current.rotation).ok_or_else(|| {
                    model(format!("`{name}`'s rotation is an expression; change it in the source"))
                })?,
            );
            RotationExpr::Basis {
                x: clean([m[0][0], m[1][0], m[2][0]]),
                z: clean([m[0][2], m[1][2], m[2][2]]),
            }
        }
    };
    let translation: [ValueExpr; 3] = translation.try_into().expect("three axes");
    occurrence.placement = Placement::Offset {
        offset: OffsetExpr {
            translation,
            rotation,
        },
    };
    Ok(())
}

/// Turns component `name` by `angle` radians about the assembly direction `axis` through its
/// origin: a mate that must rotate a component from a degenerate start (parallel faces for a
/// perpendicular mate) places it there first, as SolidWorks moves a part when a mate is added.
pub(super) fn turn(
    document: &mut AssemblyDocument,
    name: &str,
    axis: V3,
    angle: f64,
) -> Result<(), EditFailure> {
    let k = unit(axis);
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    let spin = [
        [t * k[0] * k[0] + c, t * k[0] * k[1] - s * k[2], t * k[0] * k[2] + s * k[1]],
        [t * k[0] * k[1] + s * k[2], t * k[1] * k[1] + c, t * k[1] * k[2] - s * k[0]],
        [t * k[0] * k[2] - s * k[1], t * k[1] * k[2] + s * k[0], t * k[2] * k[2] + c],
    ];
    place_component(document, name, [0.0; 3], Some(spin))
}

/// A direction pick for a pattern, in assembly coordinates: a face's normal, a cylinder's or
/// circle's axis or an edge's direction on a component, or an origin axis (`origin:x`).
/// Answers the direction and a point on it (mm).
fn direction(compiled: &Compiled, id: &str) -> Result<(V3, V3), EditFailure> {
    match id {
        "origin:x" | "axis:x" => return Ok(([1.0, 0.0, 0.0], [0.0; 3])),
        "origin:y" | "axis:y" => return Ok(([0.0, 1.0, 0.0], [0.0; 3])),
        "origin:z" | "axis:z" => return Ok(([0.0, 0.0, 1.0], [0.0; 3])),
        _ => {}
    }
    let pick = parse_pick(id)
        .and_then(|p| Some((p.uuid.clone(), p.entity?)))
        .ok_or_else(|| model("pick an edge, axis or face for the direction"))?;
    let (prototype, transform) = compiled
        .placed(&pick.0)
        .ok_or_else(|| model("the direction's component isn't drawn"))?;
    let frame = match pick.1 {
        Entity::Face { body, face } => compiled
            .body(prototype, body)
            .and_then(|b| b["faces"].as_array()?.iter().find(|f| f["index"].as_u64() == Some(face as u64)))
            .and_then(face_frame),
        Entity::Edge { body, edge } => compiled
            .body(prototype, body)
            .and_then(|b| b["edges"].as_array()?.iter().find(|e| e["index"].as_u64() == Some(edge as u64)))
            .and_then(edge_frame),
    }
    .ok_or_else(|| model("the direction pick has no direction"))?;
    let world = to_world(frame, &transform);
    Ok((world.z, world.origin.map(|c| c * 1000.0)))
}

/// `assembly.pattern`: the components repeated in a line (`spacing` mm apart) or a circle
/// (`angle` degrees in all) along or about the picked direction, `instances` in all.
pub(super) fn pattern(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    names: &[String],
    values: &Values,
) -> Result<String, EditFailure> {
    let layout_name = text(values, "layout", "linear");
    let pick = picks_of(values, "direction").into_iter().next();
    let (axis, point) = match &pick {
        Some(id) => direction(compiled, id)?,
        None => ([1.0, 0.0, 0.0], [0.0; 3]),
    };
    let count = ValueExpr::Number {
        value: number(values, "instances").unwrap_or(3.0).max(1.0).round(),
    };
    let layout = match layout_name {
        "linear" => OccurrenceLayout::Linear {
            direction: clean(axis),
            spacing: field(values, "spacing", "mm")?,
            count,
        },
        "circular" => OccurrenceLayout::Circular {
            axis_point: point.map(|c| quantity(round3(c), "mm")),
            axis: clean(axis),
            count,
            angle: field(values, "angle", "deg")?,
        },
        _ => {
            return Err(model(
                "a pattern that follows a part's feature pattern is written in the source editor in this build",
            ))
        }
    };
    let name = (1..)
        .map(|n| format!("pattern{n}"))
        .find(|n| !document.patterns.iter().any(|p| &p.name == n))
        .expect("an unused name");
    document.patterns.push(OccurrencePattern {
        name: name.clone(),
        seeds: names.to_vec(),
        layout,
        skip: Vec::new(),
        when: None,
    });
    Ok(format!("Patterned {} ({name})", names.join(", ")))
}
