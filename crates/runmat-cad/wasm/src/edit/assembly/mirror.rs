//! `assembly.mirror`: components reflected about a plane, as an assembly `mirror` statement.
//!
//! The plane is an origin plane (`origin:top`, `origin:front`, `origin:right`) or a planar face
//! of a component, in assembly coordinates. A reflection turns a part inside out, so each image
//! is placed as the reflected seed composed with a reflection in the seed's own coordinates:
//!
//! - **Mirrored copies** reuse the seed's file. The second reflection is about the seed's
//!   mid-plane across the mirror normal (through the centre of its bounds), so a part that is
//!   symmetric about that mid-plane lands exactly where its mirror image would.
//! - **Opposite-hand parts** use another file, the seed's mirror image about its own origin
//!   plane across the mirror normal (a left part and its right part).
//!
//! Seeds whose normal and mid-plane fall the same way share one statement; others get their own.

use super::frames::{dot, unit, V3};
use super::mate::quantity;
use super::place::direction;
use super::{model, parse_pick, Compiled};
use crate::features::picks_of;
use crate::runtime::{regenerate_part, resolve_path, Files};
use runmat_cad::assembly::model::{
    AssemblyDocument, DefinitionRef, MirrorHand, OccurrenceMirror, OccurrenceSource, PlaneSpec,
};
use runmat_cad::syntax::edit::EditFailure;
use serde_json::{Map, Value};

type Values = Map<String, Value>;

fn round6(v: f64) -> f64 {
    let r = (v * 1e6).round() / 1e6;
    if r == 0.0 {
        0.0
    } else {
        r
    }
}

/// The mirror plane in assembly coordinates: a point (mm) and a unit normal.
fn plane(compiled: &Compiled, pick: &str) -> Result<(V3, V3), EditFailure> {
    let normal = match pick {
        "origin:top" | "origin.xy" | "origin:xy" => Some([0.0, 0.0, 1.0]),
        "origin:front" | "origin.zx" | "origin:zx" | "origin:xz" => Some([0.0, 1.0, 0.0]),
        "origin:right" | "origin.yz" | "origin:yz" => Some([1.0, 0.0, 0.0]),
        _ => None,
    };
    if let Some(n) = normal {
        return Ok(([0.0; 3], n));
    }
    if parse_pick(pick).and_then(|p| p.entity).is_none() {
        return Err(model(
            "pick an origin plane or a flat face for the mirror plane",
        ));
    }
    let (n, point) = direction(compiled, pick)?;
    Ok((point, unit(n)))
}

/// The seed's rotation (row-major) from its display transform.
fn rotation(compiled: &Compiled, uuid: &str) -> Option<[f64; 9]> {
    let (_, t) = compiled.placed(uuid)?;
    (t.len() >= 9).then(|| std::array::from_fn(|i| t[i]))
}

/// The centre of a part definition's bounds, in its own coordinates (mm).
fn centre_mm(document: &AssemblyDocument, files: &Files, file: &str) -> Option<V3> {
    let path = resolve_path(&document.path.0, file, files);
    let part = regenerate_part(&path, files.get(&path)?).ok()?;
    let bounds = serde_json::to_value(&part.display).ok()?["bounds"].clone();
    let c = |k: usize| Some((bounds["min"][k].as_f64()? + bounds["max"][k].as_f64()?) * 500.0);
    Some([c(0)?, c(1)?, c(2)?])
}

pub(super) fn mirror(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    files: &Files,
    names: &[String],
    values: &Values,
) -> Result<String, EditFailure> {
    let pick = picks_of(values, "plane")
        .into_iter()
        .next()
        .ok_or_else(|| model("pick the mirror plane"))?;
    let (point, normal) = plane(compiled, &pick)?;
    let opposite = match values
        .get("result")
        .and_then(Value::as_str)
        .unwrap_or("copy")
    {
        "copy" => None,
        "opposite" => {
            let file = values
                .get("opposite")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            if file.is_empty() {
                return Err(model("name the opposite-hand part's file"));
            }
            let file = resolve_path(&document.path.0, file, files);
            if !file.ends_with(".part") {
                return Err(model("the opposite-hand part is a .part file"));
            }
            if !files.contains_key(&file) {
                return Err(model(format!("there is no `{file}` in the project")));
            }
            Some(file)
        }
        other => {
            return Err(model(format!(
                "a mirror makes copies or opposite-hand parts, not `{other}`"
            )))
        }
    };
    if opposite.is_some() && names.len() > 1 {
        return Err(model(
            "mirror one component at a time to an opposite-hand part",
        ));
    }
    // (normal in the seed's coordinates, point on the seed's plane) → seeds.
    let mut groups: Vec<(V3, V3, Vec<String>)> = Vec::new();
    for name in names {
        let uuid = compiled
            .rows
            .iter()
            .find(|r| r["path"].as_str() == Some(name.as_str()))
            .and_then(|r| r["uuid"].as_str())
            .ok_or_else(|| model(format!("`{name}` isn't drawn")))?;
        let r = rotation(compiled, uuid).ok_or_else(|| model(format!("`{name}` isn't drawn")))?;
        // Rᵀ n: the mirror normal in the seed's coordinates.
        let local = unit([
            r[0] * normal[0] + r[3] * normal[1] + r[6] * normal[2],
            r[1] * normal[0] + r[4] * normal[1] + r[7] * normal[2],
            r[2] * normal[0] + r[5] * normal[1] + r[8] * normal[2],
        ])
        .map(round6);
        let occurrence = document
            .occurrences
            .iter()
            .find(|o| &o.name == name)
            .ok_or_else(|| model(format!("the assembly has no component `{name}`")))?;
        let through = match (&opposite, occurrence.source.definition.file.as_str()) {
            (None, file) if file.ends_with(".part") => {
                let c = centre_mm(document, files, file)
                    .ok_or_else(|| model(format!("`{file}` doesn't regenerate")))?;
                // The mid-plane through the centre: its point along the normal is enough.
                let along = dot(c, local);
                local.map(|k| round6(k * along))
            }
            _ => [0.0; 3],
        };
        match groups
            .iter_mut()
            .find(|(n, p, _)| *n == local && *p == through)
        {
            Some((_, _, seeds)) => seeds.push(name.clone()),
            None => groups.push((local, through, vec![name.clone()])),
        }
    }
    let mm = |p: V3| p.map(|c| quantity(round6(c), "mm"));
    let mut made = Vec::new();
    for (local, through, seeds) in groups {
        let name = (1..)
            .map(|n| format!("mirror{n}"))
            .find(|n| !document.mirrors.iter().any(|m| &m.name == n))
            .expect("an unused name");
        let hand = match &opposite {
            None => MirrorHand::Symmetric {
                point: mm(through),
                normal: local,
            },
            Some(file) => MirrorHand::Opposite {
                source: OccurrenceSource::live(DefinitionRef::part(file, None)),
                configuration: Default::default(),
                point: mm(through),
                normal: local,
            },
        };
        document.mirrors.push(OccurrenceMirror {
            name: name.clone(),
            seeds,
            plane: PlaneSpec::Explicit {
                point: mm(point),
                normal: normal.map(round6),
            },
            hand,
            when: None,
        });
        made.push(name);
    }
    Ok(format!(
        "Mirrored {} ({})",
        names.join(", "),
        made.join(", ")
    ))
}
