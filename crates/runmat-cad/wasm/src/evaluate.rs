//! Interference and clearance between an assembly's placed parts, on their exact bodies.
//!
//! Every part occurrence is regenerated in its own configuration (once per file and
//! configuration) and placed at its world transform. Pairs whose world boxes overlap go to the
//! kernel's proximity query, which reports the volume of their common material. Boards and
//! other non-part occurrences have no exact body and are left out.

use crate::configured::regenerate_configured;
use crate::runtime::{resolve_assembly_in, Files};
use runmat_cad::assembly::model::DefinitionKind;
use runmat_cad::assembly::source::Aabb;
use runmat_cad::display::part_display;
use runmat_cad::regen::kernel::{BodyHandle, Placed, TessellationOptions, Transform};
use serde::Serialize;
use std::collections::BTreeMap;

/// Two occurrences that share material, by their paths in the assembly.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Overlap {
    pub a: String,
    pub b: String,
    /// What selecting each selects (`occurrence:<uuid>`, as the Outline and canvas name them).
    pub items: [String; 2],
    /// The common volume (m^3).
    pub volume: f64,
}

/// Two occurrences closer than the asked clearance without sharing material.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Near {
    pub a: String,
    pub b: String,
    /// What selecting each selects (`occurrence:<uuid>`).
    pub items: [String; 2],
    /// The gap between their exact bodies (m).
    pub distance: f64,
}

/// The result of [`assembly_interference`] and [`assembly_clearance`].
#[derive(Clone, Debug, Default, Serialize)]
pub struct Interference {
    /// Part occurrences checked.
    pub parts: usize,
    /// Pairs whose boxes overlapped and went to the exact query.
    pub candidates: usize,
    pub overlaps: Vec<Overlap>,
    /// Pairs closer than the clearance asked for (none for an interference check).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub near: Vec<Near>,
    /// The clearance checked (m); zero for an interference check.
    pub clearance: f64,
    /// Pairs the kernel could not answer, with its message.
    pub errors: Vec<String>,
}

struct Placement {
    path: String,
    item: String,
    body: BodyHandle,
    world: Transform,
    bbox: Aabb,
}

/// Overlaps smaller than this are treated as contact (m^3; 1e-3 mm^3).
const CONTACT_VOLUME: f64 = 1e-12;

/// The overlapping part pairs of the assembly `path` in its named configuration `configuration`
/// (the default when `None`).
pub fn assembly_interference(
    path: &str,
    text: &str,
    files: &Files,
    configuration: Option<&str>,
) -> Result<Interference, String> {
    assembly_clearance(path, text, files, configuration, 0.0)
}

/// [`assembly_interference`], and also the pairs whose exact bodies come closer than
/// `clearance` (m) without overlapping: boxes within the clearance go to the kernel's
/// proximity query.
pub fn assembly_clearance(
    path: &str,
    text: &str,
    files: &Files,
    configuration: Option<&str>,
    clearance: f64,
) -> Result<Interference, String> {
    let resolution = resolve_assembly_in(path, text, files, configuration)
        .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
    let resolved = &resolution.resolved;
    // (file, configuration) → the regenerated bodies and their box in the part's frame. The
    // kernels stay alive while their bodies are queried.
    let mut kernels = Vec::new();
    let mut parts: BTreeMap<(String, String), Option<(BodyHandle, Aabb)>> = BTreeMap::new();
    let mut placements = Vec::new();
    for (id, node) in resolved.nodes.iter().enumerate() {
        if node.parent.is_none() || node.failed || node.represented_by_ancestor {
            continue;
        }
        if node.definition.kind != DefinitionKind::Part {
            continue;
        }
        let file = node.definition.file.0.clone();
        let key = (file.clone(), node.configuration.digest());
        if !parts.contains_key(&key) {
            let entry = resolution.documents.get(&file).and_then(|document| {
                let (result, kernel) = regenerate_configured(document, &node.configuration);
                let display =
                    part_display(&result, kernel.as_ref(), &TessellationOptions::default());
                let index = display.bodies.first()?.index as usize;
                let body = result.state.bodies.get(index)?.handle.clone();
                let bounds = display.bounds.as_ref()?;
                let bbox = Aabb {
                    min: bounds.min.map(f64::from),
                    max: bounds.max.map(f64::from),
                };
                kernels.push(kernel);
                Some((body, bbox))
            });
            parts.insert(key.clone(), entry);
        }
        if let Some(Some((body, bbox))) = parts.get(&key) {
            placements.push(Placement {
                path: resolved.path(id as u32),
                item: format!("occurrence:{}", node.uuid),
                body: body.clone(),
                world: node.world,
                bbox: bbox.transformed(&node.world),
            });
        }
    }
    let mut out = Interference {
        parts: placements.len(),
        clearance,
        ..Default::default()
    };
    let Some(kernel) = kernels.first() else {
        return Ok(out);
    };
    for (i, a) in placements.iter().enumerate() {
        for b in &placements[i + 1..] {
            if grown(&a.bbox, clearance).intersection(&b.bbox).is_empty() {
                continue;
            }
            out.candidates += 1;
            let p = kernel.proximity(
                &Placed {
                    body: &a.body,
                    transform: a.world,
                },
                &Placed {
                    body: &b.body,
                    transform: b.world,
                },
            );
            match p {
                Ok(p) if p.common_volume > CONTACT_VOLUME => out.overlaps.push(Overlap {
                    a: a.path.clone(),
                    b: b.path.clone(),
                    items: [a.item.clone(), b.item.clone()],
                    volume: p.common_volume,
                }),
                Ok(p) if clearance > 0.0 && p.distance < clearance => out.near.push(Near {
                    a: a.path.clone(),
                    b: b.path.clone(),
                    items: [a.item.clone(), b.item.clone()],
                    distance: p.distance,
                }),
                Ok(_) => {}
                Err(e) => out.errors.push(format!("{} / {}: {e:?}", a.path, b.path)),
            }
        }
    }
    Ok(out)
}

/// A box grown by `by` on every side.
fn grown(b: &Aabb, by: f64) -> Aabb {
    Aabb {
        min: b.min.map(|v| v - by),
        max: b.max.map(|v| v + by),
    }
}
