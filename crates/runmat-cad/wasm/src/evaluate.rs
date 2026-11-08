//! Interference and clearance between an assembly's placed parts, on their exact bodies.
//!
//! Every part occurrence is regenerated in its own configuration (once per file and
//! configuration) and placed at its world transform. Pairs whose world boxes overlap go to the
//! kernel's proximity query, which reports the volume of their common material. A circuit
//! board occurrence is checked as its exact solid ([`crate::board_solid`]): the outline at the
//! board's thickness with its component bodies.
//!
//! A check can be limited to the pairs a change touches ([`assembly_clearance_of`]), and keeps its
//! bodies' boxes and pair answers between calls, so rounds of a change stay quick.

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
    /// Part and board occurrences checked.
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
    /// The occurrences whose pairs were checked, when the check was limited to a change's
    /// occurrences; empty when every pair was.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub changed: Vec<String>,
}

/// Where an occurrence's body comes from: a part file in a configuration, or a circuit board.
enum Source<'a> {
    Part(&'a runmat_cad::assembly::source::ConfigKey),
    Board,
}

struct Placement<'a> {
    path: String,
    item: String,
    file: String,
    source: Source<'a>,
    /// The body's content key: its file, what that file needs and its configuration. `None`
    /// when the file isn't among the files read; such a body is never cached.
    key: Option<u64>,
    world: Transform,
    bbox: Aabb,
    /// The occurrence is one of the changed ones (always true without a `changed` list).
    changed: bool,
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
    assembly_clearance_of(path, text, files, configuration, clearance, None)
}

/// [`assembly_clearance`] over the pairs that involve one of the `changed` occurrences (paths
/// in the assembly; a sub-assembly's path takes in its parts), when given: a change's new and
/// moved parts against their neighbours. Pairs between occurrences the change didn't touch keep
/// whatever overlaps the assembly already had, and aren't checked.
///
/// Bodies and pair answers are kept between calls (this thread, by content: the part file's
/// text, what it needs, its configuration and its world placement), so a second check after a
/// small change regenerates only the parts it changed and queries only their pairs.
pub fn assembly_clearance_of(
    path: &str,
    text: &str,
    files: &Files,
    configuration: Option<&str>,
    clearance: f64,
    changed: Option<&[String]>,
) -> Result<Interference, String> {
    let resolution = resolve_assembly_in(path, text, files, configuration)
        .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
    let resolved = &resolution.resolved;
    let is_changed = |p: &str| {
        changed.is_none_or(|c| c.iter().any(|n| p == n || p.starts_with(&format!("{n}/"))))
    };
    // Content key → the regenerated body and its box in the part's frame. The kernels stay alive
    // while their bodies are queried.
    let mut kernels = Vec::new();
    let mut loaded: BTreeMap<String, Option<(BodyHandle, Aabb)>> = BTreeMap::new();
    let mut errors = Vec::new();
    let mut load = |p: &Placement, kernels: &mut Vec<Box<dyn crate::kernel::DesignKernel>>, errors: &mut Vec<String>| -> Option<(BodyHandle, Aabb)> {
        let slot = format!("{}#{}", p.file, p.key.map_or_else(|| p.path.clone(), |k| k.to_string()));
        if let Some(entry) = loaded.get(&slot) {
            return entry.clone();
        }
        let entry = match p.source {
            Source::Board => match crate::board_solid::board_body(&p.file, files) {
                Ok(board) => {
                    errors.extend(board.problems.iter().map(|q| format!("{}: {q}", p.file)));
                    kernels.push(board.kernel);
                    Some((board.body, board.bbox))
                }
                Err(e) => {
                    errors.push(format!("{}: {e}", p.path));
                    None
                }
            },
            Source::Part(configuration) => resolution.documents.get(&p.file).and_then(|document| {
                let (result, kernel) = regenerate_configured(document, configuration);
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
            }),
        };
        loaded.insert(slot, entry.clone());
        entry
    };
    let mut placements = Vec::new();
    for (id, node) in resolved.nodes.iter().enumerate() {
        if node.parent.is_none() || node.failed || node.represented_by_ancestor {
            continue;
        }
        let file = node.definition.file.0.clone();
        let source = match node.definition.kind {
            DefinitionKind::Circuit => Source::Board,
            DefinitionKind::Part => Source::Part(&node.configuration),
            _ => continue,
        };
        let key = cache::content_key(&file, &node.configuration.digest(), files);
        let occurrence_path = resolved.path(id as u32);
        let mut placement = Placement {
            changed: is_changed(&occurrence_path),
            path: occurrence_path,
            item: format!("occurrence:{}", node.uuid),
            file,
            source,
            key,
            world: node.world,
            bbox: Aabb { min: [0.0; 3], max: [0.0; 3] },
        };
        let local = match key.and_then(cache::bbox) {
            Some(bbox) => Some(bbox),
            None => {
                let bbox = load(&placement, &mut kernels, &mut errors).map(|(_, bbox)| bbox);
                if let (Some(k), Some(b)) = (key, bbox) {
                    cache::set_bbox(k, b);
                }
                bbox
            }
        };
        let Some(local) = local else { continue };
        placement.bbox = local.transformed(&node.world);
        placements.push(placement);
    }
    if let Some(names) = changed {
        if let Some(missing) = names.iter().find(|n| !placements.iter().any(|p| p.path == **n || p.path.starts_with(&format!("{n}/")))) {
            return Err(format!("no part occurrence of the assembly is at `{missing}`"));
        }
    }
    let mut out = Interference {
        parts: placements.len(),
        clearance,
        errors,
        changed: changed.map(<[String]>::to_vec).unwrap_or_default(),
        ..Default::default()
    };
    for (i, a) in placements.iter().enumerate() {
        for b in &placements[i + 1..] {
            if !(a.changed || b.changed) || grown(&a.bbox, clearance).intersection(&b.bbox).is_empty() {
                continue;
            }
            out.candidates += 1;
            let pair = cache::pair_key(a.key, &a.world, b.key, &b.world, clearance);
            let found = match pair.and_then(cache::pair) {
                Some(found) => Ok(found),
                None => {
                    let bodies = load(a, &mut kernels, &mut out.errors).zip(load(b, &mut kernels, &mut out.errors));
                    match (bodies, kernels.first()) {
                        (Some(((body_a, _), (body_b, _))), Some(kernel)) => kernel
                            .proximity(
                                &Placed { body: &body_a, transform: a.world },
                                &Placed { body: &body_b, transform: b.world },
                            )
                            .map(|p| cache::Found { volume: p.common_volume, distance: p.distance })
                            .map_err(|e| format!("{e:?}")),
                        _ => continue,
                    }
                }
            };
            match found {
                Ok(p) => {
                    if let Some(k) = pair {
                        cache::set_pair(k, p);
                    }
                    if p.volume > CONTACT_VOLUME {
                        out.overlaps.push(Overlap {
                            a: a.path.clone(),
                            b: b.path.clone(),
                            items: [a.item.clone(), b.item.clone()],
                            volume: p.volume,
                        });
                    } else if clearance > 0.0 && p.distance < clearance {
                        out.near.push(Near {
                            a: a.path.clone(),
                            b: b.path.clone(),
                            items: [a.item.clone(), b.item.clone()],
                            distance: p.distance,
                        });
                    }
                }
                Err(e) => out.errors.push(format!("{} / {}: {e}", a.path, b.path)),
            }
        }
    }
    Ok(out)
}

/// The content key of part `file` in configuration `digest` ([`assembly_clearance_of`]'s cache).
pub fn content_key(file: &str, digest: &str, files: &Files) -> Option<u64> {
    cache::content_key(file, digest, files)
}

/// The part-frame box an interference check found for the body with content `key`.
pub fn cached_bbox(key: u64) -> Option<Aabb> {
    cache::bbox(key)
}

/// What interference checks keep between calls on this thread, by content.
mod cache {
    use super::{Aabb, Files, Transform};
    use std::cell::RefCell;
    use std::collections::{BTreeSet, HashMap};
    use std::hash::{Hash, Hasher};

    /// A pair's answer from the kernel's proximity query.
    #[derive(Clone, Copy, Debug)]
    pub struct Found {
        pub volume: f64,
        pub distance: f64,
    }

    /// Entries kept before the cache starts over (a pair answer is a few dozen bytes).
    const MAX_ENTRIES: usize = 50_000;

    #[derive(Default)]
    struct Cache {
        boxes: HashMap<u64, Aabb>,
        pairs: HashMap<u64, Found>,
    }

    thread_local! {
        static CACHE: RefCell<Cache> = RefCell::new(Cache::default());
    }

    /// The key of `file` in configuration `digest`: its text and, in turn, the texts of the files
    /// it needs. `None` when `file` isn't among `files`.
    pub fn content_key(file: &str, digest: &str, files: &Files) -> Option<u64> {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        digest.hash(&mut hasher);
        let mut seen = BTreeSet::new();
        let mut queue = vec![file.to_owned()];
        while let Some(next) = queue.pop() {
            if !seen.insert(next.clone()) {
                continue;
            }
            let Some(text) = files.get(&next) else {
                if next == file {
                    return None;
                }
                // A folder it can name designs or functions in: every file under it.
                let prefix = format!("{}/", next.trim_end_matches('/'));
                for (path, text) in files.range(prefix.clone()..).take_while(|(p, _)| p.starts_with(&prefix)) {
                    (path, text).hash(&mut hasher);
                }
                continue;
            };
            (&next, text).hash(&mut hasher);
            queue.extend(crate::runtime::dependencies(&next, text));
        }
        Some(hasher.finish())
    }

    fn placed(hasher: &mut impl Hasher, key: u64, world: &Transform) {
        key.hash(hasher);
        for v in world.rotation.iter().flatten().chain(world.translation.iter()) {
            v.to_bits().hash(hasher);
        }
    }

    /// The key of a pair at its placements and clearance (order-independent).
    pub fn pair_key(a: Option<u64>, wa: &Transform, b: Option<u64>, wb: &Transform, clearance: f64) -> Option<u64> {
        let (a, b) = (a?, b?);
        let mut ha = std::collections::hash_map::DefaultHasher::new();
        placed(&mut ha, a, wa);
        let mut hb = std::collections::hash_map::DefaultHasher::new();
        placed(&mut hb, b, wb);
        let (x, y) = (ha.finish(), hb.finish());
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (x.min(y), x.max(y), clearance.to_bits()).hash(&mut hasher);
        Some(hasher.finish())
    }

    fn with<T>(f: impl FnOnce(&mut Cache) -> T) -> T {
        CACHE.with(|c| {
            let mut c = c.borrow_mut();
            if c.boxes.len() + c.pairs.len() > MAX_ENTRIES {
                *c = Cache::default();
            }
            f(&mut c)
        })
    }

    pub fn bbox(key: u64) -> Option<Aabb> {
        with(|c| c.boxes.get(&key).copied())
    }
    pub fn set_bbox(key: u64, bbox: Aabb) {
        with(|c| c.boxes.insert(key, bbox));
    }
    pub fn pair(key: u64) -> Option<Found> {
        with(|c| c.pairs.get(&key).copied())
    }
    pub fn set_pair(key: u64, found: Found) {
        with(|c| c.pairs.insert(key, found));
    }
}

/// A box grown by `by` on every side.
fn grown(b: &Aabb, by: f64) -> Aabb {
    Aabb {
        min: b.min.map(|v| v - by),
        max: b.max.map(|v| v + by),
    }
}
