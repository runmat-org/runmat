//! Exact STEP AP242 of parts and assemblies on the installed kernel.
//!
//! - **Parts.** One product: the regenerated bodies' exact B-rep, the part's material, every
//!   named face under its name (the first face region that holds it, `region root : face = …`,
//!   which is what a simulation's `select name "root"` reads; else its reference text,
//!   `feature.body.face.side(s1.l1)`), and the part's
//!   semantic PMI (datums, geometric tolerances, sizes and locations on those faces; [`pmi`]).
//! - **Assemblies.** The resolved tree as written: each sub-assembly is its own assembly
//!   product holding its occurrences, each at its placement in its parent, so the hierarchy
//!   reads back as authored; a sub-assembly placed twice is one product used twice. A part file placed in several configurations is one product per
//!   configuration, regenerated in it.
//!
//! The kernel that cannot write STEP (the grid test double) gives `None`.

use crate::configured::regenerate_configured;
use crate::runtime::{resolve_assembly_in, Diagnostic, Files};
use runmat_cad::assembly::model::DefinitionKind;
use runmat_cad::assembly::source::ConfigKey;
use runmat_cad::display::{part_display, PartDisplay};
use runmat_cad::model::PartDocument;
use runmat_cad::regen::kernel::{
    BodyHandle, ExchangeDocument, ExchangeFaceName, ExchangeFormat, ExchangeOccurrence,
    ExchangePart, TessellationOptions,
};
use runmat_cad::regen::RegenResult;
use std::collections::BTreeMap;

pub mod export;
pub mod flat;
pub mod meshes;
pub mod pmi;

fn stem(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.split('.').next().unwrap_or(name).to_string()
}

/// The name a face of `body` travels under in STEP, which carries one name per face: the first
/// of its face regions that takes faces of that body (a region with `body = …` takes only that
/// body's copy of a shared face), under the region's `label` when it has one; else the face's
/// reference text.
pub fn exported_face_name(
    doc: &PartDocument,
    body: &runmat_cad::display::BodyDisplay,
    face: &runmat_cad::display::FaceDisplay,
) -> Option<String> {
    let region = face.regions.iter().find_map(|name| {
        match doc.regions.iter().find(|r| r.name.as_str() == name) {
            Some(r) => match &r.body {
                Some(scope) if body.reference.as_deref() != Some(scope.to_string().as_str()) => None,
                _ => Some(r.label.clone().unwrap_or_else(|| name.clone())),
            },
            None => Some(name.clone()),
        }
    });
    region.or_else(|| face.reference.clone())
}

/// Every face the display names, under its exported name, for the bodies `bodies` (by their
/// index in the regeneration), each renumbered to its position in that list.
fn face_names_of(doc: &PartDocument, display: &PartDisplay, bodies: &[u32]) -> Vec<ExchangeFaceName> {
    display
        .bodies
        .iter()
        .filter_map(|b| bodies.iter().position(|i| *i == b.index).map(|at| (at as u32, b)))
        .flat_map(|(at, b)| {
            b.faces.iter().filter_map(move |f| {
                exported_face_name(doc, b, f).map(|name| ExchangeFaceName { body: at, face: f.index, name })
            })
        })
        .collect()
}

/// One exported product: a name, its material and the bodies (by index) it holds.
struct Group {
    name: String,
    material: Option<String>,
    bodies: Vec<u32>,
}

/// The products a regenerated part exports as. A part with one output (or none) is one product
/// named `name` (or its output's `label`). A part with several outputs (a block and the fluid
/// that fills its passages) is one product per output, named by its `label` or identifier and
/// holding the output's body; bodies no output claims stay under `name`.
fn groups(doc: &PartDocument, display: &PartDisplay, name: &str) -> Vec<Group> {
    let all: Vec<u32> = display.bodies.iter().map(|b| b.index).collect();
    let material = |o: Option<&runmat_cad::model::PartOutput>| {
        o.and_then(|o| o.material.as_ref())
            .or_else(|| doc.parts.first().and_then(|o| o.material.as_ref()))
            .map(|m| m.0.clone())
    };
    if doc.parts.len() < 2 {
        let output = doc.parts.first();
        return vec![Group {
            name: output.and_then(|o| o.label.clone()).unwrap_or_else(|| name.to_owned()),
            material: material(output),
            bodies: all,
        }];
    }
    let mut out = Vec::new();
    let mut claimed = Vec::new();
    for o in &doc.parts {
        let text = o.body.to_string();
        let bodies: Vec<u32> = display
            .bodies
            .iter()
            .filter(|b| b.reference.as_deref() == Some(text.as_str()) && !claimed.contains(&b.index))
            .map(|b| b.index)
            .collect();
        claimed.extend(bodies.iter().copied());
        out.push(Group {
            name: o.label.clone().unwrap_or_else(|| o.name.to_string()),
            material: material(Some(o)),
            bodies,
        });
    }
    let rest: Vec<u32> = all.into_iter().filter(|i| !claimed.contains(i)).collect();
    if !rest.is_empty() {
        out.push(Group { name: name.to_owned(), material: material(None), bodies: rest });
    }
    out.retain(|g| !g.bodies.is_empty());
    out
}

struct Product {
    groups: Vec<Group>,
    document: PartDocument,
    result: RegenResult,
    display: PartDisplay,
    kernel: Box<dyn crate::kernel::DesignKernel>,
}

impl Product {
    fn new(
        name: &str,
        document: PartDocument,
        result: RegenResult,
        display: PartDisplay,
        kernel: Box<dyn crate::kernel::DesignKernel>,
    ) -> Self {
        let groups = groups(&document, &display, name);
        Product { groups, document, result, display, kernel }
    }

    fn parts(&self) -> Vec<ExchangePart<'_>> {
        self.groups
            .iter()
            .map(|g| ExchangePart {
                name: g.name.clone(),
                bodies: g
                    .bodies
                    .iter()
                    .filter_map(|i| self.result.state.bodies.get(*i as usize))
                    .map(|b| &b.handle as &BodyHandle)
                    .collect(),
                color: None,
                material: g.material.clone(),
                face_names: face_names_of(&self.document, &self.display, &g.bodies),
                face_colors: Vec::new(),
            })
            .collect()
    }
}

fn first_message(d: &[Diagnostic]) -> String {
    d.first().map(|d| d.message.clone()).unwrap_or_default()
}

/// The part at `path` as exact STEP AP242, or `None` when the kernel writes no STEP.
pub fn part_step(path: &str, text: &str) -> Result<Option<Vec<u8>>, String> {
    part_step_in(path, text, None)
}

/// [`part_step`] in the named configuration `configuration`.
pub fn part_step_in(path: &str, text: &str, configuration: Option<&str>) -> Result<Option<Vec<u8>>, String> {
    let part = crate::runtime::regenerate_part_in(path, text, configuration, &Default::default())
        .map_err(|d| first_message(&d))?;
    let product = Product::new(&stem(path), part.document.clone(), part.result, part.display, part.kernel);
    let doc = ExchangeDocument {
        name: stem(path),
        parts: product.parts(),
        occurrences: Vec::new(),
    };
    let Ok(step) = product.kernel.export(&doc, ExchangeFormat::Ap242) else {
        return Ok(None);
    };
    if part.document.pmi.is_empty() {
        return Ok(Some(step));
    }
    // A PMI item that can't be written refuses the export: a STEP file missing a tolerance
    // the part declares would pass for the complete definition.
    let (semantic, mut lost) = pmi::step_pmi(&part.document, &product.result, &product.display);
    let text = String::from_utf8(step).map_err(|_| "the kernel's STEP file is not text".to_owned())?;
    let (text, missing) = runmat_cad::step::pmi::embed(&text, &semantic)?;
    lost.extend(missing);
    if !lost.is_empty() {
        return Err(format!("the part's PMI can't be written to STEP: {}", lost.join("; ")));
    }
    Ok(Some(text.into_bytes()))
}

/// The part at `path` as IGES 5.3 (the exact B-rep as trimmed surfaces, with the part's and its
/// faces' names), in the named configuration; `None` when the kernel writes no IGES. IGES has no
/// semantic PMI, so the part's tolerances stay in STEP AP242.
pub fn part_iges_in(path: &str, text: &str, configuration: Option<&str>) -> Result<Option<Vec<u8>>, String> {
    let part = crate::runtime::regenerate_part_in(path, text, configuration, &Default::default())
        .map_err(|d| first_message(&d))?;
    let product = Product::new(&stem(path), part.document.clone(), part.result, part.display, part.kernel);
    let doc = ExchangeDocument { name: stem(path), parts: product.parts(), occurrences: Vec::new() };
    Ok(product.kernel.export(&doc, ExchangeFormat::Iges).ok())
}

/// The assembly at `path` as exact STEP AP242 with its hierarchy, or `None` when the kernel
/// writes no STEP or the assembly places no part.
pub fn assembly_step(path: &str, text: &str, files: &Files) -> Result<Option<Vec<u8>>, String> {
    assembly_step_in(path, text, files, None)
}

/// [`assembly_step`] in the assembly's named configuration `configuration`.
pub fn assembly_step_in(
    path: &str,
    text: &str,
    files: &Files,
    configuration: Option<&str>,
) -> Result<Option<Vec<u8>>, String> {
    let resolution =
        resolve_assembly_in(path, text, files, configuration).map_err(|d| first_message(&d))?;
    let resolved = &resolution.resolved;
    let mut products: Vec<Product> = Vec::new();
    // (file, configuration) → the product and its first exported part's position.
    let mut by_key: BTreeMap<(String, String), (u32, u32)> = BTreeMap::new();
    let mut first_part: Vec<u32> = Vec::new();
    // Node id → its occurrence's position in the export (the root has none).
    let mut placed: BTreeMap<u32, u32> = BTreeMap::new();
    // A sub-assembly (file, configuration) → the occurrence that defines its product.
    let mut assemblies: BTreeMap<(String, String), u32> = BTreeMap::new();
    let mut occurrences = Vec::new();
    for (id, node) in resolved.nodes.iter().enumerate() {
        let id = id as u32;
        let Some(parent) = node.parent else {
            continue;
        };
        if node.failed || node.represented_by_ancestor {
            continue;
        }
        let part = match node.definition.kind {
            DefinitionKind::Part => {
                let file = node.definition.file.0.clone();
                let key = (file.clone(), node.configuration.digest());
                let index = match by_key.get(&key) {
                    Some(i) => *i,
                    None => {
                        let Some(document) = resolution.documents.get(&file) else {
                            continue;
                        };
                        let (result, kernel) = regenerate_configured(document, &node.configuration);
                        let display =
                            part_display(&result, kernel.as_ref(), &TessellationOptions::default());
                        // A configuration with its own part number (the other hand) is
                        // its own product under that number.
                        let configured = document.in_configuration(node.configuration.name.as_ref());
                        let own_number = node
                            .configuration
                            .name
                            .as_ref()
                            .and_then(|n| document.configuration(n))
                            .and_then(|c| c.part_number);
                        let name = match own_number {
                            _ if node.configuration == ConfigKey::default() => stem(&file),
                            Some(pn) if node.configuration.inputs.is_empty() => pn,
                            _ => format!("{} ({})", stem(&file), node.configuration.label()),
                        };
                        let product = Product::new(&name, configured, result, display, kernel);
                        let at = first_part.last().copied().unwrap_or(0)
                            + products.last().map(|p| p.groups.len() as u32).unwrap_or(0);
                        first_part.push(at);
                        products.push(product);
                        let i = (products.len() as u32 - 1, at);
                        by_key.insert(key, i);
                        i
                    }
                };
                Some(index)
            }
            _ => None,
        };
        let parent = match resolved.node(parent).parent {
            None => None,
            Some(_) => match placed.get(&parent) {
                Some(p) => Some(*p),
                // A parent left out (failed, or a sub-assembly used again, whose contents its
                // first use holds) leaves its subtree out.
                None => continue,
            },
        };
        let (definition, same_as) = match part {
            Some(_) => (None, None),
            None => {
                let file = node.definition.file.0.clone();
                let key = (file.clone(), node.configuration.digest());
                (Some(stem(&file)), assemblies.get(&key).copied())
            }
        };
        let at = occurrences.len() as u32;
        if part.is_none() {
            let key = (node.definition.file.0.clone(), node.configuration.digest());
            assemblies.entry(key).or_insert(at);
        }
        if same_as.is_none() {
            placed.insert(id, at);
        }
        occurrences.push(ExchangeOccurrence {
            name: node.segment.to_string(),
            part: part.map(|(_, at)| at),
            definition,
            same_as,
            parent,
            transform: node.local,
        });
        // A part with several outputs places each of its other products beside the first.
        if let Some((p, at)) = part {
            for (k, g) in products[p as usize].groups.iter().enumerate().skip(1) {
                occurrences.push(ExchangeOccurrence {
                    name: format!("{} {}", node.segment, g.name),
                    part: Some(at + k as u32),
                    definition: None,
                    same_as: None,
                    parent,
                    transform: node.local,
                });
            }
        }
    }
    let Some(first) = products.first() else {
        return Ok(None);
    };
    let doc = ExchangeDocument {
        name: stem(path),
        parts: products.iter().flat_map(Product::parts).collect(),
        occurrences,
    };
    // OCCT keeps bodies in one process-wide store, so one product's kernel writes them all.
    Ok(first.kernel.export(&doc, ExchangeFormat::Ap242).ok())
}
