//! Exact STEP AP242 of parts and assemblies on the installed kernel.
//!
//! - **Parts.** One product: the regenerated bodies' exact B-rep, the part's material, every
//!   named face under its reference text (`feature.body.face.side(s1.l1)`), and the part's
//!   semantic PMI (datums, geometric tolerances, sizes and locations on those faces; [`pmi`]).
//! - **Assemblies.** The resolved tree as written: each sub-assembly is its own assembly
//!   product holding its occurrences, each at its placement in its parent, so the hierarchy
//!   reads back as authored; a sub-assembly placed twice is one product used twice. A part file placed in several configurations is one product per
//!   configuration, regenerated in it.
//!
//! The kernel that cannot write STEP (the grid test double) gives `None`.

use crate::configured::regenerate_configured;
use crate::runtime::{resolve_assembly, Diagnostic, Files};
use runmat_cad::assembly::model::DefinitionKind;
use runmat_cad::assembly::source::ConfigKey;
use runmat_cad::display::{part_display, PartDisplay};
use runmat_cad::regen::kernel::{
    BodyHandle, ExchangeDocument, ExchangeFaceName, ExchangeFormat, ExchangeOccurrence,
    ExchangePart, TessellationOptions,
};
use runmat_cad::regen::RegenResult;
use std::collections::BTreeMap;

pub mod pmi;

fn stem(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.split('.').next().unwrap_or(name).to_string()
}

/// Every face the display names, under its reference text.
pub fn face_names(display: &PartDisplay) -> Vec<ExchangeFaceName> {
    display
        .bodies
        .iter()
        .flat_map(|b| {
            b.faces.iter().filter_map(move |f| {
                f.reference.clone().map(|name| ExchangeFaceName {
                    body: b.index,
                    face: f.index,
                    name,
                })
            })
        })
        .collect()
}

struct Product {
    name: String,
    material: Option<String>,
    result: RegenResult,
    display: PartDisplay,
    kernel: Box<dyn crate::kernel::DesignKernel>,
}

impl Product {
    fn part(&self) -> ExchangePart<'_> {
        ExchangePart {
            name: self.name.clone(),
            bodies: self
                .result
                .state
                .bodies
                .iter()
                .map(|b| &b.handle as &BodyHandle)
                .collect(),
            color: None,
            material: self.material.clone(),
            face_names: face_names(&self.display),
            face_colors: Vec::new(),
        }
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
    let product = Product {
        name: stem(path),
        material: part
            .document
            .parts
            .first()
            .and_then(|o| o.material.as_ref())
            .map(|m| m.0.clone()),
        result: part.result,
        display: part.display,
        kernel: part.kernel,
    };
    let doc = ExchangeDocument {
        name: stem(path),
        parts: vec![product.part()],
        occurrences: Vec::new(),
    };
    let Ok(step) = product.kernel.export(&doc, ExchangeFormat::Ap242) else {
        return Ok(None);
    };
    if part.document.pmi.is_empty() {
        return Ok(Some(step));
    }
    let (semantic, _) = pmi::step_pmi(&part.document, &product.result, &product.display);
    let text = String::from_utf8(step).map_err(|_| "the kernel's STEP file is not text".to_owned())?;
    let (text, _) = runmat_cad::step::pmi::embed(&text, &semantic)?;
    Ok(Some(text.into_bytes()))
}

/// The assembly at `path` as exact STEP AP242 with its hierarchy, or `None` when the kernel
/// writes no STEP or the assembly places no part.
pub fn assembly_step(path: &str, text: &str, files: &Files) -> Result<Option<Vec<u8>>, String> {
    let resolution = resolve_assembly(path, text, files).map_err(|d| first_message(&d))?;
    let resolved = &resolution.resolved;
    let mut products: Vec<Product> = Vec::new();
    let mut by_key: BTreeMap<(String, String), u32> = BTreeMap::new();
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
                        let name = if node.configuration == ConfigKey::default() {
                            stem(&file)
                        } else {
                            format!("{} ({})", stem(&file), node.configuration.label())
                        };
                        products.push(Product {
                            name,
                            material: document
                                .parts
                                .first()
                                .and_then(|o| o.material.as_ref())
                                .map(|m| m.0.clone()),
                            result,
                            display,
                            kernel,
                        });
                        let i = products.len() as u32 - 1;
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
            part,
            definition,
            same_as,
            parent,
            transform: node.local,
        });
    }
    let Some(first) = products.first() else {
        return Ok(None);
    };
    let doc = ExchangeDocument {
        name: stem(path),
        parts: products.iter().map(Product::part).collect(),
        occurrences,
    };
    // OCCT keeps bodies in one process-wide store, so one product's kernel writes them all.
    Ok(first.kernel.export(&doc, ExchangeFormat::Ap242).ok())
}
