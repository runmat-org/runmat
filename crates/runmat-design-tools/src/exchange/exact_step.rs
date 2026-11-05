//! Exact STEP AP242 from the installed kernel: each part regenerated and written as its exact
//! B-rep (OCCT's STEP writer, product names and materials), assemblies as one product per part
//! file placed at every occurrence. `None` when the kernel cannot export (the grid test
//! double), so the caller falls back to the faceted writer.

use crate::{kind_of, text_of, DesignKind, Files, ToolError};
use runmat_cad::regen::kernel::{ExchangeDocument, ExchangeFormat, ExchangeOccurrence, ExchangePart, Transform};
use runmat_cad_wasm::runtime::{regenerate_part, RegeneratedPart};

fn regenerated(path: &str, text: &str) -> Result<RegeneratedPart, ToolError> {
    regenerate_part(path, text).map_err(|d| {
        ToolError::new(
            "not_compiled",
            format!("{path} has syntax errors (design.check lists them): {}", d.first().map(|d| d.message.as_str()).unwrap_or("")),
        )
    })
}

fn stem(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.split('.').next().unwrap_or(name).to_string()
}

fn part_of<'a>(path: &str, part: &'a RegeneratedPart) -> ExchangePart<'a> {
    ExchangePart {
        name: stem(path),
        bodies: part.result.state.bodies.iter().map(|b| &b.handle).collect(),
        color: None,
        material: part.document.parts.first().and_then(|o| o.material.as_ref()).map(|m| m.0.clone()),
        face_names: Vec::new(),
        face_colors: Vec::new(),
    }
}

/// The STEP AP242 text of the part or assembly at `path`, or `None` when the installed kernel
/// does not write STEP.
pub fn exact_step(path: &str, files: &Files) -> Result<Option<Vec<u8>>, ToolError> {
    let text = text_of(files, path)?;
    match kind_of(path)? {
        DesignKind::Part => {
            let part = regenerated(path, text)?;
            let doc = ExchangeDocument { name: stem(path), parts: vec![part_of(path, &part)], occurrences: Vec::new() };
            Ok(part.kernel.export(&doc, ExchangeFormat::Ap242).ok())
        }
        DesignKind::Assembly => {
            let (_, display, _) = runmat_cad_wasm::runtime::compile_assembly(path, text, files).map_err(|d| {
                ToolError::new("not_compiled", format!("{path}: {}", d.first().map(|d| d.message.as_str()).unwrap_or("")))
            })?;
            let prototypes = display["prototypes"].as_array().cloned().unwrap_or_default();
            // One regeneration per part file, all on one kernel so the export sees every body.
            let mut parts: Vec<(String, RegeneratedPart)> = Vec::new();
            let mut index_of = Vec::new();
            for p in &prototypes {
                let file = p["file"].as_str().unwrap_or("");
                let full = files.keys().find(|k| *k == file || k.ends_with(&format!("/{file}"))).cloned();
                let at = match &full {
                    Some(f) => match parts.iter().position(|(n, _)| n == f) {
                        Some(i) => Some(i),
                        None => {
                            parts.push((f.clone(), regenerated(f, &files[f])?));
                            Some(parts.len() - 1)
                        }
                    },
                    None => None,
                };
                index_of.push(at);
            }
            let Some((_, first)) = parts.first() else {
                return Ok(None);
            };
            // Each regeneration has its own kernel; OCCT keeps bodies in one process-wide store,
            // so the first kernel writes them all.
            let kernel = &first.kernel;
            let doc = ExchangeDocument {
                name: stem(path),
                parts: parts.iter().map(|(f, p)| part_of(f, p)).collect(),
                occurrences: display["occurrences"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|o| {
                        let part = index_of.get(o["prototype"].as_u64()? as usize).copied().flatten()? as u32;
                        let t: Vec<f64> = o["transform"].as_array()?.iter().filter_map(|v| v.as_f64()).collect();
                        if t.len() != 12 {
                            return None;
                        }
                        let mut transform = Transform::IDENTITY;
                        for r in 0..3 {
                            for c in 0..3 {
                                transform.rotation[r][c] = t[r * 3 + c];
                            }
                            transform.translation[r] = t[9 + r];
                        }
                        Some(ExchangeOccurrence { name: o["path"].as_str().unwrap_or("").to_string(), part: Some(part), parent: None, transform })
                    })
                    .collect(),
            };
            Ok(kernel.export(&doc, ExchangeFormat::Ap242).ok())
        }
        _ => Ok(None),
    }
}
