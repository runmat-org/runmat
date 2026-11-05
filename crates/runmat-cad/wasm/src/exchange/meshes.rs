//! An assembly's exact meshes for the mesh formats (3MF, STL, glTF): one prototype per part
//! file and configuration, tessellated by the kernel in f64, and each occurrence at its world
//! transform from the resolved tree (f64). The canvas display holds f32 transforms and coarser
//! assembly tessellations, so exports never read it: a 10 mm placement exports as 10 mm.

use crate::configured::regenerate_configured;
use crate::runtime::{resolve_assembly_in, Diagnostic, Files};
use runmat_cad::assembly::model::DefinitionKind;
use runmat_cad::assembly::source::ConfigKey;
use runmat_cad::regen::kernel::{BodyMesh, FaceMesh, TessellationOptions};
use std::collections::BTreeMap;

/// One part file in one configuration: its bodies in the part's frame (metres).
pub struct MeshPrototype {
    pub file: String,
    /// The configuration's name, when the occurrence names one.
    pub configuration: Option<String>,
    pub material: Option<String>,
    pub part_number: Option<String>,
    pub bodies: Vec<BodyMesh>,
}

/// One drawn occurrence: its display path, its prototype and its world transform (the
/// rotation's rows, then the translation in metres).
pub struct MeshOccurrence {
    pub path: String,
    pub prototype: usize,
    pub world: [f64; 12],
}

pub struct AssemblyMeshes {
    pub prototypes: Vec<MeshPrototype>,
    pub occurrences: Vec<MeshOccurrence>,
}

/// A board display's faces as kernel meshes (the board model has no kernel bodies).
fn display_bodies(display: &runmat_cad::display::PartDisplay) -> Vec<BodyMesh> {
    display
        .bodies
        .iter()
        .map(|b| BodyMesh {
            faces: b
                .faces
                .iter()
                .map(|f| FaceMesh {
                    positions: f
                        .positions
                        .chunks_exact(3)
                        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
                        .collect(),
                    triangles: f.triangles.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect(),
                    normals: Vec::new(),
                })
                .collect(),
            ..Default::default()
        })
        .collect()
}

/// The assembly at `path` (in its named configuration `configuration`) as exact meshes.
pub fn assembly_meshes(
    path: &str,
    text: &str,
    files: &Files,
    configuration: Option<&str>,
) -> Result<AssemblyMeshes, Vec<Diagnostic>> {
    let resolution = resolve_assembly_in(path, text, files, configuration)?;
    let resolved = &resolution.resolved;
    let mut out = AssemblyMeshes { prototypes: Vec::new(), occurrences: Vec::new() };
    let mut prototype_of: BTreeMap<(String, String), usize> = BTreeMap::new();
    for (id, node) in resolved.nodes.iter().enumerate() {
        if node.failed || node.represented_by_ancestor {
            continue;
        }
        let file = node.definition.file.0.clone();
        let key = (file.clone(), node.configuration.digest());
        let prototype = match prototype_of.get(&key) {
            Some(p) => *p,
            None => {
                let (bodies, document_material, part_number) = match node.definition.kind {
                    DefinitionKind::Part => {
                        let Some(document) = resolution.documents.get(&file) else {
                            continue;
                        };
                        let (result, kernel) = regenerate_configured(document, &node.configuration);
                        let options = TessellationOptions::default();
                        let bodies = result
                            .state
                            .bodies
                            .iter()
                            .filter_map(|b| kernel.tessellate(&b.handle, &options).ok())
                            .collect();
                        let output = document.parts.first();
                        (
                            bodies,
                            output.and_then(|o| o.material.as_ref()).map(|m| m.0.clone()),
                            output.and_then(|o| o.part_number.clone()),
                        )
                    }
                    DefinitionKind::Circuit => match resolution.displays.get(&file) {
                        Some(display) => (display_bodies(display), None, None),
                        None => continue,
                    },
                    _ => continue,
                };
                out.prototypes.push(MeshPrototype {
                    file: file.clone(),
                    configuration: (node.configuration != ConfigKey::default())
                        .then(|| node.configuration.name.as_ref().map(|c| c.to_string()))
                        .flatten(),
                    // The occurrence's material (an override), else the part's own.
                    material: node.material.as_ref().map(|m| m.0.clone()).or(document_material),
                    part_number: node.part_number.clone().or(part_number),
                    bodies,
                });
                let p = out.prototypes.len() - 1;
                prototype_of.insert(key, p);
                p
            }
        };
        let w = node.world;
        let mut world = [0.0f64; 12];
        for r in 0..3 {
            for c in 0..3 {
                world[r * 3 + c] = w.rotation[r][c];
            }
            world[9 + r] = w.translation[r];
        }
        out.occurrences.push(MeshOccurrence { path: resolved.path(id as u32), prototype, world });
    }
    Ok(out)
}
