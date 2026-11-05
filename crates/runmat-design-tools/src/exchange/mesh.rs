//! Part and assembly meshes, and the mesh formats binary STL and binary glTF (3MF is in
//! `three_mf.rs`).

use crate::{kind_of, text_of, DesignKind, Files, ToolError};
use serde_json::{json, Value};
use std::collections::BTreeMap;

use super::stem;

/// One body (or one occurrence's body), in metres, world placement applied.
pub struct Mesh {
    pub name: String,
    pub material: Option<String>,
    pub positions: Vec<[f64; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

/// One part file's bodies in its own frame (metres), as an assembly's prototypes or a part.
pub struct Shape {
    pub name: String,
    pub part_number: Option<String>,
    pub bodies: Vec<Mesh>,
}

/// A shape placed in the world: the rotation's rows, then the translation (metres).
pub struct Placed {
    pub name: String,
    pub shape: usize,
    pub transform: Option<[f64; 12]>,
}

/// A part's or an assembly's geometry: its shapes and where each is placed.
pub struct Scene3d {
    pub shapes: Vec<Shape>,
    pub placed: Vec<Placed>,
}

impl Scene3d {
    /// Every placed body with its placement applied.
    pub fn world_meshes(&self) -> Vec<Mesh> {
        let mut out = Vec::new();
        for p in &self.placed {
            let shape = &self.shapes[p.shape];
            for (i, b) in shape.bodies.iter().enumerate() {
                let positions = match &p.transform {
                    Some(t) => b
                        .positions
                        .iter()
                        .map(|c| {
                            [
                                t[0] * c[0] + t[1] * c[1] + t[2] * c[2] + t[9],
                                t[3] * c[0] + t[4] * c[1] + t[5] * c[2] + t[10],
                                t[6] * c[0] + t[7] * c[1] + t[8] * c[2] + t[11],
                            ]
                        })
                        .collect(),
                    None => b.positions.clone(),
                };
                out.push(Mesh {
                    name: if shape.bodies.len() == 1 || i == 0 { p.name.clone() } else { format!("{}#{i}", p.name) },
                    material: b.material.clone(),
                    positions,
                    triangles: b.triangles.clone(),
                });
            }
        }
        out
    }
}

pub(super) fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .map(|a| a.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default()
}

fn body_mesh(name: String, material: Option<String>, body: &Value) -> Mesh {
    let mut mesh = Mesh {
        name,
        material,
        positions: Vec::new(),
        triangles: Vec::new(),
    };
    // Faces share their boundary vertices by position; weld them so the shell is closed.
    let mut index: BTreeMap<[i64; 3], u32> = BTreeMap::new();
    for face in body["faces"].as_array().into_iter().flatten() {
        let p = floats(&face["positions"]);
        let local: Vec<u32> = p
            .chunks_exact(3)
            .map(|c| {
                let w = [c[0], c[1], c[2]];
                let key = w.map(|v| (v * 1e8).round() as i64);
                *index.entry(key).or_insert_with(|| {
                    mesh.positions.push(w);
                    (mesh.positions.len() - 1) as u32
                })
            })
            .collect();
        for tri in face["triangles"]
            .as_array()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .chunks_exact(3)
        {
            let g = |v: &Value| {
                local
                    .get(v.as_u64().unwrap_or(0) as usize)
                    .copied()
                    .unwrap_or(0)
            };
            let tri = [g(tri[0]), g(tri[1]), g(tri[2])];
            if tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2] {
                mesh.triangles.push(tri);
            }
        }
    }
    mesh
}

/// A body's mesh from the kernel's tessellation, its faces welded at shared positions.
fn exact_body_mesh(name: String, material: Option<String>, body: &runmat_cad::regen::kernel::BodyMesh) -> Mesh {
    let mut mesh = Mesh { name, material, positions: Vec::new(), triangles: Vec::new() };
    let mut index: BTreeMap<[i64; 3], u32> = BTreeMap::new();
    for face in &body.faces {
        let local: Vec<u32> = face
            .positions
            .iter()
            .map(|w| {
                let key = w.map(|v| (v * 1e8).round() as i64);
                *index.entry(key).or_insert_with(|| {
                    mesh.positions.push(*w);
                    (mesh.positions.len() - 1) as u32
                })
            })
            .collect();
        for t in &face.triangles {
            let g = |i: u32| local.get(i as usize).copied().unwrap_or(0);
            let tri = [g(t[0]), g(t[1]), g(t[2])];
            if tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2] {
                mesh.triangles.push(tri);
            }
        }
    }
    mesh
}

/// A part's bodies (one shape, placed once) or an assembly's prototypes and occurrences.
pub fn model_scene(path: &str, files: &Files) -> Result<Scene3d, ToolError> {
    model_scene_in(path, files, None)
}

/// [`model_scene`] of a part in its named configuration `configuration`.
pub fn model_scene_in(path: &str, files: &Files, configuration: Option<&str>) -> Result<Scene3d, ToolError> {
    let text = text_of(files, path)?;
    let not_compiled = |d: Vec<runmat_cad_wasm::runtime::Diagnostic>| {
        ToolError::new(
            "not_compiled",
            format!(
                "{path} has syntax errors (design.check lists them): {}",
                d.first().map(|d| d.message.as_str()).unwrap_or("")
            ),
        )
    };
    let mut scene = Scene3d { shapes: Vec::new(), placed: Vec::new() };
    match kind_of(path)? {
        DesignKind::Part => {
            let part = runmat_cad_wasm::runtime::regenerate_part_in(path, text, configuration, &Default::default())
                .map_err(not_compiled)?;
            let output = part.document.parts.first();
            let material = output.and_then(|o| o.material.as_ref()).map(|m| m.0.clone());
            let display = serde_json::to_value(&part.display).unwrap_or(Value::Null);
            let mut bodies = Vec::new();
            for body in display["bodies"].as_array().into_iter().flatten() {
                let name = body["reference"]
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("{}_body{}", stem(path), body["index"]));
                // The kernel's tessellation in f64 (the display holds f32 for the canvas).
                let exact = part
                    .result
                    .state
                    .bodies
                    .get(body["index"].as_u64().unwrap_or(0) as usize)
                    .and_then(|b| part.kernel.tessellate(&b.handle, &Default::default()).ok());
                bodies.push(match exact {
                    Some(m) => exact_body_mesh(name, material.clone(), &m),
                    None => body_mesh(name, material.clone(), body),
                });
            }
            // A part's bodies are placed where they are, each under its own name.
            for (i, b) in bodies.iter().enumerate() {
                scene.shapes.push(Shape {
                    name: b.name.clone(),
                    part_number: output.and_then(|o| o.part_number.clone()),
                    bodies: Vec::new(),
                });
                scene.placed.push(Placed { name: b.name.clone(), shape: i, transform: None });
            }
            for (shape, b) in scene.shapes.iter_mut().zip(bodies) {
                shape.bodies.push(b);
            }
        }
        DesignKind::Assembly => {
            let (_, display, _) = runmat_cad_wasm::runtime::compile_assembly(path, text, files)
                .map_err(not_compiled)?;
            let prototypes = display["prototypes"].as_array().cloned().unwrap_or_default();
            let mut shape_of: Vec<Option<usize>> = vec![None; prototypes.len()];
            for (k, p) in prototypes.iter().enumerate() {
                if p["shape"] != "mesh" {
                    continue;
                }
                let file = p["file"].as_str().unwrap_or("part");
                let mut name = stem(file);
                if let Some(c) = p["configuration"].as_str() {
                    name = format!("{name} [{c}]");
                }
                let material = p["material"].as_str().map(str::to_owned);
                let bodies: Vec<Mesh> = p["part"]["bodies"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .enumerate()
                    .map(|(i, body)| {
                        let n = if i == 0 { name.clone() } else { format!("{name}#{i}") };
                        body_mesh(n, material.clone(), body)
                    })
                    .collect();
                shape_of[k] = Some(scene.shapes.len());
                scene.shapes.push(Shape { name, part_number: None, bodies });
            }
            for occ in display["occurrences"].as_array().into_iter().flatten() {
                let Some(Some(shape)) = shape_of.get(occ["prototype"].as_u64().unwrap_or(u64::MAX) as usize) else {
                    continue;
                };
                let t = floats(&occ["transform"]);
                scene.placed.push(Placed {
                    name: occ["path"].as_str().unwrap_or("occurrence").to_owned(),
                    shape: *shape,
                    transform: t.get(..12).map(|t| t.try_into().expect("12 numbers")),
                });
            }
        }
        other => {
            return Err(ToolError::new(
                "unsupported",
                format!("{path} is a {other:?} file; 3D formats export parts and assemblies"),
            ))
        }
    }
    if scene.world_meshes().iter().all(|m| m.triangles.is_empty()) {
        return Err(ToolError::new(
            "not_compiled",
            format!("{path} has no geometry to export (design.check says why)"),
        ));
    }
    Ok(scene)
}

/// The meshes of a part's bodies or an assembly's occurrences, placed in the world.
pub fn model_meshes(path: &str, files: &Files) -> Result<Vec<Mesh>, ToolError> {
    Ok(model_scene(path, files)?.world_meshes())
}

pub(super) fn normal(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if l > 0.0 {
        n.map(|x| x / l)
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Binary STL in millimetres.
pub fn stl(meshes: &[Mesh], name: &str) -> Vec<u8> {
    let count: usize = meshes.iter().map(|m| m.triangles.len()).sum();
    let mut out = Vec::with_capacity(84 + count * 50);
    let mut header = format!("RunMat {name} (mm)").into_bytes();
    header.resize(80, b' ');
    out.extend(header);
    out.extend((count as u32).to_le_bytes());
    for m in meshes {
        for t in &m.triangles {
            let p = t.map(|i| m.positions[i as usize]);
            for v in normal(p[0], p[1], p[2]) {
                out.extend((v as f32).to_le_bytes());
            }
            for q in p {
                for v in q {
                    out.extend(((v * 1e3) as f32).to_le_bytes());
                }
            }
            out.extend([0, 0]);
        }
    }
    out
}

/// Binary glTF 2.0: one mesh and node per body, metres, Y up (model Z up turns to glTF Y).
pub fn glb(meshes: &[Mesh]) -> Vec<u8> {
    let mut bin: Vec<u8> = Vec::new();
    let mut accessors = Vec::new();
    let mut views = Vec::new();
    let mut gl_meshes = Vec::new();
    let mut nodes = Vec::new();
    let mut materials: Vec<Option<String>> = Vec::new();
    for m in meshes.iter().filter(|m| !m.triangles.is_empty()) {
        let material = match materials.iter().position(|x| *x == m.material) {
            Some(k) => k,
            None => {
                materials.push(m.material.clone());
                materials.len() - 1
            }
        };
        let pos_offset = bin.len();
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in &m.positions {
            let q = [p[0], p[2], -p[1]];
            for k in 0..3 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
                bin.extend((q[k] as f32).to_le_bytes());
            }
        }
        let pos_len = bin.len() - pos_offset;
        let idx_offset = bin.len();
        for t in &m.triangles {
            for i in t {
                bin.extend(i.to_le_bytes());
            }
        }
        let idx_len = bin.len() - idx_offset;
        let v = views.len();
        views.push(json!({ "buffer": 0, "byteOffset": pos_offset, "byteLength": pos_len, "target": 34962 }));
        views.push(json!({ "buffer": 0, "byteOffset": idx_offset, "byteLength": idx_len, "target": 34963 }));
        let a = accessors.len();
        accessors.push(json!({ "bufferView": v, "componentType": 5126, "count": m.positions.len(), "type": "VEC3",
            "min": lo.map(|x| x as f32), "max": hi.map(|x| x as f32) }));
        accessors.push(json!({ "bufferView": v + 1, "componentType": 5125, "count": m.triangles.len() * 3, "type": "SCALAR" }));
        let mesh = gl_meshes.len();
        gl_meshes.push(json!({ "name": m.name, "primitives": [{ "attributes": { "POSITION": a }, "indices": a + 1, "material": material }] }));
        nodes.push(json!({ "name": m.name, "mesh": mesh }));
    }
    let doc = json!({
        "asset": { "version": "2.0", "generator": "RunMat" },
        "scene": 0,
        "scenes": [{ "nodes": (0..nodes.len()).collect::<Vec<_>>() }],
        "nodes": nodes,
        "meshes": gl_meshes,
        "materials": materials.iter().map(|m| {
            let (_, color) = runmat_cad::appearance::material_appearance(m.as_deref());
            let mut mat = json!({ "name": m.as_deref().unwrap_or("design"), "pbrMetallicRoughness": { "baseColorFactor": color, "metallicFactor": 0.1, "roughnessFactor": 0.6 } });
            if color[3] < 0.98 {
                mat["alphaMode"] = json!("BLEND");
            }
            mat
        }).collect::<Vec<_>>(),
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{ "byteLength": bin.len() }],
    });
    let mut json_bytes = doc.to_string().into_bytes();
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    let total = 12 + 8 + json_bytes.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend(0x4654_6C67u32.to_le_bytes()); // glTF
    out.extend(2u32.to_le_bytes());
    out.extend((total as u32).to_le_bytes());
    out.extend((json_bytes.len() as u32).to_le_bytes());
    out.extend(0x4E4F_534Au32.to_le_bytes()); // JSON
    out.extend(json_bytes);
    out.extend((bin.len() as u32).to_le_bytes());
    out.extend(0x004E_4942u32.to_le_bytes()); // BIN
    out.extend(bin);
    out
}
