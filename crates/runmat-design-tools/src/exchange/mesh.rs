//! Part and assembly meshes, and the mesh formats: binary STL, 3MF and binary glTF.

use crate::{kind_of, text_of, DesignKind, Files, ToolError};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fmt::Write;

use super::stem;

/// One body (or one occurrence's body), in metres, world placement applied.
pub struct Mesh {
    pub name: String,
    pub positions: Vec<[f64; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

pub(super) fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .map(|a| a.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default()
}

fn body_mesh(name: String, body: &Value, t: Option<&[f64]>) -> Mesh {
    let mut mesh = Mesh {
        name,
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
                let w = match t {
                    Some(t) => [
                        t[0] * c[0] + t[1] * c[1] + t[2] * c[2] + t[9],
                        t[3] * c[0] + t[4] * c[1] + t[5] * c[2] + t[10],
                        t[6] * c[0] + t[7] * c[1] + t[8] * c[2] + t[11],
                    ],
                    None => [c[0], c[1], c[2]],
                };
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

/// The meshes of a part's bodies or an assembly's occurrences.
pub fn model_meshes(path: &str, files: &Files) -> Result<Vec<Mesh>, ToolError> {
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
    let mut out = Vec::new();
    match kind_of(path)? {
        DesignKind::Part => {
            let part = runmat_cad_wasm::runtime::compile_part(path, text).map_err(not_compiled)?;
            let display = serde_json::to_value(&part.display).unwrap_or(Value::Null);
            for body in display["bodies"].as_array().into_iter().flatten() {
                let name = body["reference"]
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("{}_body{}", stem(path), body["index"]));
                out.push(body_mesh(name, body, None));
            }
        }
        DesignKind::Assembly => {
            let (_, display, _) = runmat_cad_wasm::runtime::compile_assembly(path, text, files)
                .map_err(not_compiled)?;
            let prototypes = display["prototypes"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            for occ in display["occurrences"].as_array().into_iter().flatten() {
                let Some(p) =
                    prototypes.get(occ["prototype"].as_u64().unwrap_or(u64::MAX) as usize)
                else {
                    continue;
                };
                if p["shape"] != "mesh" {
                    continue;
                }
                let t = floats(&occ["transform"]);
                let t = (t.len() >= 12).then_some(t.as_slice());
                let path = occ["path"].as_str().unwrap_or("occurrence");
                for (i, body) in p["part"]["bodies"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .enumerate()
                {
                    let name = if i == 0 {
                        path.to_owned()
                    } else {
                        format!("{path}#{i}")
                    };
                    out.push(body_mesh(name, body, t));
                }
            }
        }
        other => {
            return Err(ToolError::new(
                "unsupported",
                format!("{path} is a {other:?} file; 3D formats export parts and assemblies"),
            ))
        }
    }
    if out.iter().all(|m| m.triangles.is_empty()) {
        return Err(ToolError::new(
            "not_compiled",
            format!("{path} has no geometry to export (design.check says why)"),
        ));
    }
    Ok(out)
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

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// The 3MF model part (millimetres).
fn three_mf_model(meshes: &[Mesh]) -> String {
    let mut s = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<model unit=\"millimeter\" xml:lang=\"en-US\" xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">\n <metadata name=\"Application\">RunMat</metadata>\n <resources>\n");
    for (i, m) in meshes.iter().enumerate() {
        let _ = writeln!(
            s,
            "  <object id=\"{}\" type=\"model\" name=\"{}\">\n   <mesh>\n    <vertices>",
            i + 1,
            xml_escape(&m.name)
        );
        for p in &m.positions {
            let _ = writeln!(
                s,
                "     <vertex x=\"{:.6}\" y=\"{:.6}\" z=\"{:.6}\"/>",
                p[0] * 1e3,
                p[1] * 1e3,
                p[2] * 1e3
            );
        }
        s.push_str("    </vertices>\n    <triangles>\n");
        for t in &m.triangles {
            let _ = writeln!(
                s,
                "     <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\"/>",
                t[0], t[1], t[2]
            );
        }
        s.push_str("    </triangles>\n   </mesh>\n  </object>\n");
    }
    s.push_str(" </resources>\n <build>\n");
    for i in 0..meshes.len() {
        let _ = writeln!(s, "  <item objectid=\"{}\"/>", i + 1);
    }
    s.push_str(" </build>\n</model>\n");
    s
}

fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, e) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 == 1 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
        *e = c;
    }
    let mut c = 0xFFFF_FFFFu32;
    for b in data {
        c = table[((c ^ *b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

/// A ZIP archive of stored (uncompressed) entries, as OPC packages (3MF) allow.
pub fn zip_stored(entries: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in entries {
        let offset = out.len() as u32;
        let crc = crc32(data);
        let size = data.len() as u32;
        let local = |buf: &mut Vec<u8>, sig: u32, central: bool| {
            buf.extend(sig.to_le_bytes());
            if central {
                buf.extend(20u16.to_le_bytes()); // made by
            }
            buf.extend(20u16.to_le_bytes()); // version needed
            buf.extend(0u16.to_le_bytes()); // flags
            buf.extend(0u16.to_le_bytes()); // stored
            buf.extend(0u16.to_le_bytes()); // time
            buf.extend(0x21u16.to_le_bytes()); // date: 1980-01-01
            buf.extend(crc.to_le_bytes());
            buf.extend(size.to_le_bytes());
            buf.extend(size.to_le_bytes());
            buf.extend((name.len() as u16).to_le_bytes());
            buf.extend(0u16.to_le_bytes()); // extra
            if central {
                buf.extend(0u16.to_le_bytes()); // comment
                buf.extend(0u16.to_le_bytes()); // disk
                buf.extend(0u16.to_le_bytes()); // internal attrs
                buf.extend(0u32.to_le_bytes()); // external attrs
                buf.extend(offset.to_le_bytes());
            }
            buf.extend(name.as_bytes());
        };
        local(&mut out, 0x0403_4b50, false);
        out.extend(data);
        local(&mut central, 0x0201_4b50, true);
    }
    let cd_offset = out.len() as u32;
    let cd_size = central.len() as u32;
    out.extend(central);
    out.extend(0x0605_4b50u32.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend((entries.len() as u16).to_le_bytes());
    out.extend((entries.len() as u16).to_le_bytes());
    out.extend(cd_size.to_le_bytes());
    out.extend(cd_offset.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out
}

pub fn three_mf(meshes: &[Mesh]) -> Vec<u8> {
    let types = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"><Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/><Default Extension=\"model\" ContentType=\"application/vnd.ms-package.3dmanufacturing-3dmodel+xml\"/></Types>\n";
    let rels = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Target=\"/3D/3dmodel.model\" Id=\"rel0\" Type=\"http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel\"/></Relationships>\n";
    zip_stored(&[
        ("[Content_Types].xml", types.as_bytes().to_vec()),
        ("_rels/.rels", rels.as_bytes().to_vec()),
        ("3D/3dmodel.model", three_mf_model(meshes).into_bytes()),
    ])
}

/// Binary glTF 2.0: one mesh and node per body, metres, Y up (model Z up turns to glTF Y).
pub fn glb(meshes: &[Mesh]) -> Vec<u8> {
    let mut bin: Vec<u8> = Vec::new();
    let mut accessors = Vec::new();
    let mut views = Vec::new();
    let mut gl_meshes = Vec::new();
    let mut nodes = Vec::new();
    for m in meshes.iter().filter(|m| !m.triangles.is_empty()) {
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
        gl_meshes.push(json!({ "name": m.name, "primitives": [{ "attributes": { "POSITION": a }, "indices": a + 1, "material": 0 }] }));
        nodes.push(json!({ "name": m.name, "mesh": mesh }));
    }
    let doc = json!({
        "asset": { "version": "2.0", "generator": "RunMat" },
        "scene": 0,
        "scenes": [{ "nodes": (0..nodes.len()).collect::<Vec<_>>() }],
        "nodes": nodes,
        "meshes": gl_meshes,
        "materials": [{ "name": "design", "pbrMetallicRoughness": { "baseColorFactor": crate::scene::FACE_RGBA, "metallicFactor": 0.1, "roughnessFactor": 0.6 } }],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_reference() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }
}
