//! `design.export` and `design.import`.
//!
//! Exports: parts and assemblies as STEP (AP214, a faceted B-rep of the regenerated bodies in
//! millimetres), STL (binary, millimetres), 3MF (millimetres) and glTF (binary `.glb`, metres,
//! Y up as glTF defines); drawing sheets and sheet-metal flat patterns as DXF (millimetres);
//! circuits and boards as a KiCad project. Imports: KiCad projects to `.circuit`,
//! `.circuit.layout` and `.circuit.lock` sources; STEP files to a `.part` whose `import` feature
//! names the file (the names inside the STEP file become the part's face names).

use crate::edit::{stage, StagedChange};
use crate::{kind_of, text_of, DesignKind, Files, ToolError};
use base64::Engine;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    Step,
    Stl,
    #[serde(rename = "3mf")]
    ThreeMf,
    Gltf,
    Dxf,
    Kicad,
}

#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExportInput {
    /// Workspace path of the design file.
    pub path: String,
    pub format: ExportFormat,
    /// Drawings: the sheet to export (default the first).
    #[serde(default)]
    pub sheet: Option<String>,
    /// KiCad: the project name (default the circuit's file stem).
    #[serde(default)]
    pub project_name: Option<String>,
}

/// One exported file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ExportedFile {
    pub name: String,
    pub mime_type: String,
    /// `utf8` or `base64`.
    pub encoding: String,
    pub content: String,
    pub bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ExportOutput {
    pub path: String,
    pub format: ExportFormat,
    pub files: Vec<ExportedFile>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

fn stem(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    for ext in [
        ".circuit.layout",
        ".circuit",
        ".part",
        ".assembly",
        ".drawing",
    ] {
        if let Some(s) = name.strip_suffix(ext) {
            return s.to_owned();
        }
    }
    name.to_owned()
}

fn text_file(name: String, mime: &str, content: String) -> ExportedFile {
    ExportedFile {
        bytes: content.len(),
        name,
        mime_type: mime.into(),
        encoding: "utf8".into(),
        content,
    }
}

fn binary_file(name: String, mime: &str, bytes: &[u8]) -> ExportedFile {
    ExportedFile {
        bytes: bytes.len(),
        name,
        mime_type: mime.into(),
        encoding: "base64".into(),
        content: base64::engine::general_purpose::STANDARD.encode(bytes),
    }
}

/* ------------------------------------------------------------------------------------------------
 * Meshes of parts and assemblies
 * --------------------------------------------------------------------------------------------- */

/// One body (or one occurrence's body), in metres, world placement applied.
pub struct Mesh {
    pub name: String,
    pub positions: Vec<[f64; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

fn floats(v: &Value) -> Vec<f64> {
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

fn normal(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
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

fn step_string(s: &str) -> String {
    s.replace('\'', "''")
}

/// STEP AP214 (`AUTOMOTIVE_DESIGN`): one faceted B-rep solid per body, planar triangular faces,
/// millimetres.
pub fn step(meshes: &[Mesh], name: &str) -> String {
    let mut e: Vec<String> = Vec::new();
    let mut add = |s: String| -> usize {
        e.push(s);
        e.len()
    };
    let ctx = add("APPLICATION_CONTEXT('automotive design')".into());
    add(format!(
        "APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{ctx})"
    ));
    let pctx = add(format!("PRODUCT_CONTEXT('',#{ctx},'mechanical')"));
    let product = add(format!(
        "PRODUCT('{n}','{n}','',(#{pctx}))",
        n = step_string(name)
    ));
    let formation = add(format!("PRODUCT_DEFINITION_FORMATION('','',#{product})"));
    let pdctx = add(format!(
        "PRODUCT_DEFINITION_CONTEXT('part definition',#{ctx},'design')"
    ));
    let pd = add(format!(
        "PRODUCT_DEFINITION('design','',#{formation},#{pdctx})"
    ));
    let pds = add(format!("PRODUCT_DEFINITION_SHAPE('','',#{pd})"));
    let len = add("(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.))".into());
    let ang = add("(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.))".into());
    let sol = add("(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT())".into());
    let unc = add(format!(
        "UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-06),#{len},'distance_accuracy_value','confusion accuracy')"
    ));
    let gctx = add(format!(
        "(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc})) GLOBAL_UNIT_ASSIGNED_CONTEXT((#{len},#{ang},#{sol})) REPRESENTATION_CONTEXT('',''))"
    ));
    let origin = add("CARTESIAN_POINT('',(0.,0.,0.))".into());
    let z = add("DIRECTION('',(0.,0.,1.))".into());
    let x = add("DIRECTION('',(1.,0.,0.))".into());
    let frame = add(format!("AXIS2_PLACEMENT_3D('',#{origin},#{z},#{x})"));
    let mut solids = Vec::new();
    let num = |v: f64| {
        let s = format!("{v:.6}");
        s.trim_end_matches('0').to_owned()
    };
    for m in meshes.iter().filter(|m| !m.triangles.is_empty()) {
        let points: Vec<usize> = m
            .positions
            .iter()
            .map(|p| {
                add(format!(
                    "CARTESIAN_POINT('',({},{},{}))",
                    num(p[0] * 1e3),
                    num(p[1] * 1e3),
                    num(p[2] * 1e3)
                ))
            })
            .collect();
        let mut faces = Vec::new();
        for t in &m.triangles {
            let p = t.map(|i| m.positions[i as usize]);
            let n = normal(p[0], p[1], p[2]);
            let r = {
                let u = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
                let l = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt().max(1e-12);
                u.map(|v| v / l)
            };
            let loop_ = add(format!(
                "POLY_LOOP('',(#{},#{},#{}))",
                points[t[0] as usize], points[t[1] as usize], points[t[2] as usize]
            ));
            let bound = add(format!("FACE_OUTER_BOUND('',#{loop_},.T.)"));
            let dn = add(format!(
                "DIRECTION('',({},{},{}))",
                num(n[0]),
                num(n[1]),
                num(n[2])
            ));
            let dr = add(format!(
                "DIRECTION('',({},{},{}))",
                num(r[0]),
                num(r[1]),
                num(r[2])
            ));
            let place = add(format!(
                "AXIS2_PLACEMENT_3D('',#{},#{dn},#{dr})",
                points[t[0] as usize]
            ));
            let plane = add(format!("PLANE('',#{place})"));
            faces.push(add(format!("FACE_SURFACE('',(#{bound}),#{plane},.T.)")));
        }
        let shell = add(format!(
            "CLOSED_SHELL('',({}))",
            faces
                .iter()
                .map(|f| format!("#{f}"))
                .collect::<Vec<_>>()
                .join(",")
        ));
        solids.push(add(format!(
            "FACETED_BREP('{}',#{shell})",
            step_string(&m.name)
        )));
    }
    let rep = add(format!(
        "FACETED_BREP_SHAPE_REPRESENTATION('{}',(#{frame},{}),#{gctx})",
        step_string(name),
        solids
            .iter()
            .map(|s| format!("#{s}"))
            .collect::<Vec<_>>()
            .join(",")
    ));
    add(format!("SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})"));
    let mut out = String::from("ISO-10303-21;\nHEADER;\n");
    let _ = writeln!(
        out,
        "FILE_DESCRIPTION(('RunMat design export: faceted B-rep, millimetres'),'2;1');"
    );
    let _ = writeln!(
        out,
        "FILE_NAME('{}.step','',(''),(''),'RunMat','RunMat','');",
        step_string(name)
    );
    out.push_str("FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\nENDSEC;\nDATA;\n");
    for (i, s) in e.iter().enumerate() {
        let _ = writeln!(out, "#{}={};", i + 1, s);
    }
    out.push_str("ENDSEC;\nEND-ISO-10303-21;\n");
    out
}

/* ------------------------------------------------------------------------------------------------
 * DXF: drawings and flat patterns
 * --------------------------------------------------------------------------------------------- */

struct Dxf {
    body: String,
}

impl Dxf {
    fn line(&mut self, layer: &str, a: [f64; 2], b: [f64; 2]) {
        let _ = write!(
            self.body,
            "0\nLINE\n8\n{layer}\n10\n{:.4}\n20\n{:.4}\n30\n0.0\n11\n{:.4}\n21\n{:.4}\n31\n0.0\n",
            a[0], a[1], b[0], b[1]
        );
    }
    fn text(&mut self, layer: &str, at: [f64; 2], height: f64, text: &str) {
        let _ = write!(
            self.body,
            "0\nTEXT\n8\n{layer}\n10\n{:.4}\n20\n{:.4}\n30\n0.0\n40\n{height:.3}\n1\n{text}\n",
            at[0], at[1]
        );
    }
    fn rect(&mut self, layer: &str, x: f64, y: f64, w: f64, h: f64) {
        self.line(layer, [x, y], [x + w, y]);
        self.line(layer, [x + w, y], [x + w, y + h]);
        self.line(layer, [x + w, y + h], [x, y + h]);
        self.line(layer, [x, y + h], [x, y]);
    }
    fn finish(self, layers: &[(&str, u8, &str)]) -> String {
        let mut s = String::from(
            "0\nSECTION\n2\nHEADER\n9\n$ACADVER\n1\nAC1009\n9\n$INSUNITS\n70\n4\n0\nENDSEC\n0\nSECTION\n2\nTABLES\n0\nTABLE\n2\nLTYPE\n70\n2\n0\nLTYPE\n2\nCONTINUOUS\n70\n0\n3\nSolid line\n72\n65\n73\n0\n40\n0.0\n0\nLTYPE\n2\nDASHED\n70\n0\n3\n__ __ __\n72\n65\n73\n2\n40\n4.5\n49\n3.0\n49\n-1.5\n0\nENDTAB\n0\nTABLE\n2\nLAYER\n",
        );
        let _ = write!(s, "70\n{}\n", layers.len());
        for (name, color, ltype) in layers {
            let _ = write!(s, "0\nLAYER\n2\n{name}\n70\n0\n62\n{color}\n6\n{ltype}\n");
        }
        s.push_str("0\nENDTAB\n0\nENDSEC\n0\nSECTION\n2\nENTITIES\n");
        s.push_str(&self.body);
        s.push_str("0\nENDSEC\n0\nEOF\n");
        s
    }
}

/// Sheet sizes in millimetres, landscape.
fn sheet_size(name: &str) -> [f64; 2] {
    match name.to_ascii_uppercase().as_str() {
        "A0" => [1189.0, 841.0],
        "A1" => [841.0, 594.0],
        "A2" => [594.0, 420.0],
        "A4" => [297.0, 210.0],
        "B" => [431.8, 279.4],
        "C" => [558.8, 431.8],
        "D" => [863.6, 558.8],
        _ => [420.0, 297.0],
    }
}

/// A drawing sheet as DXF: the frame and title block, each view's visible edges (`VISIBLE`,
/// continuous) and hidden edges (`HIDDEN`, dashed) at the sheet's scale, and view labels.
fn drawing_dxf(
    path: &str,
    text: &str,
    files: &Files,
    sheet: Option<&str>,
) -> Result<(String, String, Vec<String>), ToolError> {
    let compiled: Value =
        serde_json::from_str(&runmat_cad_wasm::drawing::compile_json(path, text, files))
            .unwrap_or(Value::Null);
    let data = &compiled["projection"]["data"];
    let doc = &data["document"];
    let sheets = doc["sheets"].as_array().cloned().unwrap_or_default();
    let Some(sheet) = sheets
        .iter()
        .find(|s| sheet.is_none_or(|n| s["name"].as_str() == Some(n)))
    else {
        return Err(ToolError::invalid(format!(
            "{path} has no sheet {}",
            sheet.unwrap_or("")
        )));
    };
    let sheet_name = sheet["name"].as_str().unwrap_or("1").to_owned();
    let format = doc["formats"]
        .as_array()
        .and_then(|f| f.iter().find(|f| f["name"] == sheet["format"]))
        .cloned()
        .unwrap_or(Value::Null);
    let size_name = format["size"]["name"].as_str().unwrap_or("A3");
    let [w, h] = sheet_size(size_name);
    let ratio = floats(&sheet["scale"]);
    let scale = if ratio.len() == 2 && ratio[1] != 0.0 {
        ratio[0] / ratio[1]
    } else {
        1.0
    };
    let mut dxf = Dxf {
        body: String::new(),
    };
    let mut notes = Vec::new();
    dxf.rect("FRAME", 10.0, 10.0, w - 20.0, h - 20.0);
    let rows = data["title_blocks"][&sheet_name]
        .as_array()
        .cloned()
        .unwrap_or_default();
    for (i, row) in rows.iter().enumerate() {
        let y = 10.0 + i as f64 * 7.0;
        let x = w - 130.0;
        dxf.rect("FRAME", x, y, 120.0, 7.0);
        dxf.text(
            "FRAME",
            [x + 2.0, y + 2.2],
            2.2,
            row[0].as_str().unwrap_or(""),
        );
        dxf.text(
            "FRAME",
            [x + 40.0, y + 2.2],
            2.6,
            row[1].as_str().unwrap_or(""),
        );
    }
    let views: Vec<&Value> = doc["views"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|v| v["sheet"].as_str() == Some(sheet_name.as_str()))
        .collect();
    for (n, view) in views.iter().enumerate() {
        let name = view["name"].as_str().unwrap_or("view");
        let hlr = &data["hlr"][name];
        let own = floats(&view["scale"]);
        let s = if own.len() == 2 && own[1] != 0.0 {
            own[0] / own[1]
        } else {
            scale
        } * 1e3;
        let runs = |key: &str| -> Vec<Vec<f64>> {
            hlr[key]
                .as_array()
                .into_iter()
                .flatten()
                .map(floats)
                .collect()
        };
        let (visible, hidden) = (runs("visible"), runs("hidden"));
        if visible.is_empty() && hidden.is_empty() {
            notes.push(format!(
                "view {name} has no edges (its model isn't in the workspace or doesn't compile)"
            ));
        }
        let mut lo = [f64::INFINITY; 2];
        let mut hi = [f64::NEG_INFINITY; 2];
        for run in visible.iter().chain(&hidden) {
            for p in run.chunks_exact(2) {
                lo = [lo[0].min(p[0]), lo[1].min(p[1])];
                hi = [hi[0].max(p[0]), hi[1].max(p[1])];
            }
        }
        let centre = if lo[0].is_finite() {
            [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0]
        } else {
            [0.0, 0.0]
        };
        let at = floats(&view["at"]);
        let origin = if at.len() == 2 {
            [at[0], at[1]]
        } else {
            [
                w * (0.3 + 0.38 * (n % 2) as f64),
                h * (0.62 - 0.37 * (n / 2 % 2) as f64),
            ]
        };
        let place = |p: &[f64]| {
            [
                origin[0] + (p[0] - centre[0]) * s,
                origin[1] + (p[1] - centre[1]) * s,
            ]
        };
        for (layer, set) in [("VISIBLE", &visible), ("HIDDEN", &hidden)] {
            for run in set.iter() {
                for seg in run.chunks_exact(2).collect::<Vec<_>>().windows(2) {
                    dxf.line(layer, place(seg[0]), place(seg[1]));
                }
            }
        }
        let half_h = if lo[1].is_finite() {
            (hi[1] - lo[1]) * s / 2.0
        } else {
            10.0
        };
        let shown = s / 1e3;
        let label = format!(
            "{}  SCALE {}",
            name.to_uppercase(),
            if shown >= 1.0 {
                format!("{shown}:1")
            } else {
                format!("1:{}", (1.0 / shown).round())
            }
        );
        dxf.text("NOTES", [origin[0], origin[1] - half_h - 7.0], 3.0, &label);
    }
    let out = dxf.finish(&[
        ("FRAME", 7, "CONTINUOUS"),
        ("VISIBLE", 7, "CONTINUOUS"),
        ("HIDDEN", 8, "DASHED"),
        ("NOTES", 3, "CONTINUOUS"),
    ]);
    Ok((sheet_name, out, notes))
}

/// The DXF of a part's flat pattern, when it has one.
fn flat_pattern_dxf(path: &str, text: &str) -> Result<String, ToolError> {
    let part = runmat_cad_wasm::runtime::regenerate_part(path, text)
        .map_err(|_| ToolError::new("not_compiled", format!("{path} has syntax errors")))?;
    for record in part.result.state.records.values() {
        if let runmat_cad::regen::records::FeatureRecord::FlatPattern { layout, .. } = record {
            return Ok(runmat_cad::drawing::flat_pattern::flat_pattern_sheet(layout).to_dxf());
        }
    }
    Err(ToolError::new(
        "unsupported",
        format!("{path} has no flat pattern; DXF exports drawings and sheet-metal flat patterns (add a flat_pattern feature)"),
    ))
}

pub fn export(input: &ExportInput, files: &Files) -> Result<ExportOutput, ToolError> {
    let kind = kind_of(&input.path)?;
    let text = text_of(files, &input.path)?;
    let name = stem(&input.path);
    let mut notes = Vec::new();
    let files_out = match input.format {
        ExportFormat::Step | ExportFormat::Stl | ExportFormat::ThreeMf | ExportFormat::Gltf => {
            let meshes = model_meshes(&input.path, files)?;
            match input.format {
                ExportFormat::Step => {
                    notes.push(
                        "STEP AP214 faceted B-rep of the regenerated bodies, millimetres".into(),
                    );
                    vec![text_file(
                        format!("{name}.step"),
                        "model/step",
                        step(&meshes, &name),
                    )]
                }
                ExportFormat::Stl => vec![binary_file(
                    format!("{name}.stl"),
                    "model/stl",
                    &stl(&meshes, &name),
                )],
                ExportFormat::ThreeMf => vec![binary_file(
                    format!("{name}.3mf"),
                    "model/3mf",
                    &three_mf(&meshes),
                )],
                _ => vec![binary_file(
                    format!("{name}.glb"),
                    "model/gltf-binary",
                    &glb(&meshes),
                )],
            }
        }
        ExportFormat::Dxf => match kind {
            DesignKind::Drawing => {
                let (sheet, dxf, n) =
                    drawing_dxf(&input.path, text, files, input.sheet.as_deref())?;
                notes.extend(n);
                vec![text_file(
                    format!("{name}-{sheet}.dxf"),
                    "image/vnd.dxf",
                    dxf,
                )]
            }
            DesignKind::Part => vec![text_file(
                format!("{name}-flat.dxf"),
                "image/vnd.dxf",
                flat_pattern_dxf(&input.path, text)?,
            )],
            other => {
                return Err(ToolError::new(
                    "unsupported",
                    format!(
                        "DXF exports drawings and part flat patterns; {} is a {other:?} file",
                        input.path
                    ),
                ))
            }
        },
        ExportFormat::Kicad => {
            if !kind.is_circuit() {
                return Err(ToolError::new(
                    "unsupported",
                    format!(
                        "KiCad exports circuits and boards; {} is a {kind:?} file",
                        input.path
                    ),
                ));
            }
            let project = input.project_name.clone().unwrap_or_else(|| name.clone());
            let out = runmat_circuit::runtime::export_kicad(
                &input.path,
                text,
                files,
                &Default::default(),
                &project,
            )
            .map_err(|d| {
                ToolError::new(
                    "not_compiled",
                    format!(
                        "{} doesn't compile (design.check lists why): {}",
                        input.path,
                        d.first().map(|d| d.message.as_str()).unwrap_or("")
                    ),
                )
            })?;
            out.into_iter()
                .map(|(n, t)| text_file(n, "application/x-kicad", t))
                .collect()
        }
    };
    Ok(ExportOutput {
        path: input.path.clone(),
        format: input.format,
        files: files_out,
        notes,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ImportFormat {
    Kicad,
    Step,
}

/// A file handed to `design.import`.
#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ImportFile {
    /// File name (`board.kicad_pcb`, `root.kicad_sch`, `housing.step`).
    pub name: String,
    pub content: String,
    /// `utf8` (default) or `base64`.
    #[serde(default)]
    pub encoding: Option<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ImportInput {
    pub format: ImportFormat,
    pub files: Vec<ImportFile>,
    /// KiCad: the target stem (`boards/io` writes `boards/io.circuit`, `.circuit.layout` and
    /// `.circuit.lock`). STEP: the `.part` to write (`parts/housing.part`); the STEP file goes
    /// beside it.
    pub target: String,
    /// KiCad: the root schematic's file name (default: the schematic no other sheet names).
    #[serde(default)]
    pub root: Option<String>,
}

fn decode(file: &ImportFile) -> Result<String, ToolError> {
    match file.encoding.as_deref() {
        Some("base64") => {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(file.content.as_bytes())
                .map_err(|e| ToolError::invalid(format!("{}: not base64: {e}", file.name)))?;
            String::from_utf8(bytes).map_err(|_| {
                ToolError::invalid(format!(
                    "{} isn't text (STEP and KiCad files are)",
                    file.name
                ))
            })
        }
        _ => Ok(file.content.clone()),
    }
}

pub fn import(input: &ImportInput, files: &Files) -> Result<StagedChange, ToolError> {
    let target = input.target.trim_start_matches("./").to_owned();
    match input.format {
        ImportFormat::Kicad => {
            let mut kicad = BTreeMap::new();
            for f in &input.files {
                kicad.insert(f.name.clone(), decode(f)?);
            }
            let stem = target
                .strip_suffix(".circuit.layout")
                .or_else(|| target.strip_suffix(".circuit"))
                .unwrap_or(&target)
                .to_owned();
            let read = runmat_circuit::runtime::import_kicad(&kicad, input.root.as_deref(), &stem)
                .map_err(ToolError::invalid)?;
            let primary = read
                .files
                .keys()
                .find(|k| k.ends_with(".circuit"))
                .or_else(|| read.files.keys().next())
                .cloned()
                .unwrap_or_default();
            let summary = format!(
                "Import KiCad project as {}",
                read.files.keys().cloned().collect::<Vec<_>>().join(", ")
            );
            let mut change = stage(summary, read.files.into_iter().collect(), files, &primary);
            change.notes.extend(read.warnings);
            Ok(change)
        }
        ImportFormat::Step => {
            let Some(file) = input.files.first() else {
                return Err(ToolError::invalid("give the STEP file in `files`"));
            };
            let content = decode(file)?;
            if !content.trim_start().starts_with("ISO-10303-21") {
                return Err(ToolError::invalid(format!(
                    "{} isn't a STEP file (ISO-10303-21)",
                    file.name
                )));
            }
            let part_path = if target.ends_with(".part") {
                target.clone()
            } else {
                format!("{target}.part")
            };
            let folder = part_path.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
            let step_name = file
                .name
                .rsplit('/')
                .next()
                .unwrap_or(&file.name)
                .to_owned();
            let step_path = if folder.is_empty() {
                step_name.clone()
            } else {
                format!("{folder}/{step_name}")
            };
            let digest = hex::encode(&Sha256::digest(content.as_bytes())[..8]);
            let part_name: String = stem(&part_path)
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() {
                        c.to_ascii_lowercase()
                    } else {
                        '_'
                    }
                })
                .collect();
            let part = format!(
                "runmat 0.7\nstd 1\n\npart {part_name} {{\n    feature step = import(source = \"{step_name}\", digest = sha256_{digest})\n}}\n"
            );
            let summary = format!("Import {step_name} as {part_path}");
            let mut change = stage(
                summary,
                vec![(step_path, content), (part_path.clone(), part)],
                files,
                &part_path,
            );
            change.notes.push(
                "the STEP file is the source of truth: its solid and face names become the part's (feature.step.face(\"<name>\"))"
                    .into(),
            );
            Ok(change)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_reference() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }
}
