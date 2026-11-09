//! Conformance checks of a 3MF model against the core specification and the production
//! extension: what the XSD states (enumerations, colors, UUID forms) and the rules the
//! specification adds in prose (resource ids unique and defined before use, indices in range,
//! meshes closed, consistently oriented and outward, production UUIDs everywhere once used).

use super::{Content, Model};
use std::collections::{BTreeMap, BTreeSet};

const UNITS: &[&str] = &[
    "micron",
    "millimeter",
    "centimeter",
    "inch",
    "foot",
    "meter",
];
const KINDS: &[&str] = &["model", "solidsupport", "support", "surface", "other"];

fn is_color(s: &str) -> bool {
    let h = s.strip_prefix('#').unwrap_or("");
    s.starts_with('#') && (h.len() == 6 || h.len() == 8) && h.chars().all(|c| c.is_ascii_hexdigit())
}

fn is_uuid(s: &str) -> bool {
    let parts: Vec<&str> = s.split('-').collect();
    parts.len() == 5
        && parts.iter().map(|p| p.len()).eq([8, 4, 4, 4, 12])
        && parts.iter().all(|p| {
            p.chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
        })
}

/// Every problem found, empty when the model conforms.
pub fn check(m: &Model) -> Vec<String> {
    let mut out = Vec::new();
    if !UNITS.contains(&m.unit.as_str()) {
        out.push(format!("unit `{}` is not a 3MF unit", m.unit));
    }
    // Resource ids: unique, positive, and each defined before anything refers to it.
    let mut defined: BTreeSet<u32> = BTreeSet::new();
    let mut materials: BTreeMap<u32, usize> = BTreeMap::new();
    for (id, mats) in &m.materials {
        if *id == 0 || !defined.insert(*id) {
            out.push(format!("resource id {id} is zero or used twice"));
        }
        if mats.is_empty() {
            out.push(format!("basematerials {id} has no <base>"));
        }
        for b in mats {
            if !is_color(&b.color) {
                out.push(format!(
                    "base material `{}`: displaycolor `{}` is not #RRGGBB(AA)",
                    b.name, b.color
                ));
            }
        }
        materials.insert(*id, mats.len());
    }
    let production = m.build_uuid.is_some()
        || m.items.iter().any(|i| i.uuid.is_some())
        || m.objects.iter().any(|o| o.uuid.is_some());
    let mut uuids = BTreeSet::new();
    let mut need_uuid = |what: String, u: &Option<String>, out: &mut Vec<String>| {
        if !production {
            return;
        }
        match u {
            None => out.push(format!(
                "{what} has no p:UUID (the production extension requires one)"
            )),
            Some(u) if !is_uuid(u) => out.push(format!("{what}: `{u}` is not a UUID")),
            Some(u) if !uuids.insert(u.to_ascii_lowercase()) => {
                out.push(format!("{what}: UUID {u} is used twice"))
            }
            _ => {}
        }
    };
    let mut kinds: BTreeMap<u32, &str> = BTreeMap::new();
    for o in &m.objects {
        let what = format!("object {}", o.id);
        if o.id == 0 || defined.contains(&o.id) {
            out.push(format!("{what}: resource id is zero or used twice"));
        }
        if !KINDS.contains(&o.kind.as_str()) {
            out.push(format!(
                "{what}: type `{}` is not a 3MF object type",
                o.kind
            ));
        }
        if let Some((pid, pindex)) = o.material {
            match materials.get(&pid) {
                None => out.push(format!(
                    "{what}: pid {pid} names no base-material group defined before it"
                )),
                Some(n) if pindex as usize >= *n => {
                    out.push(format!("{what}: pindex {pindex} is out of range"))
                }
                _ => {}
            }
        }
        need_uuid(what.clone(), &o.uuid, &mut out);
        match &o.content {
            Content::Mesh {
                vertices,
                triangles,
            } => mesh_problems(
                &what,
                vertices,
                triangles,
                o.kind == "model" || o.kind == "solidsupport",
                &mut out,
            ),
            Content::Components(cs) => {
                if cs.is_empty() {
                    out.push(format!("{what}: <components> is empty"));
                }
                for (k, c) in cs.iter().enumerate() {
                    if !kinds.contains_key(&c.object) {
                        out.push(format!(
                            "{what}: component {k} names object {}, which is not defined before it",
                            c.object
                        ));
                    }
                    need_uuid(format!("{what} component {k}"), &c.uuid, &mut out);
                }
            }
        }
        defined.insert(o.id);
        kinds.insert(o.id, o.kind.as_str());
    }
    need_uuid("the build".into(), &m.build_uuid, &mut out);
    for (k, i) in m.items.iter().enumerate() {
        match kinds.get(&i.object) {
            None => out.push(format!(
                "build item {k} names object {}, which does not exist",
                i.object
            )),
            Some(&"other") => out.push(format!(
                "build item {k} names object {} of type other",
                i.object
            )),
            _ => {}
        }
        need_uuid(format!("build item {k}"), &i.uuid, &mut out);
    }
    out
}

/// Indices in range, no degenerate triangles, and (for solids) a closed, consistently
/// oriented 2-manifold of positive volume.
fn mesh_problems(what: &str, v: &[[f64; 3]], t: &[[u32; 3]], solid: bool, out: &mut Vec<String>) {
    if v.len() < 3 || t.is_empty() {
        out.push(format!(
            "{what}: a mesh needs three vertices and a triangle"
        ));
        return;
    }
    let mut edges: BTreeMap<(u32, u32), i32> = BTreeMap::new();
    let mut volume = 0.0;
    for (k, tri) in t.iter().enumerate() {
        if tri.iter().any(|&i| i as usize >= v.len()) {
            out.push(format!("{what}: triangle {k} indexes past the vertices"));
            return;
        }
        if tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] {
            out.push(format!("{what}: triangle {k} repeats a vertex"));
        }
        for e in 0..3 {
            let (a, b) = (tri[e], tri[(e + 1) % 3]);
            *edges.entry((a.min(b), a.max(b))).or_insert(0) += if a < b { 1 } else { -1 };
        }
        let [a, b, c] = tri.map(|i| v[i as usize]);
        volume += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]);
    }
    if !solid {
        return;
    }
    // Each undirected edge must be used once in each direction.
    let mut uses: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    for tri in t {
        for e in 0..3 {
            let (a, b) = (tri[e], tri[(e + 1) % 3]);
            *uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    let open = uses.values().filter(|&&n| n != 2).count();
    let flipped = edges.values().filter(|&&s| s != 0).count();
    if open > 0 {
        out.push(format!("{what}: {open} edges are not shared by exactly two triangles (the mesh is not closed and manifold)"));
    } else if flipped > 0 {
        out.push(format!(
            "{what}: {flipped} edges are used twice in one direction (inconsistent orientation)"
        ));
    } else if volume <= 0.0 {
        out.push(format!(
            "{what}: the triangles face inward (negative volume)"
        ));
    }
}
