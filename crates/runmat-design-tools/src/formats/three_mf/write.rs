//! The 3MF package writer.

use super::{Content, Model, Transform, CORE_NS, MODEL_CONTENT_TYPE, MODEL_REL, PRODUCTION_NS};
use crate::formats::opc::zip_stored;
use std::fmt::Write;

pub(crate) fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// A number as the specification's `ST_Number`: shortest round-trip form, no exponent.
fn num(v: f64) -> String {
    let v = if v == 0.0 { 0.0 } else { v };
    let s = format!("{v}");
    if s.contains('e') {
        format!("{v:.9}")
    } else {
        s
    }
}

fn transform(t: &Transform) -> String {
    t.iter().map(|v| num(*v)).collect::<Vec<_>>().join(" ")
}

fn opt(name: &str, v: &Option<String>) -> String {
    v.as_ref()
        .map(|v| format!(" {name}=\"{}\"", xml_escape(v)))
        .unwrap_or_default()
}

/// The `3D/3dmodel.model` part.
pub fn model_xml(m: &Model) -> String {
    let production = m.build_uuid.is_some()
        || m.items.iter().any(|i| i.uuid.is_some())
        || m.objects.iter().any(|o| o.uuid.is_some());
    let mut s = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<model unit=\"{}\" xml:lang=\"en-US\" xmlns=\"{CORE_NS}\"",
        m.unit
    );
    if production {
        let _ = write!(s, " xmlns:p=\"{PRODUCTION_NS}\"");
    }
    s.push_str(">\n");
    for (k, v) in &m.metadata {
        let _ = writeln!(s, " <metadata name=\"{}\">{}</metadata>", xml_escape(k), xml_escape(v));
    }
    s.push_str(" <resources>\n");
    for (id, mats) in &m.materials {
        let _ = writeln!(s, "  <basematerials id=\"{id}\">");
        for b in mats {
            let _ = writeln!(
                s,
                "   <base name=\"{}\" displaycolor=\"{}\"/>",
                xml_escape(&b.name),
                b.color
            );
        }
        s.push_str("  </basematerials>\n");
    }
    for o in &m.objects {
        let _ = write!(s, "  <object id=\"{}\" type=\"{}\"", o.id, o.kind);
        if let Some((pid, pindex)) = o.material {
            let _ = write!(s, " pid=\"{pid}\" pindex=\"{pindex}\"");
        }
        s.push_str(&opt("name", &o.name));
        s.push_str(&opt("partnumber", &o.part_number));
        s.push_str(&opt("p:UUID", &o.uuid));
        s.push_str(">\n");
        match &o.content {
            Content::Mesh { vertices, triangles } => {
                s.push_str("   <mesh>\n    <vertices>\n");
                for v in vertices {
                    let _ = writeln!(
                        s,
                        "     <vertex x=\"{}\" y=\"{}\" z=\"{}\"/>",
                        num(v[0]),
                        num(v[1]),
                        num(v[2])
                    );
                }
                s.push_str("    </vertices>\n    <triangles>\n");
                for t in triangles {
                    let _ = writeln!(s, "     <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\"/>", t[0], t[1], t[2]);
                }
                s.push_str("    </triangles>\n   </mesh>\n");
            }
            Content::Components(cs) => {
                s.push_str("   <components>\n");
                for c in cs {
                    let _ = write!(s, "    <component objectid=\"{}\"", c.object);
                    if let Some(t) = &c.transform {
                        let _ = write!(s, " transform=\"{}\"", transform(t));
                    }
                    s.push_str(&opt("p:UUID", &c.uuid));
                    s.push_str("/>\n");
                }
                s.push_str("   </components>\n");
            }
        }
        s.push_str("  </object>\n");
    }
    s.push_str(" </resources>\n");
    let _ = writeln!(s, " <build{}>", opt("p:UUID", &m.build_uuid));
    for i in &m.items {
        let _ = write!(s, "  <item objectid=\"{}\"", i.object);
        if let Some(t) = &i.transform {
            let _ = write!(s, " transform=\"{}\"", transform(t));
        }
        s.push_str(&opt("partnumber", &i.part_number));
        s.push_str(&opt("p:UUID", &i.uuid));
        s.push_str("/>\n");
    }
    s.push_str(" </build>\n</model>\n");
    s
}

pub const CONTENT_TYPES: &str = "[Content_Types].xml";
pub const ROOT_RELS: &str = "_rels/.rels";
pub const MODEL_PART: &str = "3D/3dmodel.model";

/// The 3MF package of a model.
pub fn write(m: &Model) -> Vec<u8> {
    let types = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"><Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/><Default Extension=\"model\" ContentType=\"{MODEL_CONTENT_TYPE}\"/></Types>\n"
    );
    let rels = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Target=\"/{MODEL_PART}\" Id=\"rel0\" Type=\"{MODEL_REL}\"/></Relationships>\n"
    );
    zip_stored(&[
        (CONTENT_TYPES, types.into_bytes()),
        (ROOT_RELS, rels.into_bytes()),
        (MODEL_PART, model_xml(m).into_bytes()),
    ])
}
