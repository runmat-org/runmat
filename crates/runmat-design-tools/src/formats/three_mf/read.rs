//! The 3MF package reader: the root model part found through the package relationships, its
//! base materials, objects (meshes and components) and build.

use super::{
    BaseMaterial, Component, Content, Item, Model, Object, Transform, CORE_NS, MODEL_REL,
    PRODUCTION_NS,
};
use crate::formats::opc::unzip;
use crate::formats::xml::{parse, Element};

fn number(e: &Element, a: &str) -> Result<f64, String> {
    e.attr(a)
        .ok_or_else(|| format!("<{}> has no {a}", e.local()))?
        .trim()
        .parse()
        .map_err(|_| format!("<{}> {a} is not a number", e.local()))
}

fn index(e: &Element, a: &str) -> Result<u32, String> {
    e.attr(a)
        .ok_or_else(|| format!("<{}> has no {a}", e.local()))?
        .trim()
        .parse()
        .map_err(|_| format!("<{}> {a} is not a non-negative integer", e.local()))
}

fn transform(e: &Element) -> Result<Option<Transform>, String> {
    let Some(t) = e.attr("transform") else {
        return Ok(None);
    };
    let v: Vec<f64> = t
        .split_whitespace()
        .map(|x| {
            x.parse()
                .map_err(|_| format!("transform `{t}` is not 12 numbers"))
        })
        .collect::<Result<_, _>>()?;
    let m: Transform = v
        .try_into()
        .map_err(|_| format!("transform `{t}` is not 12 numbers"))?;
    Ok(Some(m))
}

fn uuid(e: &Element) -> Option<String> {
    e.attr(&format!("{{{PRODUCTION_NS}}}UUID"))
        .map(str::to_owned)
}

/// The root model part's path in the package.
fn model_path(entries: &[(String, Vec<u8>)]) -> Result<String, String> {
    let rels = entries
        .iter()
        .find(|(n, _)| n == "_rels/.rels")
        .ok_or("the package has no _rels/.rels")?;
    let rels = parse(&String::from_utf8_lossy(&rels.1))?;
    let target = rels
        .all("Relationship")
        .find(|r| r.attr("Type") == Some(MODEL_REL))
        .and_then(|r| r.attr("Target"))
        .ok_or("the package has no 3D model relationship")?;
    Ok(target.trim_start_matches('/').to_owned())
}

/// Parse a model part's XML.
pub fn read_model_xml(text: &str) -> Result<Model, String> {
    let root = parse(text)?;
    if root.name != format!("{{{CORE_NS}}}model") {
        return Err(format!(
            "the root element is {}, not the 3MF core <model>",
            root.name
        ));
    }
    let mut m = Model {
        unit: root.attr("unit").unwrap_or("millimeter").to_owned(),
        metadata: root
            .all("metadata")
            .map(|e| (e.attr("name").unwrap_or("").to_owned(), e.text.clone()))
            .collect(),
        materials: Vec::new(),
        objects: Vec::new(),
        build_uuid: None,
        items: Vec::new(),
    };
    let resources = root
        .first("resources")
        .ok_or("<model> has no <resources>")?;
    for r in &resources.children {
        if r.namespace() != Some(CORE_NS) {
            continue; // other extensions' resources
        }
        match r.local() {
            "basematerials" => m.materials.push((
                index(r, "id")?,
                r.all("base")
                    .map(|b| BaseMaterial {
                        name: b.attr("name").unwrap_or("").to_owned(),
                        color: b.attr("displaycolor").unwrap_or("").to_owned(),
                    })
                    .collect(),
            )),
            "object" => {
                let content = if let Some(mesh) = r.first("mesh") {
                    let vertices = mesh
                        .first("vertices")
                        .ok_or("a <mesh> has no <vertices>")?
                        .all("vertex")
                        .map(|v| Ok([number(v, "x")?, number(v, "y")?, number(v, "z")?]))
                        .collect::<Result<_, String>>()?;
                    let triangles = mesh
                        .first("triangles")
                        .map(|t| {
                            t.all("triangle")
                                .map(|t| Ok([index(t, "v1")?, index(t, "v2")?, index(t, "v3")?]))
                                .collect::<Result<Vec<_>, String>>()
                        })
                        .transpose()?
                        .unwrap_or_default();
                    Content::Mesh {
                        vertices,
                        triangles,
                    }
                } else if let Some(cs) = r.first("components") {
                    Content::Components(
                        cs.all("component")
                            .map(|c| {
                                Ok(Component {
                                    object: index(c, "objectid")?,
                                    transform: transform(c)?,
                                    uuid: uuid(c),
                                })
                            })
                            .collect::<Result<_, String>>()?,
                    )
                } else {
                    return Err(format!(
                        "object {} has neither a mesh nor components",
                        r.attr("id").unwrap_or("?")
                    ));
                };
                let material = match (r.attr("pid"), r.attr("pindex")) {
                    (Some(_), _) => Some((
                        index(r, "pid")?,
                        r.attr("pindex")
                            .map(|_| index(r, "pindex"))
                            .transpose()?
                            .unwrap_or(0),
                    )),
                    _ => None,
                };
                m.objects.push(Object {
                    id: index(r, "id")?,
                    name: r.attr("name").map(str::to_owned),
                    part_number: r.attr("partnumber").map(str::to_owned),
                    kind: r.attr("type").unwrap_or("model").to_owned(),
                    material,
                    uuid: uuid(r),
                    content,
                });
            }
            _ => {}
        }
    }
    let build = root.first("build").ok_or("<model> has no <build>")?;
    m.build_uuid = uuid(build);
    for i in build.all("item") {
        m.items.push(Item {
            object: index(i, "objectid")?,
            transform: transform(i)?,
            part_number: i.attr("partnumber").map(str::to_owned),
            uuid: uuid(i),
        });
    }
    Ok(m)
}

/// Read a 3MF package.
pub fn read(bytes: &[u8]) -> Result<Model, String> {
    let entries = unzip(bytes)?;
    if !entries.iter().any(|(n, _)| n == "[Content_Types].xml") {
        return Err("the package has no [Content_Types].xml".into());
    }
    let path = model_path(&entries)?;
    let part = entries
        .iter()
        .find(|(n, _)| *n == path)
        .ok_or_else(|| format!("the model part {path} is missing"))?;
    let text = std::str::from_utf8(&part.1).map_err(|_| format!("{path} is not UTF-8"))?;
    read_model_xml(text)
}
