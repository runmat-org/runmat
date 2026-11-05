//! A part or assembly as a 3MF model (millimetres): one mesh object per body, a base-material
//! group of the materials used (named, with their display colors), a components object per
//! multi-body part, and one build item per placement with its transform. Every object,
//! component, item and the build carry production-extension UUIDs derived from the file path
//! and the object's place in it, so the same model exports the same UUIDs.

use super::mesh::Scene3d;
use crate::formats::three_mf::{self, BaseMaterial, Component, Content, Item, Model, Object};
use runmat_cad::appearance::{material_appearance, srgb_hex};
use sha2::{Digest, Sha256};

/// A name-based UUID (RFC 4122 layout, version 5 bits) of `seed`.
fn uuid(seed: &str) -> String {
    let h = Sha256::digest(seed.as_bytes());
    let mut b = [0u8; 16];
    b.copy_from_slice(&h[..16]);
    b[6] = (b[6] & 0x0F) | 0x50;
    b[8] = (b[8] & 0x3F) | 0x80;
    let x = hex::encode(b);
    format!("{}-{}-{}-{}-{}", &x[..8], &x[8..12], &x[12..16], &x[16..20], &x[20..])
}

fn to_mm(t: &[f64; 12]) -> three_mf::Transform {
    let r = [[t[0], t[1], t[2]], [t[3], t[4], t[5]], [t[6], t[7], t[8]]];
    three_mf::from_rows(r, [t[9] * 1e3, t[10] * 1e3, t[11] * 1e3])
}

pub fn scene_model(scene: &Scene3d, path: &str, title: &str) -> Model {
    let mut names: Vec<Option<String>> = Vec::new();
    for b in scene.shapes.iter().flat_map(|s| &s.bodies) {
        if !names.contains(&b.material) {
            names.push(b.material.clone());
        }
    }
    let group = 1u32;
    let materials = names
        .iter()
        .map(|m| BaseMaterial {
            name: m.as_deref().map(|m| m.strip_prefix("material.").unwrap_or(m)).unwrap_or("default").to_owned(),
            color: srgb_hex(material_appearance(m.as_deref()).1),
        })
        .collect();
    let mut objects = Vec::new();
    let mut next = group + 1;
    let mut shape_object = Vec::new();
    for (si, shape) in scene.shapes.iter().enumerate() {
        let mut ids = Vec::new();
        for (bi, b) in shape.bodies.iter().enumerate() {
            if b.triangles.is_empty() {
                continue;
            }
            let pindex = names.iter().position(|n| *n == b.material).unwrap_or(0) as u32;
            objects.push(Object {
                id: next,
                name: Some(b.name.clone()),
                part_number: shape.part_number.clone(),
                kind: "model".into(),
                material: Some((group, pindex)),
                uuid: Some(uuid(&format!("{path}#shape{si}#body{bi}"))),
                content: Content::Mesh {
                    vertices: b.positions.iter().map(|p| p.map(|v| v * 1e3)).collect(),
                    triangles: b.triangles.clone(),
                },
            });
            ids.push(next);
            next += 1;
        }
        let id = match ids.len() {
            0 => None,
            1 => Some(ids[0]),
            _ => {
                objects.push(Object {
                    id: next,
                    name: Some(shape.name.clone()),
                    part_number: shape.part_number.clone(),
                    kind: "model".into(),
                    material: None,
                    uuid: Some(uuid(&format!("{path}#shape{si}"))),
                    content: Content::Components(
                        ids.iter()
                            .map(|&o| Component {
                                object: o,
                                transform: None,
                                uuid: Some(uuid(&format!("{path}#shape{si}#component{o}"))),
                            })
                            .collect(),
                    ),
                });
                next += 1;
                Some(next - 1)
            }
        };
        shape_object.push(id);
    }
    let items = scene
        .placed
        .iter()
        .enumerate()
        .filter_map(|(k, p)| {
            Some(Item {
                object: shape_object[p.shape]?,
                transform: p.transform.as_ref().map(to_mm),
                part_number: scene.shapes[p.shape].part_number.clone(),
                uuid: Some(uuid(&format!("{path}#item{k}#{}", p.name))),
            })
        })
        .collect();
    Model {
        unit: "millimeter".into(),
        metadata: vec![("Title".into(), title.into()), ("Application".into(), "RunMat".into())],
        materials: vec![(group, materials)],
        objects,
        build_uuid: Some(uuid(&format!("{path}#build"))),
        items,
    }
}

/// The 3MF package of a part or assembly.
pub fn three_mf(scene: &Scene3d, path: &str, title: &str) -> Vec<u8> {
    three_mf::write(&scene_model(scene, path, title))
}
