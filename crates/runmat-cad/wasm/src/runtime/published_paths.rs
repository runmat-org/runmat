//! Edge sets a part publishes, as paths an assembly reads: a cam's profile, a path mate's
//! track. A publish of edges (`entities` of kind edge) or of an edge region becomes the edges'
//! display polylines, joined end to end into chains; a closed loop's chain ends where it starts.

use super::RegeneratedPart;
use runmat_cad::assembly::source::{PathChain, PublishedGeometry};
use runmat_cad::ids::PublishName;
use runmat_cad::refs::name::TopoKind;
use runmat_cad::refs::resolve::EntityHandle;
use std::collections::BTreeMap;

/// Ends closer than this (metres) join.
const JOIN: f64 = 1e-6;

/// Every published edge set of `part` as a path.
pub(super) fn published_paths(part: &RegeneratedPart) -> BTreeMap<PublishName, PublishedGeometry> {
    let mut out = BTreeMap::new();
    for (name, target) in &part.result.publishes {
        let Ok(target) = target else { continue };
        if target.entities.is_empty() || target.entities.iter().any(|e| e.kind != TopoKind::Edge) {
            continue;
        }
        let mut pieces: Vec<Vec<[f64; 3]>> = Vec::new();
        for (b, body) in part.result.state.bodies.iter().enumerate() {
            let Some(display) = part.display.bodies.iter().find(|d| d.index as usize == b) else {
                continue;
            };
            for e in 0..body.topology.edges.len() {
                let names = part.result.state.entity_names(&EntityHandle {
                    body: b as u32,
                    kind: TopoKind::Edge,
                    index: e as u32,
                });
                if names.is_empty() || !target.entities.iter().any(|t| t.names == names) {
                    continue;
                }
                let Some(edge) = display.edges.iter().find(|d| d.index as usize == e) else {
                    continue;
                };
                let points: Vec<[f64; 3]> = edge
                    .points
                    .chunks_exact(3)
                    .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
                    .collect();
                if points.len() >= 2 {
                    pieces.push(points);
                }
            }
        }
        if pieces.is_empty() {
            continue;
        }
        let chains = join(pieces)
            .into_iter()
            .enumerate()
            .map(|(i, points)| PathChain {
                key: format!("{name}.{i}"),
                points,
                normals: Vec::new(),
            })
            .collect();
        out.insert(name.clone(), PublishedGeometry::Path { chains });
    }
    out
}

fn near(a: [f64; 3], b: [f64; 3]) -> bool {
    (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt() < JOIN
}

/// Polylines joined end to end, each turned as it needs to be.
pub(super) fn join(mut pieces: Vec<Vec<[f64; 3]>>) -> Vec<Vec<[f64; 3]>> {
    let mut chains = Vec::new();
    while let Some(mut chain) = pieces.pop() {
        loop {
            let (head, tail) = (chain[0], chain[chain.len() - 1]);
            let Some(at) = pieces.iter().position(|p| {
                let (s, e) = (p[0], p[p.len() - 1]);
                near(s, tail) || near(e, tail) || near(s, head) || near(e, head)
            }) else {
                break;
            };
            let mut p = pieces.remove(at);
            let (s, e) = (p[0], p[p.len() - 1]);
            if near(s, tail) {
                chain.extend(p.drain(1..));
            } else if near(e, tail) {
                p.reverse();
                chain.extend(p.drain(1..));
            } else {
                if near(s, head) {
                    p.reverse();
                }
                p.pop();
                p.extend(chain);
                chain = p;
            }
        }
        chains.push(chain);
    }
    chains
}

#[cfg(test)]
mod tests {
    use super::{join, published_paths};
    use runmat_cad::assembly::source::PublishedGeometry;
    use serde_json::{json, Value};

    const PIN: &str = "runmat 0.7\nstd 1\n\npart pin {\n    sketch s1 on origin.xy {\n        c1 = circle(center = [0 mm, 0 mm], radius = 5 mm)\n    }\n    feature body = extrude(profile = sketch.s1, distance = 30 mm)\n    output pin = { body = feature.body.body }\n}\n";

    /// Publish's Path target writes the picked edges, and the part publishes them as one
    /// closed path: the pin's top rim, 5 mm round its axis at 30 mm.
    #[test]
    fn publish_writes_a_path_of_picked_edges() {
        let rim = "edges_between(feature.body.face.side(s1.c1), feature.body.face.end)";
        let request = json!({ "command": "simulate.publish", "values": { "name": "rim", "target": "path", "edges": [rim] } });
        let out: Value = serde_json::from_str(&crate::edit::edit_json(
            "/p/pin.part",
            PIN,
            &Default::default(),
            &request.to_string(),
        ))
        .unwrap();
        assert_eq!(out["ok"], true, "{out}");
        let text = out["files"][0]["text"].as_str().unwrap();
        assert!(text.contains("publish rim = "), "{text}");
        let part = crate::runtime::regenerate_part_with(
            "/p/pin.part",
            text,
            &crate::runtime::ASSEMBLY_TESSELLATION,
        )
        .expect("regenerates");
        let paths = published_paths(&part);
        let Some(PublishedGeometry::Path { chains }) =
            paths.get(&runmat_cad::ids::PublishName::new("rim"))
        else {
            panic!("no rim path: {paths:?}");
        };
        assert_eq!(chains.len(), 1, "{chains:?}");
        let points = &chains[0].points;
        assert!(points.len() > 8);
        for p in points {
            assert!(
                (p[0].hypot(p[1]) - 0.005).abs() < 1e-5 && (p[2] - 0.030).abs() < 1e-6,
                "{p:?}"
            );
        }
        let (a, b) = (points[0], points[points.len() - 1]);
        assert!(
            (0..3).all(|k| (a[k] - b[k]).abs() < 1e-6),
            "the loop closes: {a:?} {b:?}"
        );
        // A second publish under the same name is refused.
        let out: Value = serde_json::from_str(&crate::edit::edit_json(
            "/p/pin.part",
            text,
            &Default::default(),
            &request.to_string(),
        ))
        .unwrap();
        assert!(
            out["message"]
                .as_str()
                .unwrap_or("")
                .contains("already publishes `rim`"),
            "{out}"
        );
    }

    #[test]
    fn pieces_join_end_to_end_in_either_direction() {
        let a = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let b = vec![[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
        let c = vec![[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
        let chains = join(vec![a, b, c]);
        assert_eq!(chains.len(), 1);
        let chain = &chains[0];
        assert_eq!(chain.len(), 4, "{chain:?}");
        assert_eq!(chain[0], chain[3], "a closed loop ends where it starts");
    }
}
