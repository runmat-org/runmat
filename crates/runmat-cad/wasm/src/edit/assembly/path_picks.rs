//! A path mate's path picked in the assembly: edges of the first pick's component, picked in the
//! view or the tree. The edges become a path that component's part publishes (a new
//! `publish pathN = …` written into the part file, as Publish's Path target writes it), and the
//! mate names that path. A typed name still names a path the part publishes already.

use super::super::publish::publish;
use super::{model, parse_pick, Compiled, Entity};
use crate::features::picks_of;
use crate::runtime::Files;
use crate::xref_repairs::FileEdit;
use runmat_cad::syntax::edit::EditFailure;
use runmat_cad::syntax::parse_part;
use serde_json::{json, Map, Value};

/// When `values` is a path mate whose `path` holds edge picks, publishes those edges from the
/// first pick's part (its change goes in `others`) and answers the values with `path` set to
/// the new path's name; any other values come back as they are.
pub(super) fn publish_picked_path(
    compiled: &Compiled,
    files: &Files,
    values: &Map<String, Value>,
    others: &mut Vec<FileEdit>,
) -> Result<Map<String, Value>, EditFailure> {
    let picks: Vec<String> = picks_of(values, "path").into_iter().filter(|p| p.starts_with("occurrence:")).collect();
    if values.get("type").and_then(Value::as_str) != Some("path") || picks.is_empty() {
        return Ok(values.clone());
    }
    // The path belongs to the first pick's component; editing a mate without a new first pick,
    // to the component the edges are on.
    let first = picks_of(values, "first").into_iter().chain(picks.iter().cloned()).find_map(|p| parse_pick(&p));
    let Some(first) = first else {
        return Err(model("pick the edges the path runs along"));
    };
    let Some((prototype, _)) = compiled.placed(&first.uuid) else {
        return Err(model("the first pick's component isn't in this assembly any more"));
    };
    let mut references = Vec::new();
    for p in &picks {
        let pick = parse_pick(p).ok_or_else(|| model("pick edges for the path"))?;
        let Some(Entity::Edge { body, edge }) = pick.entity else {
            return Err(model("a path runs along edges: pick edges for it"));
        };
        if pick.uuid != first.uuid {
            return Err(model("a path mate's path runs along edges of the first pick's component"));
        }
        let reference = compiled
            .body(prototype, body)
            .and_then(|b| b["edges"].as_array())
            .and_then(|edges| edges.iter().find(|e| e["index"].as_u64() == Some(edge as u64)))
            .and_then(|e| e["reference"].as_str())
            .ok_or_else(|| model("the picked edge has no stable name in its part; pick another edge"))?;
        references.push(reference.to_owned());
    }
    let file = compiled.prototype_file(prototype).ok_or_else(|| model("the component's part file isn't known"))?;
    let text = files.get(&file).ok_or_else(|| model(format!("{file} isn't open to write the path into")))?;
    let parsed = parse_part(&file, text);
    let document = parsed.document.ok_or(EditFailure::Invalid(parsed.diagnostics))?;
    let name = (1..)
        .map(|k| format!("path{k}"))
        .find(|n| !document.publishes.iter().any(|p| p.name.as_str() == n))
        .expect("a free name");
    let request: Map<String, Value> = serde_json::from_value(json!({ "name": name, "target": "path", "edges": references })).expect("an object");
    let (change, _) = publish(&file, text, &request, &crate::picks::Picks::new(&document, &file, text))?;
    others.push((file, text.clone(), change));
    let mut out = values.clone();
    out.insert("path".into(), Value::String(name));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use crate::edit::edit_json;
    use crate::runtime::{compile_assembly, Files};
    use serde_json::{json, Value};

    const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])\n        l2 = line(start = [80 mm, 0 mm], end = [80 mm, 50 mm])\n        l3 = line(start = [80 mm, 50 mm], end = [0 mm, 50 mm])\n        l4 = line(start = [0 mm, 50 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    output plate = { body = feature.body.body }\n}\n";
    const PIN: &str = "runmat 0.7\nstd 1\n\npart pin {\n    sketch s1 on origin.xy {\n        c1 = circle(center = [0 mm, 0 mm], radius = 5 mm)\n    }\n    feature body = extrude(profile = sketch.s1, distance = 30 mm)\n    output pin = { body = feature.body.body }\n}\n";
    const RIG: &str = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part plate as plate\n    use part pin as pin {\n        placement = { placement = offset, offset = { translation = [20 mm, 10 mm, 10 mm] } }\n    }\n}\n";

    fn files() -> Files {
        [("/p/plate.part", PLATE), ("/p/pin.part", PIN)].map(|(p, t)| (p.to_owned(), t.to_owned())).into()
    }

    /// The pick of `path`'s first edge whose points `test` accepts, and of its first face.
    fn picks(display: &Value, index: &Value, path: &str, test: impl Fn(&[f64]) -> bool) -> (String, String) {
        let uuid = index["occurrences"].as_array().unwrap().iter().find(|r| r["path"] == path).unwrap()["uuid"].as_str().unwrap().to_owned();
        let occ = display["occurrences"].as_array().unwrap().iter().find(|o| o["uuid"].as_str() == Some(uuid.as_str())).unwrap();
        let body = &display["prototypes"][occ["prototype"].as_u64().unwrap() as usize]["part"]["bodies"][0];
        let edge = body["edges"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| {
                let p: Vec<f64> = e["points"].as_array().unwrap().iter().filter_map(Value::as_f64).collect();
                test(&p)
            })
            .unwrap_or_else(|| panic!("no such edge on {path}"));
        let face = &body["faces"][0];
        (
            format!("occurrence:{uuid}/edge:{}:{}", body["index"], edge["index"]),
            format!("occurrence:{uuid}/face:{}:{}", body["index"], face["index"]),
        )
    }

    /// Picking the plate's front top edge as the path publishes it from the plate's part and
    /// mates the pin to it by that name; the assembly compiles with the new publish.
    #[test]
    fn a_path_picked_in_the_assembly_is_published_from_its_part() {
        let (index, display, _) = compile_assembly("/p/rig.assembly", RIG, &files()).expect("compiles");
        // The plate's top edge along x at y = 0: every point at z = 10 mm, y = 0.
        let front = |p: &[f64]| p.chunks(3).all(|q| q[1].abs() < 1e-9 && (q[2] - 0.010).abs() < 1e-9);
        let (edge, plate_face) = picks(&display, &index, "plate", front);
        let (_, pin_face) = picks(&display, &index, "pin", |_| true);
        let request = json!({ "command": "assembly.mate", "values": { "type": "path", "first": [plate_face], "second": [pin_face], "path": [edge] } });
        let out: Value = serde_json::from_str(&edit_json("/p/rig.assembly", RIG, &files(), &request.to_string())).unwrap();
        assert_eq!(out["ok"], true, "{out}");
        let changed: Vec<&Value> = out["files"].as_array().unwrap().iter().collect();
        let part = changed.iter().find(|f| f["path"] == "/p/plate.part").unwrap_or_else(|| panic!("the plate isn't written: {out}"));
        let part = part["text"].as_str().unwrap();
        assert!(part.contains("publish path1 = "), "{part}");
        let rig = changed.iter().find(|f| f["path"] == "/p/rig.assembly").unwrap()["text"].as_str().unwrap();
        assert!(rig.contains("path1"), "{rig}");
        let mut with = files();
        with.insert("/p/plate.part".into(), part.to_owned());
        let (_, _, problems) = compile_assembly("/p/rig.assembly", rig, &with).expect("compiles with the path");
        let errors: Vec<_> = problems.iter().filter(|d| d.severity == "error").collect();
        assert!(errors.is_empty(), "{errors:?}\n{rig}\n{part}");
        // An edge of the other component is refused.
        let (pin_edge, _) = picks(&display, &index, "pin", |_| true);
        let request = json!({ "command": "assembly.mate", "values": { "type": "path", "first": [plate_face], "second": [pin_face], "path": [pin_edge] } });
        let out: Value = serde_json::from_str(&edit_json("/p/rig.assembly", RIG, &files(), &request.to_string())).unwrap();
        assert!(out["message"].as_str().unwrap_or("").contains("edges of the first pick's component"), "{out}");
    }
}
