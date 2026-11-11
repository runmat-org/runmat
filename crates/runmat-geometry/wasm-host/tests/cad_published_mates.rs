//! Mates on OCCT anchored to what the parts publish: a pin mated concentric to a plate's hole
//! follows the hole when the plate is edited (the hole moves, and the front edge split in two
//! renumbers the plate's faces).

use serde_json::{json, Value};

fn call(function: &str, args: &[&str]) -> Value {
    let request = json!({ "fn": function, "args": args });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

/// A 40 x 20 x 10 mm plate with a 6 mm hole through it at (`x`, 10 mm); `split` draws its front
/// edge as two lines, which adds a side face and renumbers the faces after it.
fn plate(x: f64, split: bool) -> String {
    let front = if split {
        "        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])\n        l1b = line(start = [20 mm, 0 mm], end = [40 mm, 0 mm])\n"
    } else {
        "        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n"
    };
    let rest = "        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n";
    let circle = format!("        c1 = circle(center = [{x} mm, 10 mm], radius = 3 mm)\n");
    format!("runmat 0.7\nstd 1\n\npart plate {{\n    sketch s1 on origin.xy {{\n{front}{rest}{circle}    }}\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    output plate = {{ body = feature.body.body }}\n}}\n")
}

const PIN: &str = "runmat 0.7\nstd 1\n\npart pin {\n    sketch s1 on origin.xy {\n        c1 = circle(center = [0 mm, 0 mm], radius = 3 mm)\n    }\n    feature body = extrude(profile = sketch.s1, distance = 30 mm)\n    output pin = { body = feature.body.body }\n}\n";

const RIG: &str = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part plate as plate\n    use part pin as pin {\n        placement = { placement = offset, offset = { translation = [60 mm, 30 mm, 25 mm] } }\n    }\n}\n";

fn compile(text: &str, files: &Value) -> Value {
    let out = call(
        "compile",
        &["/p/rig.assembly", text, &files.to_string(), "{}"],
    );
    let errors: Vec<&Value> = out["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|d| d["severity"] == "error")
        .collect();
    assert!(errors.is_empty(), "{errors:?}\n{text}\n{files}");
    out
}

fn uuid(out: &Value, path: &str) -> String {
    out["projection"]["data"]["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["path"] == path)
        .unwrap()["uuid"]
        .as_str()
        .unwrap()
        .to_owned()
}

fn occurrence<'a>(out: &'a Value, path: &str) -> &'a Value {
    let id = uuid(out, path);
    out["display"]["data"]["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .find(|o| o["uuid"].as_str() == Some(id.as_str()))
        .unwrap()
}

/// The round face of `path` whose points all lie `radius` from (`cx`, `cy`): its index and pick.
fn round_face(out: &Value, path: &str, cx: f64, cy: f64, radius: f64) -> (u64, String) {
    let prototype = occurrence(out, path)["prototype"].as_u64().unwrap() as usize;
    let body = &out["display"]["data"]["prototypes"][prototype]["part"]["bodies"][0];
    let face = body["faces"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| {
            let p: Vec<f64> = f["positions"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(Value::as_f64)
                .collect();
            p.chunks(3)
                .all(|q| ((q[0] - cx).hypot(q[1] - cy) - radius).abs() < 1e-5)
        })
        .unwrap_or_else(|| panic!("no round face on {path}"));
    let index = face["index"].as_u64().unwrap();
    (
        index,
        format!(
            "occurrence:{}/face:{}:{index}",
            uuid(out, path),
            body["index"]
        ),
    )
}

#[test]
fn a_concentric_mate_follows_the_hole_when_the_plate_is_edited() {
    let mut files = json!({ "/p/plate.part": plate(10.0, false), "/p/pin.part": PIN });
    let before = compile(RIG, &files);
    let (hole_index, hole) = round_face(&before, "plate", 0.010, 0.010, 0.003);
    let (_, wall) = round_face(&before, "pin", 0.0, 0.0, 0.003);
    let request = json!({ "command": "assembly.mate", "values": { "type": "concentric", "first": [hole], "second": [wall] } });
    let out = call(
        "edit",
        &[
            "/p/rig.assembly",
            RIG,
            &files.to_string(),
            &request.to_string(),
        ],
    );
    assert_eq!(out["ok"], true, "{out}");
    let changed = out["files"].as_array().unwrap();
    let text = changed[0]["text"].as_str().unwrap().to_owned();
    for f in &changed[1..] {
        files[f["path"].as_str().unwrap()] = f["text"].clone();
    }
    let part = files["/p/plate.part"].as_str().unwrap().to_owned();
    assert!(
        part.contains("publish mate_face1 = ") && part.contains("kind = face"),
        "{part}"
    );
    let t = |out: &Value| -> Vec<f64> {
        occurrence(out, "pin")["transform"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    };
    let mated = t(&compile(&text, &files));
    assert!(
        (mated[9] - 0.010).abs() < 1e-6 && (mated[10] - 0.010).abs() < 1e-6,
        "{mated:?}\n{text}"
    );
    // The hole moves to x = 30 mm and the front edge splits in two: the hole's face renumbers.
    let published: String = part
        .lines()
        .filter(|l| l.contains("publish "))
        .map(|l| format!("{l}\n"))
        .collect();
    let edited =
        plate(30.0, true).replace("    output plate", &format!("{published}    output plate"));
    files["/p/plate.part"] = Value::String(edited.clone());
    let after = compile(&text, &files);
    let (now, _) = round_face(&after, "plate", 0.030, 0.010, 0.003);
    assert_ne!(
        hole_index, now,
        "the edit should renumber the hole's face\n{edited}"
    );
    let moved = t(&after);
    assert!(
        (moved[9] - 0.030).abs() < 1e-6 && (moved[10] - 0.010).abs() < 1e-6,
        "the pin follows the hole: {moved:?}\n{edited}"
    );
}
