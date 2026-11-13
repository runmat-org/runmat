//! Mates anchored to what the parts publish: a face mate follows its part through an edit that
//! renumbers the faces, and vertex, datum and mate-connector picks solve.
use super::frames::{datum_frame, face_frame, to_world, Frame, FrameKind};
use crate::edit::edit_json;
use crate::runtime::{compile_assembly, Files};
use serde_json::{json, Value};

/// A plate `width` mm long; `split` draws its front edge as two lines, which adds a side face
/// and renumbers the faces after it.
fn plate(width: f64, split: bool) -> String {
    let (w, h) = (width, width / 2.0);
    let front = if split {
        format!("        l1 = line(start = [0 mm, 0 mm], end = [{h} mm, 0 mm])\n        l1b = line(start = [{h} mm, 0 mm], end = [{w} mm, 0 mm])\n")
    } else {
        format!("        l1 = line(start = [0 mm, 0 mm], end = [{w} mm, 0 mm])\n")
    };
    let sketch = format!(
        "{front}        l2 = line(start = [{w} mm, 0 mm], end = [{w} mm, 50 mm])\n        l3 = line(start = [{w} mm, 50 mm], end = [0 mm, 50 mm])\n        l4 = line(start = [0 mm, 50 mm], end = [0 mm, 0 mm])\n"
    );
    format!("runmat 0.7\nstd 1\n\npart plate {{\n    sketch s1 on origin.xy {{\n{sketch}    }}\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    output plate = {{ body = feature.body.body }}\n}}\n")
}

const BLOCK: &str = "runmat 0.7\nstd 1\n\npart block {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])\n        l2 = line(start = [20 mm, 0 mm], end = [20 mm, 20 mm])\n        l3 = line(start = [20 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 15 mm)\n    feature mc = mate_connector(\n        placement = { on = frame, frame = { frame = on_face, face = feature.body.face.start } }\n    )\n    output block = { body = feature.body.body }\n}\n";

const RIG: &str = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part plate as plate\n    use part block as block {\n        placement = { placement = offset, offset = { translation = [120 mm, 10 mm, 40 mm] } }\n    }\n}\n";

const TOP: &str = "/p/rig.assembly";

fn files(plate_text: &str) -> Files {
    [("/p/plate.part", plate_text), ("/p/block.part", BLOCK)]
        .map(|(p, t)| (p.to_owned(), t.to_owned()))
        .into()
}

struct Compiled {
    index: Value,
    display: Value,
}

fn compile(text: &str, files: &Files) -> Compiled {
    let (index, display, problems) = compile_assembly(TOP, text, files)
        .unwrap_or_else(|d| panic!("doesn't compile: {d:?}\n{text}"));
    let errors: Vec<_> = problems.iter().filter(|d| d.severity == "error").collect();
    assert!(errors.is_empty(), "{errors:?}\n{text}\n{files:?}");
    Compiled { index, display }
}

impl Compiled {
    fn uuid(&self, path: &str) -> String {
        self.index["occurrences"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["path"] == path)
            .unwrap()["uuid"]
            .as_str()
            .unwrap()
            .to_owned()
    }
    fn occurrence(&self, path: &str) -> &Value {
        let uuid = self.uuid(path);
        self.display["occurrences"]
            .as_array()
            .unwrap()
            .iter()
            .find(|o| o["uuid"].as_str() == Some(uuid.as_str()))
            .unwrap()
    }
    fn transform(&self, path: &str) -> Vec<f64> {
        self.occurrence(path)["transform"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    }
    fn part(&self, path: &str) -> &Value {
        let prototype = self.occurrence(path)["prototype"].as_u64().unwrap() as usize;
        &self.display["prototypes"][prototype]["part"]
    }
    /// The face of `path` whose frame `test` accepts: its index and pick ID.
    fn face(&self, path: &str, test: impl Fn(&Frame) -> bool) -> (u64, String) {
        let body = &self.part(path)["bodies"][0];
        let face = body["faces"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| face_frame(f).is_some_and(|frame| test(&frame)))
            .unwrap_or_else(|| panic!("no such face on {path}"));
        let index = face["index"].as_u64().unwrap();
        (
            index,
            format!(
                "occurrence:{}/face:{}:{index}",
                self.uuid(path),
                body["index"]
            ),
        )
    }
    /// The vertex of `path` at `at` (metres, part coordinates): its pick, and whether it has a
    /// stable name.
    fn vertex(&self, path: &str, at: [f64; 3]) -> (String, bool) {
        let body = &self.part(path)["bodies"][0];
        let v = body["vertices"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| (0..3).all(|k| (v["point"][k].as_f64().unwrap() - at[k]).abs() < 1e-6))
            .unwrap_or_else(|| panic!("no vertex at {at:?} on {path}"));
        (
            format!(
                "occurrence:{}/vertex:{}:{}",
                self.uuid(path),
                body["index"],
                v["index"]
            ),
            v["reference"].is_string(),
        )
    }
    /// Datum `feature` of `path`'s part in assembly coordinates.
    fn datum(&self, path: &str, feature: &str) -> Frame {
        let r = self.part(path)["references"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["id"] == feature)
            .unwrap_or_else(|| panic!("{path} draws no {feature}"));
        to_world(datum_frame(r).unwrap(), &self.transform(path))
    }
}

/// Runs `request` on the assembly with `files`: the assembly's text, with every part file the
/// edit changed written into `files`.
fn run(text: &str, files: &mut Files, request: Value) -> String {
    let out: Value =
        serde_json::from_str(&edit_json(TOP, text, files, &request.to_string())).unwrap();
    assert_eq!(out["ok"], true, "{out}");
    let changed = out["files"].as_array().unwrap();
    for f in &changed[1..] {
        files.insert(
            f["path"].as_str().unwrap().to_owned(),
            f["text"].as_str().unwrap().to_owned(),
        );
    }
    changed[0]["text"].as_str().unwrap().to_owned()
}

fn mate(text: &str, files: &mut Files, kind: &str, first: String, second: String) -> String {
    run(
        text,
        files,
        json!({ "command": "assembly.mate", "values": { "type": kind, "first": [first], "second": [second] } }),
    )
}

fn plus_x(f: &Frame) -> bool {
    f.kind == FrameKind::Plane && f.z[0] > 0.99
}
fn minus_x(f: &Frame) -> bool {
    f.kind == FrameKind::Plane && f.z[0] < -0.99
}
fn bottom(f: &Frame) -> bool {
    f.kind == FrameKind::Plane && f.z[2] < -0.99
}

/// The plate's +X side published by the mate keeps the block on it after the plate grows from
/// 80 to 100 mm and its front edge is split in two, which renumbers that face.
#[test]
fn a_face_mate_follows_its_part_when_the_faces_renumber() {
    let mut files = files(&plate(80.0, false));
    let c = compile(RIG, &files);
    let (before, side) = c.face("plate", plus_x);
    let text = mate(
        RIG,
        &mut files,
        "coincident",
        side,
        c.face("block", minus_x).1,
    );
    let part = &files["/p/plate.part"];
    assert!(part.contains("publish mate_face1 = "), "{part}");
    assert!(text.contains("published"), "{text}");
    assert!(
        (compile(&text, &files).transform("block")[9] - 0.080).abs() < 1e-6,
        "{text}"
    );
    // The plate edited: 100 mm long, its front edge in two lines.
    let grown = plate(100.0, true);
    let published = part
        .lines()
        .filter(|l| l.contains("publish "))
        .collect::<Vec<_>>()
        .join("\n");
    let edited = grown.replace(
        "    output plate",
        &format!("{published}\n    output plate"),
    );
    files.insert("/p/plate.part".into(), edited.clone());
    let after = compile(&text, &files);
    let (now, _) = after.face("plate", plus_x);
    assert_ne!(
        before, now,
        "the edit should renumber the +X face\n{edited}"
    );
    let t = after.transform("block");
    assert!(
        (t[9] - 0.100).abs() < 1e-6,
        "the block stays on the moved face: {t:?}\n{edited}"
    );
}

/// A vertex on each part: a coincident mate makes a ball joint that puts the block's corner on
/// the plate's.
#[test]
fn vertex_picks_mate_point_to_point() {
    let mut files = files(&plate(80.0, false));
    let c = compile(RIG, &files);
    let (corner, named) = c.vertex("plate", [0.08, 0.05, 0.01]);
    let text = mate(
        RIG,
        &mut files,
        "coincident",
        corner,
        c.vertex("block", [0.0, 0.0, 0.0]).0,
    );
    assert!(text.contains("mate = ball"), "{text}");
    // A vertex with a stable name is published; one without (the test kernel names no
    // vertices) keeps its frame in the part's coordinates.
    assert_eq!(
        files["/p/plate.part"].contains("kind = vertex"),
        named,
        "{}",
        files["/p/plate.part"]
    );
    let t = compile(&text, &files).transform("block");
    assert!(
        (t[9] - 0.08).abs() < 1e-6 && (t[10] - 0.05).abs() < 1e-6 && (t[11] - 0.01).abs() < 1e-6,
        "{t:?}\n{text}"
    );
}

/// A datum plane of the plate (25 mm above its base) takes the block's bottom face, and the
/// part's own mate connectors fasten the block to the plate.
#[test]
fn datum_and_connector_picks_solve() {
    let mut files = files(&plate(80.0, false));
    // The plate gets a datum plane 25 mm up, and a mate connector on its top face.
    let plane = json!({ "command": "part.plane", "values": { "type": "offset", "base": ["origin:top"], "distance": 25 } });
    let out: Value = serde_json::from_str(&edit_json(
        "/p/plate.part",
        &files["/p/plate.part"],
        &files,
        &plane.to_string(),
    ))
    .unwrap();
    assert_eq!(out["ok"], true, "{out}");
    let with_plane = out["files"][0]["text"].as_str().unwrap().to_owned();
    let with_connector = with_plane.replace(
        "    output plate",
        "    feature top = mate_connector(\n        placement = { on = frame, frame = { frame = on_face, face = feature.body.face.end } }\n    )\n    output plate",
    );
    files.insert("/p/plate.part".into(), with_connector);
    let c = compile(RIG, &files);
    let uuid = |p: &str| c.uuid(p);
    let mut planar = files.clone();
    let text = mate(
        RIG,
        &mut planar,
        "coincident",
        format!("occurrence:{}/datum:plane1", uuid("plate")),
        c.face("block", bottom).1,
    );
    assert!(
        planar["/p/plate.part"].contains("publish plane1 = "),
        "{}",
        planar["/p/plate.part"]
    );
    assert!(text.contains("mate = planar"), "{text}");
    let t = compile(&text, &planar).transform("block");
    assert!((t[11] - 0.025).abs() < 1e-6, "{t:?}\n{text}");
    // The two parts' mate connectors, fastened: the block's lands on the plate's.
    let text = mate(
        RIG,
        &mut files,
        "fasten",
        format!("occurrence:{}/datum:top", uuid("plate")),
        format!("occurrence:{}/datum:mc", uuid("block")),
    );
    assert!(
        files["/p/block.part"].contains("publish mc = "),
        "{}",
        files["/p/block.part"]
    );
    let after = compile(&text, &files);
    let (a, b) = (after.datum("plate", "top"), after.datum("block", "mc"));
    for k in 0..3 {
        assert!(
            (a.origin[k] - b.origin[k]).abs() < 1e-6,
            "{a:?} {b:?}\n{text}"
        );
    }
    // A coincident mate between two connectors fastens them too.
    let mut again = files.clone();
    let text = mate(
        RIG,
        &mut again,
        "coincident",
        format!("occurrence:{}/datum:top", uuid("plate")),
        format!("occurrence:{}/datum:mc", uuid("block")),
    );
    assert!(text.contains("mate = fasten"), "{text}");
}
