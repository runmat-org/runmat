use crate::edit::edit_json;
use crate::runtime::Files;
use serde_json::{json, Value};

/// A block with a frame layout above it and an L-shaped sketch to save as a profile.
const FRAME: &str = "runmat 0.7\nstd 1\n\npart frame {\n    sketch base on origin.xy {\n        b1 = line(start = [0 mm, 0 mm], end = [100 mm, 0 mm])\n        b2 = line(start = [100 mm, 0 mm], end = [100 mm, 60 mm])\n        b3 = line(start = [100 mm, 60 mm], end = [0 mm, 60 mm])\n        b4 = line(start = [0 mm, 60 mm], end = [0 mm, 0 mm])\n    }\n    feature block = extrude(profile = sketch.base, distance = 10 mm)\n    sketch layout on origin.zx {\n        bottom = line(start = [0 mm, 200 mm], end = [400 mm, 200 mm])\n        right = line(start = [400 mm, 200 mm], end = [400 mm, 500 mm])\n    }\n    sketch ell on origin.xy {\n        e1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        e2 = line(start = [40 mm, 0 mm], end = [40 mm, 5 mm])\n        e3 = line(start = [40 mm, 5 mm], end = [5 mm, 5 mm])\n        e4 = line(start = [5 mm, 5 mm], end = [5 mm, 40 mm])\n        e5 = line(start = [5 mm, 40 mm], end = [0 mm, 40 mm])\n        e6 = line(start = [0 mm, 40 mm], end = [0 mm, 0 mm])\n    }\n}\n";

const FIXTURES: &str = r#"runmat 0.7
std 1

part fixtures {
    feature_def slot = {
        params = [
            { name = width, kind = length, default = 6 mm },
            { name = length, kind = length, default = 20 mm },
            { name = depth, kind = length, default = 3 mm }
        ],
        features = [
            {
                id = s,
                kind = {
                    feature = sketch,
                    plane = origin.xy,
                    entities = [
                        { id = a, geometry = { entity = line, start = { point = coords, x = 10 mm, y = 10 mm }, end = { point = coords, x = input.length, y = 10 mm } } },
                        { id = b, geometry = { entity = line, start = { point = coords, x = input.length, y = 10 mm }, end = { point = coords, x = input.length, y = input.width } } },
                        { id = c, geometry = { entity = line, start = { point = coords, x = input.length, y = input.width }, end = { point = coords, x = 10 mm, y = input.width } } },
                        { id = d, geometry = { entity = line, start = { point = coords, x = 10 mm, y = input.width }, end = { point = coords, x = 10 mm, y = 10 mm } } }
                    ]
                }
            },
            {
                id = cut,
                kind = { feature = extrude, profile = sketch.s, extent = blind(distance = input.depth), reverse = true, mode = cut }
            }
        ]
    }
}
"#;

const PIN: &str = "runmat 0.7\nstd 1\n\npart pin {\n    input d : length = 6 mm\n    sketch s on origin.xy {\n        c = circle(center = [150 mm, 30 mm], radius = input.d)\n    }\n    feature body = extrude(profile = sketch.s, distance = 20 mm)\n}\n";

fn run(text: &str, files: &Files, command: &str, values: Value) -> Value {
    let request = json!({ "command": command, "values": values });
    serde_json::from_str(&edit_json(
        "/p/frame.part",
        text,
        files,
        &request.to_string(),
    ))
    .unwrap()
}

fn own(out: &Value) -> String {
    assert_eq!(out["ok"], true, "{out}");
    out["files"][0]["text"].as_str().unwrap().to_owned()
}

/// The part compiled with `files` in the project.
fn compiled(text: &str, files: &Files) -> Value {
    let mut all = files.clone();
    all.insert("/p/frame.part".into(), text.into());
    let host = crate::runtime::values_host::ValueHost {
        files: all,
        runner: Box::new(
            runmat_cad::value::m_table::TableRunner::new(std::collections::BTreeMap::new(), "t").0,
        ),
    };
    crate::runtime::values_host::with_value_host(host, || {
        serde_json::from_str(&crate::runtime::compile_json("/p/frame.part", text, files)).unwrap()
    })
}

fn errors(c: &Value) -> Vec<String> {
    let mut out = Vec::new();
    for list in [
        &c["diagnostics"],
        &c["projection"]["data"]["index"]["diagnostics"],
    ] {
        for d in list.as_array().into_iter().flatten() {
            if d["severity"] == "error" {
                out.push(d["message"].as_str().unwrap_or("").to_owned());
            }
        }
    }
    out
}

fn cut_list(text: &str, files: &Files) -> Value {
    let request = json!({ "command": "weldment.cut-list", "values": {} }).to_string();
    let mut all = files.clone();
    all.insert("/p/frame.part".into(), text.into());
    serde_json::from_str(&crate::analyze::analyze_json(
        "/p/frame.part",
        text,
        &all,
        &request,
    ))
    .unwrap()
}

#[test]
fn a_standard_size_places_members_with_its_table_section() {
    let files = Files::new();
    let out = run(
        FRAME,
        &files,
        "weldment.profile-library",
        json!({ "profile": "ipe200", "segments": ["sketch:layout"] }),
    );
    let text = own(&out);
    assert!(
        text.contains("profile = standard(designation = \"IPE 200\")"),
        "{text}"
    );
    // IPE 200 by hand: 2·100·8.5 + 183·5.6 + (4 − π)·12² = 2848.4 mm²; Iy 1943 cm⁴ in the table.
    let summary = out["summary"].as_str().unwrap();
    assert!(summary.contains("2 members in ipe200"), "{summary}");
    assert!(summary.contains("A = 28.48 cm²"), "{summary}");
    assert!(summary.contains("Ixx = 194"), "{summary}");
    let c = compiled(&text, &files);
    assert!(errors(&c).is_empty(), "{:?}", errors(&c));
    let list = cut_list(&text, &files).to_string();
    assert!(list.contains("IPE 200"), "{list}");
    // A size the tables don't have is refused with what they hold.
    let out = run(
        FRAME,
        &files,
        "weldment.profile-library",
        json!({ "profile": "IPE 210", "segments": ["sketch:layout"] }),
    );
    assert_eq!(out["ok"], false);
    assert!(
        out["message"]
            .as_str()
            .unwrap()
            .contains("isn't a standard size"),
        "{out}"
    );
}

#[test]
fn a_saved_sketch_profile_is_copied_in_and_its_region_is_the_section() {
    let mut files = Files::new();
    let out = run(
        FRAME,
        &files,
        "weldment.profile-library",
        json!({ "action": "save", "sketch": ["sketch:ell"], "name": "ell_40" }),
    );
    assert_eq!(out["ok"], true, "{out}");
    let file = &out["files"][0];
    assert_eq!(file["path"], "/p/library/profiles/ell_40.part");
    let saved = file["text"].as_str().unwrap().to_owned();
    assert!(saved.contains("sketch profile on origin.xy"), "{saved}");
    // An L 40 × 40 × 5 by hand: 40·5 + 35·5 = 375 mm².
    let summary = out["summary"].as_str().unwrap();
    assert!(summary.contains("A = 3.750 cm²"), "{summary}");
    files.insert("/p/library/profiles/ell_40.part".into(), saved);
    // Saving the same name again is refused.
    let again = run(
        FRAME,
        &files,
        "weldment.profile-library",
        json!({ "action": "save", "sketch": ["sketch:ell"], "name": "ell_40" }),
    );
    assert_eq!(again["ok"], false);
    // Placing it copies the sketch in under the profile's name.
    let out = run(
        FRAME,
        &files,
        "weldment.profile-library",
        json!({ "profile": "ell_40", "segments": ["sketch.layout.bottom"], "material": "al_6061" }),
    );
    let text = own(&out);
    assert!(text.contains("sketch ell_40 on origin.xy"), "{text}");
    assert!(
        text.contains("profile = sketch_region(sketch = ell_40)"),
        "{text}"
    );
    let c = compiled(&text, &files);
    assert!(errors(&c).is_empty(), "{:?}", errors(&c));
    let list = cut_list(&text, &files).to_string();
    assert!(list.contains("Profile ell_40 (3.75 cm²)"), "{list}");
    // A sketch of this part works directly.
    let out = run(
        FRAME,
        &files,
        "weldment.profile-library",
        json!({ "profile": "sketch.ell", "segments": ["sketch.layout.right"] }),
    );
    let text = own(&out);
    assert!(
        text.contains("profile = sketch_region(sketch = ell)"),
        "{text}"
    );
    assert!(errors(&compiled(&text, &files)).is_empty());
}

#[test]
fn a_library_feature_lands_on_the_picked_face_with_its_inputs() {
    let mut files = Files::new();
    files.insert("/p/library/fixtures.part".into(), FIXTURES.into());
    let out = run(
        FRAME,
        &files,
        "part.library",
        json!({
            "definition": "library/fixtures.part#feature_def.slot",
            "placement": ["feature.block.face.end"],
            "args": "width = 20 mm, depth = 4 mm"
        }),
    );
    let text = own(&out);
    assert!(
        text.contains("library/fixtures.part#feature_def.slot"),
        "{text}"
    );
    assert!(text.contains("width = 20 mm"), "{text}");
    assert!(text.contains("plane = feature.block.face.end"), "{text}");
    let c = compiled(&text, &files);
    assert!(errors(&c).is_empty(), "{:?}", errors(&c));
    // The cut ran: the expanded features are in the index.
    let features = c["projection"]["data"]["index"]["features"].to_string();
    assert!(features.contains("slot1.cut"), "{features}");
    // The panel's search lists the definition, the standard sizes and no profile yet.
    let library = c["projection"]["data"]["index"]["library"].to_string();
    assert!(
        library.contains("library/fixtures.part#feature_def.slot"),
        "{library}"
    );
    assert!(library.contains("HEB 300"), "{library}");
    // An input the definition lacks is refused, naming the ones it has.
    let out = run(
        FRAME,
        &files,
        "part.library",
        json!({ "definition": "library/fixtures.part#feature_def.slot", "args": "radius = 2 mm" }),
    );
    assert_eq!(out["ok"], false);
    assert!(
        out["message"]
            .as_str()
            .unwrap()
            .contains("width, length, depth"),
        "{out}"
    );
}

#[test]
fn a_library_part_inserts_as_a_body() {
    let mut files = Files::new();
    files.insert("/p/library/pin.part".into(), PIN.into());
    let out = run(
        FRAME,
        &files,
        "part.library",
        json!({ "definition": "library/pin.part", "args": "d = 4 mm" }),
    );
    let text = own(&out);
    assert!(text.contains("feature pin1 = library("), "{text}");
    let c = compiled(&text, &files);
    assert!(errors(&c).is_empty(), "{:?}", errors(&c));
    let bodies = c["projection"]["data"]["index"]["bodies"]
        .as_array()
        .unwrap()
        .len();
    assert_eq!(bodies, 2, "the block and the pin");
    // A part takes no placement.
    let out = run(
        FRAME,
        &files,
        "part.library",
        json!({ "definition": "library/pin.part", "placement": ["origin.xy"] }),
    );
    assert_eq!(out["ok"], false);
    // A library file that isn't in the project is reported by the compile.
    let c = compiled(&text, &Files::new());
    let all = c.to_string();
    assert!(
        all.contains("library/pin.part isn't in the project"),
        "{all}"
    );
}
