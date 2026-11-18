//! Interface check, hole alignment and the hole/fastener check on small assemblies.

use super::*;

/// A 60 × 40 × 10 mm plate with a Ø6 mm hole through it at (`x`, 20) mm, and its top face
/// published as `top`.
fn plate(x: f64) -> String {
    format!(
        "runmat 0.7\nstd 1\n\npart plate {{\n    sketch s1 on origin.xy {{\n        l1 = line(start = [0 mm, 0 mm], end = [60 mm, 0 mm])\n        l2 = line(start = [60 mm, 0 mm], end = [60 mm, 40 mm])\n        l3 = line(start = [60 mm, 40 mm], end = [0 mm, 40 mm])\n        l4 = line(start = [0 mm, 40 mm], end = [0 mm, 0 mm])\n    }}\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    feature bore = hole(face = feature.body.face.end, at = {{ a = point({x} mm, 20 mm, 10 mm) }}, diameter = 6 mm, through_all)\n    output plate = {{ body = feature.body.body }}\n    region deck : face = {{ select = [feature.body.face.end], expect = exactly_one }}\n    publish top = region(region = deck)\n}}\n"
    )
}

const STACK: &str = "runmat 0.7\nstd 1\n\nassembly stack {\n    use part plate as lower\n    use part upper as upper {\n        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 10 mm] } }\n    }\n}\n";

const MATED: &str = "runmat 0.7\nstd 1\n\nassembly stack {\n    use part plate as lower\n    use part upper as upper {\n        fixed = false\n        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 12 mm] } }\n    }\n\n    connector deck = { occurrence = [{ name = lower }], anchor = { anchor = published, name = top } }\n    connector seat = { occurrence = [{ name = upper }] }\n\n    mate rest = { kind = { mate = fasten }, a = deck, b = seat }\n}\n";

fn files(upper_x: f64) -> Files {
    let upper = plate(upper_x)
        .replace("part plate", "part upper")
        .replace("output plate", "output upper");
    [("/p/plate.part", plate(30.0)), ("/p/upper.part", upper)]
        .map(|(p, t)| (p.to_owned(), t))
        .into()
}

fn run(text: &str, files: &Files, command: &str, values: Value) -> Value {
    let out: Value = serde_json::from_str(&analyze_json(
        "/p/stack.assembly",
        text,
        files,
        &json!({ "command": command, "values": values }).to_string(),
    ))
    .unwrap();
    assert_eq!(out["ok"], true, "{out}");
    out
}

#[test]
fn holes_that_line_up_are_aligned_and_an_offset_one_is_listed_with_its_dimension() {
    let out = run(
        STACK,
        &files(30.0),
        "assembly.hole-alignment",
        json!({ "tolerance": 0.1 }),
    );
    assert_eq!(out["data"]["pairs"], 1, "{out}");
    assert_eq!(out["data"]["misaligned"], 0, "{out}");
    assert!(
        out["rows"][0]["cells"][3]
            .as_str()
            .unwrap()
            .starts_with("Aligned"),
        "{out}"
    );
    assert!(
        out["dimensions"].as_array().is_none_or(Vec::is_empty),
        "{out}"
    );

    let out = run(
        STACK,
        &files(30.5),
        "assembly.hole-alignment",
        json!({ "tolerance": 0.1 }),
    );
    assert_eq!(out["data"]["misaligned"], 1, "{out}");
    let row = &out["rows"][0];
    assert_eq!(row["severity"], "error", "{out}");
    assert_eq!(row["cells"][2], "0.50 mm", "{out}");
    // The holes' faces are the row's items, so selecting it shows them.
    assert!(
        row["items"]
            .as_array()
            .unwrap()
            .iter()
            .any(|i| i.as_str().unwrap().contains("/face:")),
        "{out}"
    );
    assert_eq!(out["dimensions"][0]["label"], "0.50 mm", "{out}");

    // A looser tolerance accepts it.
    let out = run(
        STACK,
        &files(30.5),
        "assembly.hole-alignment",
        json!({ "tolerance": 1.0 }),
    );
    assert_eq!(out["data"]["misaligned"], 0, "{out}");
}

#[test]
fn hole_check_lists_alignment_and_says_when_there_are_no_fasteners() {
    let out = run(STACK, &files(30.5), "assembly.hole-check", json!({}));
    assert_eq!(out["title"], "Hole/fastener check");
    assert_eq!(out["data"]["misaligned"], 1, "{out}");
    assert_eq!(out["data"]["fasteners"], 0, "{out}");
    assert!(
        out["summary"]
            .as_str()
            .unwrap()
            .contains("0 fasteners checked"),
        "{out}"
    );
}

#[test]
fn components_limit_the_pairs() {
    let c = crate::runtime::compile_assembly("/p/stack.assembly", STACK, &files(30.5)).unwrap();
    let uuid = |name: &str| {
        c.0["occurrences"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["path"] == name)
            .unwrap()["uuid"]
            .as_str()
            .unwrap()
            .to_owned()
    };
    let lower = format!("occurrence:{}", uuid("lower"));
    let out = run(
        STACK,
        &files(30.5),
        "assembly.hole-alignment",
        json!({ "components": [lower] }),
    );
    assert_eq!(out["data"]["pairs"], 1, "{out}");
    let out = run(
        STACK,
        &files(30.5),
        "assembly.hole-alignment",
        json!({ "components": ["occurrence:00000000000000000000000000000000"] }),
    );
    assert_eq!(out["data"]["pairs"], 0, "{out}");
}

#[test]
fn interface_check_passes_a_mate_on_a_published_connector_and_flags_it_when_it_goes() {
    let f = files(30.0);
    let out = run(MATED, &f, "assembly.interface-check", json!({}));
    let rows = out["rows"].as_array().unwrap();
    let mate = rows.iter().find(|r| r["cells"][1] == "fasten").unwrap();
    assert_eq!(mate["cells"][3], "Solved", "{out}");
    let deck = rows
        .iter()
        .find(|r| r["cells"][1] == "connector deck")
        .unwrap();
    assert!(deck["severity"].is_null(), "{out}");
    assert!(
        deck["cells"][2].as_str().unwrap().contains("top (plane)"),
        "{out}"
    );
    assert_eq!(deck["cells"][3], "Matches", "{out}");
    let seat = rows
        .iter()
        .find(|r| r["cells"][1] == "connector seat")
        .unwrap();
    assert_eq!(seat["severity"], "info", "{out}");
    assert_eq!(out["data"]["errors"], 0, "{out}");

    // The plate's connector renamed: the mate's interface is gone.
    let mut broken = f.clone();
    let text = broken["/p/plate.part"].replace("publish top = ", "publish lid = ");
    broken.insert("/p/plate.part".into(), text);
    let out = run(MATED, &broken, "assembly.interface-check", json!({}));
    let rows = out["rows"].as_array().unwrap();
    let deck = rows
        .iter()
        .find(|r| r["cells"][1] == "connector deck")
        .unwrap();
    assert_eq!(deck["severity"], "error", "{out}");
    assert!(
        deck["cells"][3]
            .as_str()
            .unwrap()
            .contains("no longer publishes `top`"),
        "{out}"
    );
    assert!(out["summary"].as_str().unwrap().contains("broken"), "{out}");
}

#[test]
fn interface_check_lists_what_the_assembly_publishes() {
    let text = MATED.replace(
        "    mate rest",
        "    publish mount = { target = { publishes = connector, connector = deck } }\n    publish gone = { target = { publishes = connector, connector = nowhere } }\n\n    mate rest",
    );
    let out = run(&text, &files(30.0), "assembly.interface-check", json!({}));
    let rows = out["rows"].as_array().unwrap();
    let mount = rows.iter().find(|r| r["cells"][0] == "mount").unwrap();
    assert!(mount["severity"].is_null(), "{out}");
    let gone = rows.iter().find(|r| r["cells"][0] == "gone").unwrap();
    assert_eq!(gone["severity"], "error", "{out}");
}

#[test]
fn the_checks_are_for_assemblies() {
    let out: Value = serde_json::from_str(&analyze_json(
        "/p/plate.part",
        &plate(30.0),
        &Files::new(),
        &json!({ "command": "assembly.hole-check" }).to_string(),
    ))
    .unwrap();
    assert_eq!(out["ok"], false);
}
