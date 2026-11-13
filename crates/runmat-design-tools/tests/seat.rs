//! `design.seat_part`: one call inserts a part, writes its connectors and fasten, solves, and
//! answers with the pose, the interference and the clearance. A block seats up under a plate
//! by a ray, seats again on the plate's top face turned along Y (replacing the first seat),
//! and every refusal names the pick. Gaps and overlaps need exact proximity, which the grid test
//! kernel lacks: the CAD host's cad_design_tools tests measure them on OCCT.

use runmat_design_tools::{call, Files};
use serde_json::{json, Value};

const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])\n        l2 = line(start = l1.end, end = [80 mm, 50 mm])\n        l3 = line(start = l2.end, end = [0 mm, 50 mm])\n        l4 = line(start = l3.end, end = l1.start)\n    }\n    feature body = extrude(profile = sketch.s, distance = 10 mm)\n}\n";

const BLOCK: &str = "runmat 0.7\nstd 1\n\npart block {\n    sketch s on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])\n        l2 = line(start = l1.end, end = [20 mm, 10 mm])\n        l3 = line(start = l2.end, end = [0 mm, 10 mm])\n        l4 = line(start = l3.end, end = l1.start)\n    }\n    feature box = extrude(profile = sketch.s, distance = 5 mm)\n}\n";

/// The plate lifted to z = 100 mm, fixed.
const RIG: &str = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part plate as plate {\n        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 100 mm] } }\n    }\n}\n";

fn files(rig: &str) -> Files {
    [
        ("/p/plate.part", PLATE),
        ("/p/mech/block.part", BLOCK),
        ("/p/rig.assembly", rig),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v.to_owned()))
    .collect()
}

fn seat(input: Value, files: &Files) -> Value {
    call("design.seat_part", input, files).unwrap_or_else(|e| panic!("design.seat_part: {e}"))
}

fn close(a: &Value, b: [f64; 3]) -> bool {
    (0..3).all(|k| (a[k].as_f64().unwrap() - b[k]).abs() < 1e-3)
}

/// The text a staged change leaves `path` with.
fn staged_text(out: &Value, path: &str) -> String {
    out["files"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["path"] == path)
        .unwrap()["text"]
        .as_str()
        .unwrap()
        .to_owned()
}

#[test]
fn a_block_seats_under_a_plate_by_a_ray_in_one_call() {
    let files = files(RIG);
    let out = seat(
        json!({ "path": "/p/rig.assembly", "component": "foot", "part": "/p/mech/block.part",
            "target": { "component": "plate", "from": [40.0, 25.0, 0.0], "direction": [0.0, 0.0, 1.0] },
            "own": { "side": "+z" }, "offset": 0.5, "clearance": ["plate"] }),
        &files,
    );
    let seat = &out["seat"];
    // The block's top centre (10, 5, 5) goes under the ray's hit (40, 25, 100), 0.5 mm short.
    assert!(
        close(&seat["pose"]["translation_mm"], [30.0, 20.0, 94.5]),
        "{seat:#}"
    );
    assert!(
        seat["pose"]["rotation_deg"].as_f64().unwrap().abs() < 1e-3,
        "{seat:#}"
    );
    assert!(
        close(&seat["contact"]["point_mm"], [40.0, 25.0, 99.5]),
        "{seat:#}"
    );
    assert_eq!(seat["mate_kind"], "fasten", "{seat:#}");
    assert_eq!(seat["mate_status"], "satisfied", "{seat:#}");
    assert_eq!(seat["pose"]["status"], "solved", "{seat:#}");
    // Both kernels preserve the pose. OCCT measures the half-millimetre gap;
    // the grid kernel reports that proximity is unmeasured.
    assert_eq!(seat["interference"], json!([]), "{seat:#}");
    if cfg!(feature = "occt") {
        assert_eq!(
            seat["clearance"],
            json!([{ "with": "plate", "distance_mm": 0.5 }]),
            "{seat:#}"
        );
        assert!(seat.get("notes").is_none(), "{seat:#}");
        assert_eq!(seat["summary"], "Seated foot on plate (fasten, satisfied) at [30, 20, 94.5] mm: no interference; 0.5 mm from plate.", "{seat:#}");
    } else {
        assert_eq!(
            seat["clearance"],
            json!([{ "with": "plate", "not_measured": true }]),
            "{seat:#}"
        );
        assert!(
            seat["notes"]
                .as_array()
                .unwrap()
                .iter()
                .any(|n| n.as_str().unwrap().starts_with("plate / foot: ")),
            "{seat:#}"
        );
        assert!(seat["summary"].as_str().unwrap().starts_with("Seated foot on plate (fasten, satisfied) at [30, 20, 94.5] mm: interference with plate not measured"), "{seat:#}");
    }
    // The staged change inserts the occurrence free to move, its two connectors and the fasten.
    assert_eq!(out["ok"], true, "{:#}", out["diagnostics"]);
    let text = staged_text(&out, "/p/rig.assembly");
    assert!(
        text.contains("use part block from \"./mech\" as foot {\n        fixed = false\n    }"),
        "{text}"
    );
    assert!(text.contains("connector foot_seat_on = { occurrence = [{ name = plate }], at = [40 mm, 25 mm, -0.5 mm], z = [0, 0, -1], x = [1, 0, 0] }"), "{text}");
    assert!(text.contains("connector foot_seat = { occurrence = [{ name = foot }], at = [10 mm, 5 mm, 5 mm], z = [0, 0, 1], x = [1, 0, 0], flip = true }"), "{text}");
    assert!(
        text.contains(
            "mate foot_seated = { kind = { mate = fasten }, a = foot_seat_on, b = foot_seat }"
        ),
        "{text}"
    );
}

#[test]
fn seating_again_on_a_named_face_replaces_the_seat_and_turns_it_along() {
    let first = seat(
        json!({ "path": "/p/rig.assembly", "component": "foot", "part": "/p/mech/block.part",
            "target": { "component": "plate", "from": [40.0, 25.0, 0.0], "direction": [0.0, 0.0, 1.0] }, "own": { "side": "+z" } }),
        &files(RIG),
    );
    let kept = files(&staged_text(&first, "/p/rig.assembly"));
    // The plate's top face, by the name design.inspect gives it.
    let faces = call(
        "design.inspect",
        json!({ "path": "/p/plate.part", "sections": ["faces"] }),
        &kept,
    )
    .unwrap();
    let top = faces["sections"]["faces"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["normal"][2].as_f64().unwrap() > 0.99)
        .unwrap()["reference"]
        .as_str()
        .unwrap()
        .to_owned();
    let out = seat(
        json!({ "path": "/p/rig.assembly", "component": "foot", "target": { "component": "plate", "face": top },
            "own": { "side": "-z" }, "along": [0.0, 1.0, 0.0] }),
        &kept,
    );
    let seat = &out["seat"];
    // Turned a quarter about Z, its bottom centre (10, 5, 0) lands on the top's centre (40, 25, 110).
    assert!(
        (seat["pose"]["rotation_deg"].as_f64().unwrap() - 90.0).abs() < 1e-3,
        "{seat:#}"
    );
    assert!(
        close(&seat["pose"]["translation_mm"], [45.0, 15.0, 110.0]),
        "{seat:#}"
    );
    assert_eq!(seat["mate_status"], "satisfied", "{seat:#}");
    assert_eq!(seat["interference"], json!([]), "{seat:#}");
    let text = staged_text(&out, "/p/rig.assembly");
    assert_eq!(text.matches("mate foot_seated =").count(), 1, "{text}");
    assert_eq!(
        text.matches("connector foot_seat_on =").count(),
        1,
        "{text}"
    );
    assert_eq!(text.matches("as foot {").count(), 1, "{text}");
}

#[test]
fn every_refusal_names_the_pick() {
    let files = files(RIG);
    let refuse = |input: Value| call("design.seat_part", input, &files).unwrap_err().message;
    let m = refuse(
        json!({ "path": "/p/rig.assembly", "component": "foot", "part": "/p/mech/block.part", "target": { "component": "plate", "face": "Nowhere 9" } }),
    );
    assert!(
        m.contains("target face `Nowhere 9` isn't a face of `plate`; its faces include"),
        "{m}"
    );
    let m = refuse(
        json!({ "path": "/p/rig.assembly", "component": "foot", "part": "/p/mech/block.part",
        "target": { "component": "plate", "from": [200.0, 25.0, 0.0], "direction": [0.0, 0.0, 1.0] } }),
    );
    assert!(
        m.contains("the ray from target.from [200.0, 25.0, 0.0]")
            && m.contains("doesn't meet `plate`"),
        "{m}"
    );
    let m = refuse(
        json!({ "path": "/p/rig.assembly", "component": "foot", "target": { "component": "plate", "from": [40.0, 25.0, 0.0], "direction": [0.0, 0.0, 1.0] } }),
    );
    assert!(
        m.contains("component `foot` isn't a top-level occurrence"),
        "{m}"
    );
    let m = refuse(
        json!({ "path": "/p/rig.assembly", "component": "foot", "part": "/p/mech/block.part", "target": { "component": "deck", "from": [40.0, 25.0, 0.0], "direction": [0.0, 0.0, 1.0] } }),
    );
    assert!(
        m.contains("target component `deck` isn't a part occurrence"),
        "{m}"
    );
    let m = refuse(
        json!({ "path": "/p/rig.assembly", "component": "foot", "part": "/p/mech/block.part",
        "target": { "component": "plate", "from": [40.0, 25.0, 0.0], "direction": [0.0, 0.0, 1.0] }, "own": { "side": "+w" } }),
    );
    assert!(m.contains("own side `+w`"), "{m}");
    let m = refuse(
        json!({ "path": "/p/rig.assembly", "component": "foot", "part": "/p/mech/block.part", "target": { "component": "plate" } }),
    );
    assert!(
        m.contains("pick where on `plate` the component goes"),
        "{m}"
    );
}

/// Parts go in through design.seat_part: a proposed edit that adds an occurrence, a connector
/// or a seated kind of mate by hand is refused with an example seating call, and an edit to an
/// existing mate's values goes through.
#[test]
fn a_hand_placement_edit_is_refused_and_points_to_seat_part() {
    let seated = seat(
        json!({ "path": "/p/rig.assembly", "component": "foot", "part": "/p/mech/block.part",
            "target": { "component": "plate", "from": [40.0, 25.0, 0.0], "direction": [0.0, 0.0, 1.0] }, "own": { "side": "+z" } }),
        &files(RIG),
    );
    let text = staged_text(&seated, "/p/rig.assembly");
    let kept = files(&text);
    let add = "    use part block from \"./mech\" as foot2 {\n        fixed = false\n    }\n}\n";
    let by_hand = text.replacen("\n}\n", &format!("\n{add}"), 1);
    let e = call(
        "design.propose_edit",
        json!({ "path": "/p/rig.assembly", "text": by_hand }),
        &kept,
    )
    .unwrap_err();
    assert_eq!(e.code, "use_seat_part", "{}", e.message);
    assert!(
        e.message.contains("adds part occurrence `foot2`")
            && e.message.contains("place parts with design_seat_part")
            && e.message.contains("\"component\": \"foot2\""),
        "{}",
        e.message
    );
    let mate = "mate foot_seated = { kind = { mate = fasten }, a = foot_seat_on, b = foot_seat }";
    let e = call(
        "design.propose_edit",
        json!({ "path": "/p/rig.assembly", "replace": [{ "old": mate, "new": format!("{mate}\n    connector extra = {{ occurrence = [{{ name = foot }}], at = [0 mm, 0 mm, 0 mm], z = [0, 0, 1], x = [1, 0, 0] }}\n    mate foot_again = {{ kind = {{ mate = fasten }}, a = foot_seat_on, b = extra }}") }] }),
        &kept,
    )
    .unwrap_err();
    assert!(
        e.message
            .contains("adds connector `extra`, mate `foot_again`"),
        "{}",
        e.message
    );
    // Changing the existing connector's values stays allowed.
    let out = call(
        "design.propose_edit",
        json!({ "path": "/p/rig.assembly", "replace": [{ "old": "at = [40 mm, 25 mm, 0 mm]", "new": "at = [40 mm, 25 mm, -1 mm]" }] }),
        &kept,
    )
    .unwrap();
    assert_eq!(out["ok"], true, "{out:#}");
}

/// A clamp: 12 mm long along X, 14 wide, 20 tall, with an 8.2 mm bore along X 5 mm above its
/// base.
const CLAMP: &str = "runmat 0.7\nstd 1\n\npart skid_clamp {\n    sketch s on origin.yz {\n        l1 = line(start = [-7 mm, 0 mm], end = [7 mm, 0 mm])\n        l2 = line(start = l1.end, end = [7 mm, 20 mm])\n        l3 = line(start = l2.end, end = [-7 mm, 20 mm])\n        l4 = line(start = l3.end, end = l1.start)\n        bore = circle(center = [0 mm, 5 mm], radius = 4.1 mm)\n    }\n    feature body = extrude(profile = sketch.s, distance = 12 mm)\n}\n";

/// Two plates whose undersides differ by 4 mm: `high` at z = 100 mm over x 0..80, `low` at
/// z = 96 mm over x 100..180.
const STEP: &str = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part plate as high {\n        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 100 mm] } }\n    }\n    use part plate as low {\n        placement = { placement = offset, offset = { translation = [100 mm, 0 mm, 96 mm] } }\n    }\n}\n";

/// Clamps for one straight tube under undersides at two heights share one bore axis with
/// `levelWith`: the later clamp under the higher underside stops short of it, level with the
/// first. Under a lower underside the axis would press it in, so it stays at first contact and
/// the notes say so.
#[test]
fn a_later_clamp_levels_with_the_first_on_one_bore_axis() {
    let mut files = files(STEP);
    files.insert("/p/mech/skid_clamp.part".into(), CLAMP.into());
    let ray = |x: f64| json!({ "component": if x > 90.0 { "low" } else { "high" }, "from": [x, 25.0, 0.0], "direction": [0.0, 0.0, 1.0] });
    let first = seat(
        json!({ "path": "/p/rig.assembly", "component": "front", "part": "/p/mech/skid_clamp.part", "target": ray(140.0),
            "own": { "side": "+z" }, "offset": 0.5, "along": [1.0, 0.0, 0.0] }),
        &files,
    );
    // Its top 0.5 mm under the low underside (96 mm): the base at 75.5 mm.
    assert!(
        (first["seat"]["pose"]["translation_mm"][2].as_f64().unwrap() - 75.5).abs() < 1e-3,
        "{:#}",
        first["seat"]
    );
    files.insert(
        "/p/rig.assembly".into(),
        staged_text(&first, "/p/rig.assembly"),
    );
    let rear = seat(
        json!({ "path": "/p/rig.assembly", "component": "rear", "part": "/p/mech/skid_clamp.part", "target": ray(40.0),
            "own": { "side": "+z" }, "offset": 0.5, "levelWith": "front" }),
        &files,
    );
    let s = &rear["seat"];
    // Level with the front clamp, 4.5 mm short of the high underside, along X like it.
    assert!(
        close(&s["pose"]["translation_mm"], [34.0, 25.0, 75.5]),
        "{s:#}"
    );
    assert!(
        s["pose"]["rotation_deg"].as_f64().unwrap().abs() < 1e-3,
        "{s:#}"
    );
    assert_eq!(s["mate_status"], "satisfied", "{s:#}");
    assert!(s["notes"].as_array().unwrap().iter().any(|n| n.as_str().unwrap().starts_with("on `front`'s axis: moved 4.00 mm from first contact, about 4.50 mm short of `high`")), "{s:#}");
    files.insert(
        "/p/rig.assembly".into(),
        staged_text(&rear, "/p/rig.assembly"),
    );
    // Under the low underside, level with the rear clamp seated again up at the high one would
    // press it 3.5 mm in.
    let high = seat(
        json!({ "path": "/p/rig.assembly", "component": "rear", "target": ray(40.0), "own": { "side": "+z" }, "offset": 0.5, "along": [1.0, 0.0, 0.0] }),
        &files,
    );
    files.insert(
        "/p/rig.assembly".into(),
        staged_text(&high, "/p/rig.assembly"),
    );
    let pressed = seat(
        json!({ "path": "/p/rig.assembly", "component": "front", "target": ray(140.0), "own": { "side": "+z" }, "offset": 0.5, "levelWith": "rear" }),
        &files,
    );
    let s = &pressed["seat"];
    assert!(
        (s["pose"]["translation_mm"][2].as_f64().unwrap() - 75.5).abs() < 1e-3,
        "{s:#}"
    );
    assert!(
        s["notes"].as_array().unwrap().iter().any(|n| n
            .as_str()
            .unwrap()
            .starts_with("on `rear`'s axis `front` would press 3.50 mm into `low`")),
        "{s:#}"
    );
    // A component with no round face can't be levelled, and nothing levels with itself.
    let m = call("design.seat_part", json!({ "path": "/p/rig.assembly", "component": "foot", "part": "/p/mech/block.part", "target": ray(40.0), "levelWith": "front" }), &files).unwrap_err().message;
    assert!(m.contains("component `foot` has no round face"), "{m}");
    let m = call("design.seat_part", json!({ "path": "/p/rig.assembly", "component": "front", "target": ray(140.0), "levelWith": "front" }), &files).unwrap_err().message;
    assert!(m.contains("can't be levelled with itself"), "{m}");
}

/// An 8 mm tube 120 mm long along X.
const TUBE: &str = "runmat 0.7\nstd 1\n\npart tube {\n    sketch s on origin.yz {\n        outer = circle(center = [0 mm, 0 mm], radius = 4 mm)\n    }\n    feature body = extrude(profile = sketch.s, distance = 120 mm)\n}\n";

/// The STEP rig with the clamp and the tube parts.
fn step_files() -> Files {
    let mut files = files(STEP);
    files.insert("/p/mech/skid_clamp.part".into(), CLAMP.into());
    files.insert("/p/mech/tube.part".into(), TUBE.into());
    files
}

fn under(x: f64) -> Value {
    json!({ "component": if x > 90.0 { "low" } else { "high" }, "from": [x, 25.0, 0.0], "direction": [0.0, 0.0, 1.0] })
}

/// A tube seated `through` two clamps' bores lies on their common axis with its middle midway
/// between them, fastened to the first; clamps whose bores differ in height are refused with
/// the levelWith seating that lines them up.
#[test]
fn a_tube_through_two_clamps_lies_on_their_axis_centred_between_them() {
    let mut files = step_files();
    for (clamp, x) in [("front", 140.0), ("rear", 40.0)] {
        let out = seat(
            json!({ "path": "/p/rig.assembly", "component": clamp, "part": "/p/mech/skid_clamp.part", "target": under(x),
            "own": { "side": "+z" }, "offset": 0.5, "along": [1.0, 0.0, 0.0] }),
            &files,
        );
        files.insert(
            "/p/rig.assembly".into(),
            staged_text(&out, "/p/rig.assembly"),
        );
    }
    // The rear clamp sits under the higher underside: its bore is 4 mm above the front one's.
    let tube = json!({ "path": "/p/rig.assembly", "component": "skid", "part": "/p/mech/tube.part", "through": ["front", "rear"] });
    let m = call("design.seat_part", tube.clone(), &files)
        .unwrap_err()
        .message;
    assert!(
        m.contains("the bores aren't on one axis")
            && m.contains("4.00 mm from `front.")
            && m.contains("seat `rear` again with levelWith `front`"),
        "{m}"
    );
    let out = seat(
        json!({ "path": "/p/rig.assembly", "component": "rear", "target": under(40.0), "own": { "side": "+z" }, "offset": 0.5, "levelWith": "front" }),
        &files,
    );
    files.insert(
        "/p/rig.assembly".into(),
        staged_text(&out, "/p/rig.assembly"),
    );
    let out = seat(tube, &files);
    let s = &out["seat"];
    // Bores at x 140 and 40 (each clamp's middle), 5 mm above the 75.5 mm bases: the tube's
    // middle (60 mm along it) at x 90.
    assert!(
        close(&s["pose"]["translation_mm"], [30.0, 25.0, 80.5]),
        "{s:#}"
    );
    assert!(
        s["pose"]["rotation_deg"].as_f64().unwrap().abs() < 1e-3,
        "{s:#}"
    );
    assert_eq!(s["target"], "front", "{s:#}");
    assert_eq!(s["mate_status"], "satisfied", "{s:#}");
    let rows = s["through"].as_array().unwrap();
    assert_eq!(rows.len(), 2, "{s:#}");
    for (row, station) in rows.iter().zip([50.0, -50.0]) {
        assert!(
            row["off_axis_mm"].as_f64().unwrap() < 1e-3
                && (row["station_mm"].as_f64().unwrap() - station).abs() < 1e-3,
            "{s:#}"
        );
    }
    let text = staged_text(&out, "/p/rig.assembly");
    assert_eq!(text.matches("mate skid_seated =").count(), 1, "{text}");
    // Through takes no target, and at least two holders.
    let m = call("design.seat_part", json!({ "path": "/p/rig.assembly", "component": "skid", "through": ["front", "rear"], "target": under(40.0) }), &files).unwrap_err().message;
    assert!(m.contains("leave out target"), "{m}");
    let m = call(
        "design.seat_part",
        json!({ "path": "/p/rig.assembly", "component": "skid", "through": ["front"] }),
        &files,
    )
    .unwrap_err()
    .message;
    assert!(m.contains("two or more holders"), "{m}");
}

/// `mirror` seats a clamp as the mirror image of one already seated, across the plate's middle
/// plane square to Y: a placement turned half round Z, the same height and station.
#[test]
fn a_mirrored_clamp_is_the_first_one_reflected_across_the_target() {
    let mut files = files(RIG);
    files.insert("/p/mech/skid_clamp.part".into(), CLAMP.into());
    let out = seat(
        json!({ "path": "/p/rig.assembly", "component": "left", "part": "/p/mech/skid_clamp.part",
        "target": { "component": "plate", "from": [40.0, 40.0, 0.0], "direction": [0.0, 0.0, 1.0] },
        "own": { "side": "+z" }, "offset": 0.5, "along": [1.0, 0.0, 0.0] }),
        &files,
    );
    // Its top centre (6, 0, 20) 0.5 mm under the ray's hit (40, 40, 100).
    assert!(
        close(&out["seat"]["pose"]["translation_mm"], [34.0, 40.0, 79.5]),
        "{:#}",
        out["seat"]
    );
    files.insert(
        "/p/rig.assembly".into(),
        staged_text(&out, "/p/rig.assembly"),
    );
    let out = seat(
        json!({ "path": "/p/rig.assembly", "component": "right", "part": "/p/mech/skid_clamp.part",
        "target": { "component": "plate" }, "mirror": "left" }),
        &files,
    );
    let s = &out["seat"];
    // The plate's middle plane is y = 25: the reflection spans x 34..46, y 3..17, z 79.5..99.5,
    // which is the clamp turned 180° about Z with its origin at (46, 10, 79.5).
    assert!(
        close(&s["pose"]["translation_mm"], [46.0, 10.0, 79.5]),
        "{s:#}"
    );
    assert!(
        (s["pose"]["rotation_deg"].as_f64().unwrap() - 180.0).abs() < 1e-3,
        "{s:#}"
    );
    assert!(
        s["pose"]["rotation_axis"][2].as_f64().unwrap().abs() > 0.999,
        "{s:#}"
    );
    assert_eq!(s["mate_status"], "satisfied", "{s:#}");
    // Only a copy of the same part mirrors, and mirror takes the target component alone.
    let m = call("design.seat_part", json!({ "path": "/p/rig.assembly", "component": "odd", "part": "/p/mech/block.part", "target": { "component": "plate" }, "mirror": "left" }), &files).unwrap_err().message;
    assert!(m.contains("isn't the same part as `left`"), "{m}");
    let m = call("design.seat_part", json!({ "path": "/p/rig.assembly", "component": "right", "part": "/p/mech/skid_clamp.part", "target": { "component": "plate", "face": "x" }, "mirror": "left" }), &files).unwrap_err().message;
    assert!(m.contains("give target.component only"), "{m}");
}

/// A leg, sketched square to X and extruded 8 mm along it: a 16 mm pad on top (y -8..8), a web
/// down and out to +y, and a foot 50 mm down holding a bore along X 25 mm out. Its box's top
/// is centred 12.5 mm out from the pad.
const LEG: &str = "runmat 0.7\nstd 1\n\npart leg {\n    sketch s on origin.yz {\n        l1 = line(start = [-8 mm, 0 mm], end = [8 mm, 0 mm])\n        l2 = line(start = l1.end, end = [33 mm, -42 mm])\n        l3 = line(start = l2.end, end = [33 mm, -58 mm])\n        l4 = line(start = l3.end, end = [17 mm, -58 mm])\n        l5 = line(start = l4.end, end = [17 mm, -42 mm])\n        l6 = line(start = l5.end, end = [-8 mm, -4 mm])\n        l7 = line(start = l6.end, end = l1.start)\n        bore = circle(center = [25 mm, -50 mm], radius = 4.2 mm)\n    }\n    feature body = extrude(profile = sketch.s, distance = 8 mm)\n}\n";

fn leg_under_plate(rotation: f64, along: [f64; 3]) -> Value {
    let mut files = files(RIG);
    files.insert("/p/mech/leg.part".into(), LEG.into());
    let out = seat(
        json!({ "path": "/p/rig.assembly", "component": "leg", "part": "/p/mech/leg.part",
            "target": { "component": "plate", "from": [40.0, 25.0, 0.0], "direction": [0.0, 0.0, 1.0] },
            "own": { "side": "+z" }, "offset": 0.5, "along": along, "rotation": rotation }),
        &files,
    );
    out["seat"].clone()
}

/// `own.side` meets the target on the part's material at that side: the leg's pad, not the
/// middle of its box's top, which lies 12.5 mm out over the web.
#[test]
fn a_side_contact_is_on_the_material_at_that_side() {
    let s = leg_under_plate(0.0, [1.0, 0.0, 0.0]);
    // The pad's middle (4, 0, 0) 0.5 mm under the ray's hit (40, 25, 100).
    assert!(
        close(&s["contact"]["point_mm"], [40.0, 25.0, 99.5]),
        "{s:#}"
    );
    assert!(
        close(&s["pose"]["translation_mm"], [36.0, 25.0, 99.5]),
        "{s:#}"
    );
    assert_eq!(s["mate_status"], "satisfied", "{s:#}");
}

/// `rotation` turns from `along`: at 0 the leg's long axis (its bore's) points its own +X along
/// `along` whatever sign the kernel gives the bore's axis, at 180 the other way, so the web
/// swings from +y to -y about the pad. The same picks give the same pose, and along -X at 0 is
/// along +X at 180.
#[test]
fn rotation_turns_from_along_whatever_the_bore_axis_sign() {
    let straight = leg_under_plate(0.0, [1.0, 0.0, 0.0]);
    assert!(
        straight["pose"]["rotation_deg"].as_f64().unwrap().abs() < 1e-3,
        "{straight:#}"
    );
    let turned = leg_under_plate(180.0, [1.0, 0.0, 0.0]);
    assert!(
        (turned["pose"]["rotation_deg"].as_f64().unwrap() - 180.0).abs() < 1e-3,
        "{turned:#}"
    );
    assert!(
        turned["pose"]["rotation_axis"][2].as_f64().unwrap().abs() > 0.999,
        "{turned:#}"
    );
    // Turned about the pad's middle: (4, 0, 0) still lands under the hit.
    assert!(
        close(&turned["pose"]["translation_mm"], [44.0, 25.0, 99.5]),
        "{turned:#}"
    );
    assert!(
        close(&turned["contact"]["point_mm"], [40.0, 25.0, 99.5]),
        "{turned:#}"
    );
    assert_eq!(
        leg_under_plate(180.0, [1.0, 0.0, 0.0])["pose"],
        turned["pose"]
    );
    let back = leg_under_plate(0.0, [-1.0, 0.0, 0.0]);
    assert_eq!(
        back["pose"]["translation_mm"], turned["pose"]["translation_mm"],
        "{back:#}"
    );
    assert!(
        (back["pose"]["rotation_deg"].as_f64().unwrap() - 180.0).abs() < 1e-3,
        "{back:#}"
    );
}
