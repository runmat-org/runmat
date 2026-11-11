//! `design.inspect`'s `stance` section: how an assembly stands on its contact components. A
//! 200 × 100 × 20 mm aluminium body sits 100 mm up with a 20 mm camera cube hanging under it
//! and two 160 × 10 × 10 mm skid bars 80 mm apart. The bars clamped high leave the camera
//! lowest; lowered on legs they carry it clear with the centre of mass inside the support; moved
//! aft they leave the centre of mass outside it.

use runmat_design_tools::{call, Files};
use serde_json::{json, Value};

fn block(name: &str, x: u32, y: u32, z: u32) -> String {
    format!(
        "runmat 0.7\nstd 1\n\npart {name} {{\n    sketch s on origin.xy {{\n        l1 = line(start = [0 mm, 0 mm], end = [{x} mm, 0 mm])\n        l2 = line(start = l1.end, end = [{x} mm, {y} mm])\n        l3 = line(start = l2.end, end = [0 mm, {y} mm])\n        l4 = line(start = l3.end, end = l1.start)\n    }}\n    feature body = extrude(profile = sketch.s, distance = {z} mm)\n    output {name} = {{ body = feature.body.body, material = \"material.aluminum_6061\" }}\n}}\n"
    )
}

fn placed(part: &str, name: &str, at: [f64; 3]) -> String {
    format!(
        "    use part {part} as {name} {{\n        placement = {{ placement = offset, offset = {{ translation = [{} mm, {} mm, {} mm] }} }}\n    }}\n",
        at[0], at[1], at[2]
    )
}

/// The rig with the skid bars' lower faces at `z` mm, shifted `dx` mm along X.
fn rig(z: f64, dx: f64) -> Files {
    let assembly = format!(
        "runmat 0.7\nstd 1\n\nassembly craft {{\n{}{}{}{}}}\n",
        placed("frame", "body", [-100.0, -50.0, 100.0]),
        placed("cam", "camera", [-10.0, -10.0, 60.0]),
        placed("bar", "skid_left", [-80.0 + dx, 35.0, z]),
        placed("bar", "skid_right", [-80.0 + dx, -45.0, z]),
    );
    [
        ("/p/frame.part".to_owned(), block("frame", 200, 100, 20)),
        ("/p/cam.part".to_owned(), block("cam", 20, 20, 20)),
        ("/p/bar.part".to_owned(), block("bar", 160, 10, 10)),
        ("/p/craft.assembly".to_owned(), assembly),
    ]
    .into_iter()
    .collect()
}

fn stance(files: &Files) -> Value {
    let out = call(
        "design.inspect",
        json!({ "path": "/p/craft.assembly", "sections": ["stance"], "contact": ["skid_*"] }),
        files,
    )
    .unwrap_or_else(|e| panic!("design.inspect: {e}"));
    out["sections"]["stance"].clone()
}

fn near(v: &Value, want: f64) -> bool {
    v.as_f64().is_some_and(|x| (x - want).abs() < 0.05)
}

#[test]
fn bars_clamped_under_the_body_leave_the_camera_lowest() {
    let s = stance(&rig(70.0, 0.0));
    assert_eq!(s["contact"], json!(["skid_left", "skid_right"]), "{s:#}");
    assert!(near(&s["ground_mm"], 70.0), "{s:#}");
    // The camera hangs 10 mm below the bars: it meets the ground first.
    assert_eq!(
        s["ground_clearance"]["lowest"]["component"], "camera",
        "{s:#}"
    );
    assert!(
        near(&s["ground_clearance"]["lowest"]["clearance_mm"], -10.0),
        "{s:#}"
    );
    assert!(
        near(
            &s["ground_clearance"]["components"][1]["clearance_mm"],
            30.0
        ),
        "{s:#}"
    );
    assert_eq!(s["ok"], false, "{s:#}");
    let message = s["findings"][0]["message"].as_str().unwrap();
    assert!(
        message.starts_with("camera reaches 10 mm below the contact set's lowest point"),
        "{message}"
    );
}

#[test]
fn bars_on_legs_carry_the_camera_clear_with_the_centre_of_mass_inside() {
    let s = stance(&rig(30.0, 0.0));
    assert_eq!(s["ok"], true, "{s:#}");
    assert_eq!(s["findings"], json!([]), "{s:#}");
    assert!(
        near(&s["ground_clearance"]["lowest"]["clearance_mm"], 30.0),
        "{s:#}"
    );
    // Aluminium throughout: the centre of mass is the volume-weighted centroid,
    // (400000·110 + 8000·70 + 32000·35) / 440000 = 103.8 mm up, 73.8 mm above the ground.
    let com = &s["center_of_mass"];
    assert!(
        near(&com["point_mm"][0], 0.0) && near(&com["point_mm"][1], 0.0),
        "{com:#}"
    );
    assert!(near(&com["height_mm"], 73.8), "{com:#}");
    assert_eq!(com["complete"], true, "{com:#}");
    // The bars' lower faces span 160 × 90 mm: 45 mm to the side edges, 80 mm to the ends.
    let support = &s["support"];
    assert_eq!(support["com_inside"], true, "{support:#}");
    assert!(near(&support["min_margin_mm"], 45.0), "{support:#}");
    assert!(near(&support["min_tip_angle_deg"], 31.4), "{support:#}");
    let sides: Vec<&str> = support["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["side"].as_str().unwrap())
        .collect();
    assert_eq!(sides.len(), 4, "{support:#}");
    assert!(
        ["+y", "-y"].contains(&sides[0]) && ["+x", "-x"].contains(&sides[3]),
        "{support:#}"
    );
    assert!(near(&support["edges"][3]["margin_mm"], 80.0), "{support:#}");
    let stance = &s["stance"];
    assert!(
        near(&stance["x_mm"], 160.0) && near(&stance["y_mm"], 90.0),
        "{stance:#}"
    );
    assert!(
        near(&stance["overall_mm"]["x"], 200.0) && near(&stance["overall_mm"]["y"], 100.0),
        "{stance:#}"
    );
    assert!(near(&stance["ratio"]["y"], 0.9), "{stance:#}");
    assert!(s["summary"].as_str().unwrap().starts_with("Standing on skid_left, skid_right: lowest other component camera, 30 mm above the ground; centre of mass inside the support, 45 mm from its"), "{s:#}");
}

#[test]
fn bars_drawn_in_tip_under_thirty_degrees_and_the_stance_is_not_ok() {
    // The bars 10 mm closer together: 40 mm from the centre of mass to the side edges,
    // atan(40 / 73.8) = 28.5°, under the 30° the stance needs on every edge.
    let mut files = rig(30.0, 0.0);
    let assembly = files.get_mut("/p/craft.assembly").unwrap();
    *assembly = assembly
        .replace("35 mm, 30 mm", "30 mm, 30 mm")
        .replace("-45 mm, 30 mm", "-40 mm, 30 mm");
    let s = stance(&files);
    assert!(near(&s["support"]["min_margin_mm"], 40.0), "{s:#}");
    assert!(near(&s["support"]["min_tip_angle_deg"], 28.5), "{s:#}");
    assert_eq!(s["ok"], false, "{s:#}");
    assert_eq!(s["findings"][0]["severity"], "error", "{s:#}");
    let message = s["findings"][0]["message"].as_str().unwrap();
    assert!(
        message.contains("after leaning 28.5° (needs at least 30°)"),
        "{message}"
    );
}

#[test]
fn bars_moved_aft_leave_the_centre_of_mass_outside() {
    // The bars span x 20..180 mm; their own mass pulls the centre to x = 32000·100 / 440000.
    let s = stance(&rig(30.0, 100.0));
    assert_eq!(s["ok"], false, "{s:#}");
    assert_eq!(s["support"]["com_inside"], false, "{s:#}");
    assert!(
        near(&s["support"]["min_margin_mm"], -(20.0 - 3200.0 / 440.0)),
        "{s:#}"
    );
    let message = s["findings"][0]["message"].as_str().unwrap();
    assert!(
        message.contains("outside the support past its -x edge"),
        "{message}"
    );
}

#[test]
fn the_stance_needs_its_contact_set() {
    let files = rig(30.0, 0.0);
    let e = call(
        "design.inspect",
        json!({ "path": "/p/craft.assembly", "sections": ["stance"] }),
        &files,
    )
    .unwrap_err();
    assert!(e.to_string().contains("needs contact"), "{e}");
    let e = call(
        "design.inspect",
        json!({ "path": "/p/craft.assembly", "sections": ["stance"], "contact": ["wheel"] }),
        &files,
    )
    .unwrap_err();
    assert!(
        e.to_string().contains("no occurrence matches `wheel`"),
        "{e}"
    );
}
