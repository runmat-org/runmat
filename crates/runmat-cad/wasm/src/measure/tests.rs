//! Measure's permutation matrix: every pair of vertex, edge, face and body for distance, the
//! one-item sizes, angles, radii and path length, on a part and across an assembly's components.

use super::*;
use crate::analyze::{analyze_json, Analysis};
use runmat_cad::display::PartDisplay;

/// A 100 × 10 × 10 mm block from the origin.
const BLOCK: &str = "runmat 0.7\nstd 1\n\npart block {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [100 mm, 0 mm])\n        l2 = line(start = [100 mm, 0 mm], end = [100 mm, 10 mm])\n        l3 = line(start = [100 mm, 10 mm], end = [0 mm, 10 mm])\n        l4 = line(start = [0 mm, 10 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n}\n";
/// A radius 10 mm, 8 mm tall cylinder on the origin.
const BOSS: &str = "runmat 0.7\nstd 1\n\npart boss {\n    sketch s1 on origin.xy {\n        c = circle(center = [0 mm, 0 mm], radius = 10 mm)\n    }\n    feature body = extrude(profile = sketch.s1, distance = 8 mm)\n}\n";
/// The boss 50 mm along X and 30 mm up from the block.
const RIG: &str = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part block as block\n    use part boss as boss {\n        placement = { placement = offset, offset = { translation = [150 mm, 5 mm, 30 mm] } }\n    }\n}\n";

fn files() -> Files {
    [("/p/block.part", BLOCK), ("/p/boss.part", BOSS)].map(|(p, t)| (p.to_owned(), t.to_owned())).into()
}

fn display(text: &str) -> PartDisplay {
    let path = if text == BOSS { "/p/boss.part" } else { "/p/block.part" };
    match crate::runtime::regenerate_part_in(path, text, None, &TessellationOptions::default()) {
        Ok(part) => part.display,
        Err(d) => panic!("doesn't regenerate: {:?}", d.iter().map(|d| d.message.clone()).collect::<Vec<_>>()),
    }
}

fn centroid(points: &[f32]) -> V3 {
    let n = (points.len() / 3).max(1) as f64;
    let mut c = [0.0; 3];
    for p in points.chunks_exact(3) {
        for i in 0..3 {
            c[i] += p[i] as f64 / n;
        }
    }
    c
}

fn near(a: V3, b: V3) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() < 2e-4)
}

/// The face whose vertex centroid is at `at` (metres): `face:<body>:<index>`.
fn face(d: &PartDisplay, at: V3) -> String {
    for b in &d.bodies {
        for f in &b.faces {
            if near(centroid(&f.positions), at) {
                return format!("face:{}:{}", b.index, f.index);
            }
        }
    }
    panic!("no face at {at:?}");
}

/// The edge whose points' centroid is at `at`.
fn edge(d: &PartDisplay, at: V3) -> String {
    for b in &d.bodies {
        for e in &b.edges {
            if near(centroid(&e.points), at) {
                return format!("edge:{}:{}", b.index, e.index);
            }
        }
    }
    panic!("no edge at {at:?}");
}

fn vertex(d: &PartDisplay, at: V3) -> String {
    for b in &d.bodies {
        for v in &b.vertices {
            if near(v.point.map(f64::from), at) {
                return format!("vertex:{}:{}", b.index, v.index);
            }
        }
    }
    panic!("no vertex at {at:?}");
}

/// A curved face (no plane in its surface): the boss's side.
fn curved_face(d: &PartDisplay) -> String {
    for b in &d.bodies {
        for f in &b.faces {
            if !matches!(f.surface, Some(runmat_cad::refs::history::SurfaceKind::Plane { .. })) {
                return format!("face:{}:{}", b.index, f.index);
            }
        }
    }
    panic!("no curved face");
}

/// A closed edge (a circle): first point equals last.
fn circle_edge(d: &PartDisplay, z: f64) -> String {
    for b in &d.bodies {
        for e in &b.edges {
            if e.points.len() > 9 && (e.points[2] as f64 - z).abs() < 1e-6 {
                return format!("edge:{}:{}", b.index, e.index);
            }
        }
    }
    panic!("no circular edge at z {z}");
}

fn run(path: &str, text: &str, mode: &str, items: &[String]) -> Result<Analysis, String> {
    let mut values = Map::new();
    values.insert("mode".into(), json!(mode));
    values.insert("items".into(), json!(items));
    measure(path, text, &files(), &values)
}

fn part(mode: &str, items: &[String]) -> Analysis {
    run("/p/block.part", BLOCK, mode, items).unwrap_or_else(|e| panic!("{mode} {items:?}: {e}"))
}

/// The row `quantity`'s value in mm (or ° or mm²).
fn value(a: &Analysis, quantity: &str) -> f64 {
    let row = a.rows.iter().find(|r| r.cells[0] == quantity).unwrap_or_else(|| panic!("no {quantity} in {:?}", a.rows));
    row.cells[1].split_whitespace().next().unwrap().trim_end_matches('°').parse().unwrap()
}

fn has(a: &Analysis, quantity: &str) -> bool {
    a.rows.iter().any(|r| r.cells[0] == quantity)
}

const MM: f64 = 1e-3;

struct Block {
    left: String,   // the x = 0 face
    right: String,  // the x = 100 face
    bottom: String, // z = 0
    front: String,  // y = 0
    top_back: String, // the edge at y = 10, z = 10
    bottom_front: String, // the edge at y = 0, z = 0
    left_up: String, // the edge at x = 0, y = 0, along Z
    origin: String,  // the vertex at the origin
    far: String,     // the vertex at (100, 10, 10)
}

fn block() -> Block {
    let d = display(BLOCK);
    Block {
        left: face(&d, [0.0, 5.0 * MM, 5.0 * MM]),
        right: face(&d, [100.0 * MM, 5.0 * MM, 5.0 * MM]),
        bottom: face(&d, [50.0 * MM, 5.0 * MM, 0.0]),
        front: face(&d, [50.0 * MM, 0.0, 5.0 * MM]),
        top_back: edge(&d, [50.0 * MM, 10.0 * MM, 10.0 * MM]),
        bottom_front: edge(&d, [50.0 * MM, 0.0, 0.0]),
        left_up: edge(&d, [0.0, 0.0, 5.0 * MM]),
        origin: vertex(&d, [0.0; 3]),
        far: vertex(&d, [100.0 * MM, 10.0 * MM, 10.0 * MM]),
    }
}

/// Within the display tessellation's chordal tolerance, for distances to curved surfaces.
fn within(got: f64, want: f64, tolerance: f64) {
    assert!((got - want).abs() < tolerance, "got {got}, want {want}");
}

fn close(got: f64, want: f64) {
    assert!((got - want).abs() < 2e-3, "got {got}, want {want}");
}

#[test]
fn face_to_face_parallel_gives_minimum_and_normal_distance() {
    let b = block();
    let a = part("distance", &[b.left.clone(), b.right.clone()]);
    close(value(&a, "Minimum distance"), 100.0);
    close(value(&a, "Normal distance"), 100.0);
    close(value(&a, "ΔX"), 100.0);
    assert!(!has(&a, "Angle"), "{:?}", a.rows);
    assert_eq!(a.dimensions.len(), 1);
    close(norm(sub(a.dimensions[0].to, a.dimensions[0].from)) * 1e3, 100.0);
    assert_eq!(a.dimensions[0].label, "100.000 mm");
}

#[test]
fn face_to_face_adjacent_touch_at_ninety_degrees() {
    let b = block();
    let a = part("distance", &[b.left.clone(), b.bottom.clone()]);
    close(value(&a, "Minimum distance"), 0.0);
    close(value(&a, "Angle"), 90.0);
    assert!(!has(&a, "Normal distance"));
}

#[test]
fn every_pair_of_vertex_edge_and_face_measures_its_minimum_distance() {
    let b = block();
    // (first, second, distance in mm)
    let diag = (100.0f64 * 100.0 + 10.0 * 10.0 + 10.0 * 10.0).sqrt();
    let cases: [(&str, &str, f64); 9] = [
        (&b.origin, &b.far, diag),                   // vertex–vertex
        (&b.origin, &b.top_back, (200.0f64).sqrt()), // vertex–edge: (0,0,0) to the line y=10, z=10
        (&b.far, &b.left, 100.0),                    // vertex–face
        (&b.top_back, &b.bottom_front, (200.0f64).sqrt()), // edge–edge, parallel
        (&b.left_up, &b.top_back, 10.0),             // edge–edge, skew: (0,0,z) to (x,10,10)
        (&b.top_back, &b.bottom, 10.0),              // edge–face
        (&b.left_up, &b.right, 100.0),               // edge–face, parallel
        (&b.front, &b.top_back, 10.0),               // face–edge
        (&b.right, &b.origin, 100.0),                // face–vertex
    ];
    for (x, y, want) in cases {
        let a = part("distance", &[x.to_owned(), y.to_owned()]);
        close(value(&a, "Minimum distance"), want);
        let d = &a.dimensions[0];
        close(norm(sub(d.to, d.from)) * 1e3, a.rows.iter().find(|r| r.cells[0] == "Normal distance").map_or(want, |_| value(&a, "Normal distance")));
    }
}

#[test]
fn one_item_gives_its_own_size() {
    let b = block();
    close(value(&part("distance", &[b.bottom_front.clone()]), "Length"), 100.0);
    close(value(&part("distance", &[b.left_up.clone()]), "Length"), 10.0);
    close(value(&part("distance", &[b.bottom.clone()]), "Area"), 1000.0);
    close(value(&part("distance", &[b.left.clone()]), "Area"), 100.0);
    let v = part("distance", &[b.far.clone()]);
    close(value(&v, "X"), 100.0);
    close(value(&v, "Z"), 10.0);
    let body = part("distance", &["body:0".to_owned()]);
    close(value(&body, "Extent X"), 100.0);
    close(value(&body, "Extent Y"), 10.0);
}

#[test]
fn angles_between_faces_edges_and_both() {
    let b = block();
    close(value(&part("angle", &[b.left.clone(), b.bottom.clone()]), "Angle"), 90.0);
    close(value(&part("angle", &[b.left.clone(), b.right.clone()]), "Angle"), 0.0);
    close(value(&part("angle", &[b.left_up.clone(), b.bottom_front.clone()]), "Angle"), 90.0);
    close(value(&part("angle", &[b.top_back.clone(), b.bottom_front.clone()]), "Angle"), 0.0);
    // A line and a plane: the edge along Z stands square to the bottom, lies in the left face.
    close(value(&part("angle", &[b.left_up.clone(), b.bottom.clone()]), "Angle"), 90.0);
    close(value(&part("angle", &[b.left_up.clone(), b.left.clone()]), "Angle"), 0.0);
    close(value(&part("angle", &[b.left.clone(), b.bottom.clone()]), "Supplement"), 90.0);
    let err = run("/p/block.part", BLOCK, "angle", &[b.origin.clone(), b.left.clone()]).unwrap_err();
    assert!(err.contains("no direction"), "{err}");
}

#[test]
fn path_length_adds_edges() {
    let b = block();
    close(value(&part("path", &[b.bottom_front.clone(), b.left_up.clone(), b.top_back.clone()]), "Path length"), 210.0);
    assert!(run("/p/block.part", BLOCK, "path", &[b.left.clone()]).unwrap_err().contains("isn't an edge"));
}

#[test]
fn radius_and_diameter_of_a_circular_edge_and_a_cylindrical_face() {
    let d = display(BOSS);
    let side = curved_face(&d);
    let rim = circle_edge(&d, 8.0 * MM);
    for pick in [&side, &rim] {
        let a = run("/p/boss.part", BOSS, "radius", &[pick.clone()]).unwrap();
        close(value(&a, "Radius"), 10.0);
        close(value(&a, "Diameter"), 20.0);
        assert_eq!(a.dimensions[0].label, "R 10.000 mm");
        close(norm(sub(a.dimensions[0].to, a.dimensions[0].from)) * 1e3, 10.0);
    }
    // One circular edge: its arc length is the circle's, not its chords'.
    let one = run("/p/boss.part", BOSS, "distance", &[rim.clone()]).unwrap();
    close(value(&one, "Length"), 20.0 * std::f64::consts::PI);
    close(value(&one, "Radius"), 10.0);
    // Top to bottom: parallel planes 8 mm apart.
    let top = face(&d, [0.0, 0.0, 8.0 * MM]);
    let bottom = face(&d, [0.0, 0.0, 0.0]);
    close(value(&run("/p/boss.part", BOSS, "distance", &[top.clone(), bottom]).unwrap(), "Normal distance"), 8.0);
    // The two rims: parallel planes 8 mm apart on one axis.
    let both = run("/p/boss.part", BOSS, "distance", &[rim.clone(), circle_edge(&d, 0.0)]).unwrap();
    close(value(&both, "Normal distance"), 8.0);
    close(value(&both, "Axis distance"), 0.0);
    // A cylinder's axis stands square to its top.
    close(value(&run("/p/boss.part", BOSS, "angle", &[side.clone(), top.clone()]).unwrap(), "Angle"), 90.0);
    let err = run("/p/boss.part", BOSS, "radius", &[top]).unwrap_err();
    assert!(err.contains("isn't circular"), "{err}");
}

fn uuid(name: &str) -> String {
    let (index, _, _) = crate::runtime::compile_assembly("/p/rig.assembly", RIG, &files()).unwrap();
    index["occurrences"].as_array().unwrap().iter().find(|r| r["path"] == name).unwrap()["uuid"].as_str().unwrap().to_owned()
}

fn rig(mode: &str, items: &[String]) -> Analysis {
    run("/p/rig.assembly", RIG, mode, items).unwrap_or_else(|e| panic!("{mode} {items:?}: {e}"))
}

#[test]
fn components_measure_their_nearest_material() {
    let (block, boss) = (uuid("block"), uuid("boss"));
    let a = rig("distance", &[format!("occurrence:{block}"), format!("occurrence:{boss}")]);
    // The boss's rim at x = 140 mm (centre 150, radius 10) to the block's end at 100; 30 mm up
    // from the block's top at 10: √(40² + 20²).
    within(value(&a, "Minimum distance"), 40.0f64.hypot(20.0), 0.05);
    assert_eq!(a.settings, vec!["block".to_owned(), "boss".to_owned()]);
}

#[test]
fn entities_across_components_follow_their_placement() {
    let (block, boss) = (uuid("block"), uuid("boss"));
    let (bd, sd) = (display(BLOCK), display(BOSS));
    let block_top = format!("occurrence:{block}/{}", face(&bd, [50.0 * MM, 5.0 * MM, 10.0 * MM]));
    let boss_bottom = format!("occurrence:{boss}/{}", face(&sd, [0.0, 0.0, 0.0]));
    let a = rig("distance", &[block_top.clone(), boss_bottom.clone()]);
    close(value(&a, "Normal distance"), 20.0);
    let rim = format!("occurrence:{boss}/{}", circle_edge(&sd, 8.0 * MM));
    let r = rig("radius", &[rim.clone()]);
    close(value(&r, "Radius"), 10.0);
    // The rim's centre sits at the boss's placement.
    assert!(near(r.dimensions[0].from, [150.0 * MM, 5.0 * MM, 38.0 * MM]), "{:?}", r.dimensions[0]);
    let edge_to_face = rig("distance", &[rim, block_top]);
    close(value(&edge_to_face, "Normal distance"), 28.0);
    // A circle parallel to a plane: no angle between them.
    assert!(!has(&edge_to_face, "Angle"), "{:?}", edge_to_face.rows);
    let vertex = format!("occurrence:{block}/{}", vertex(&bd, [100.0 * MM, 10.0 * MM, 10.0 * MM]));
    let v = rig("distance", &[vertex, format!("occurrence:{boss}")]);
    // (100, 10, 10) to the boss's bottom rim: 50.25 − 10 across, 20 down.
    within(value(&v, "Minimum distance"), (50.0f64.hypot(5.0) - 10.0).hypot(20.0), 0.05);
}

#[test]
fn the_browser_and_the_agent_reach_measure_through_analyze() {
    let b = block();
    let out: Value = serde_json::from_str(&analyze_json(
        "/p/block.part",
        BLOCK,
        &files(),
        &json!({ "command": "view.measure", "values": { "mode": "distance", "items": [b.left, b.right], "pin": true } }).to_string(),
    ))
    .unwrap();
    assert_eq!(out["ok"], true, "{out}");
    assert_eq!(out["title"], "Measure");
    assert!(out["summary"].as_str().unwrap().starts_with("Minimum distance 100.000 mm"), "{out}");
    assert_eq!(out["dimensions"][0]["label"], "100.000 mm");
    let none: Value = serde_json::from_str(&analyze_json("/p/block.part", BLOCK, &files(), &json!({ "command": "view.measure" }).to_string())).unwrap();
    assert_eq!(none["ok"], false);
    assert!(none["message"].as_str().unwrap().starts_with("pick one or two items"), "{none}");
}

#[test]
fn reference_picks_resolve_like_topology_picks() {
    let d = display(BLOCK);
    let named = d.bodies[0].faces.iter().find(|f| f.reference.is_some()).expect("a named face");
    let by_reference = part("distance", &[named.reference.clone().unwrap()]);
    let by_index = part("distance", &[format!("face:{}:{}", d.bodies[0].index, named.index)]);
    assert_eq!(by_reference.rows, by_index.rows);
}
