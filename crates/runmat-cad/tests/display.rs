//! The display projection: tessellation through the kernel, joined with persistent names.

mod common;
use common::*;
use runmat_cad::display::{part_display, sketch_display, PartDisplay};
use runmat_cad::model::*;
use runmat_cad::refs::*;
use runmat_cad::regen::engine::RegenResult;
use runmat_cad::regen::kernel::{TessellationOptions, Tessellator};

fn display(d: &PartDocument) -> (RegenResult, PartDisplay) {
    let env = Env::new();
    let r = env.regen(d);
    let shown = part_display(&r, &env.kernel, &TessellationOptions::default());
    (r, shown)
}

fn tri_area(p: &[f32], t: &[u32]) -> f64 {
    t.chunks(3)
        .map(|t| {
            let v = |i: u32| {
                let i = i as usize * 3;
                [p[i] as f64, p[i + 1] as f64, p[i + 2] as f64]
            };
            let (a, b, c) = (v(t[0]), v(t[1]), v(t[2]));
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let n = [
                u[1] * w[2] - u[2] * w[1],
                u[2] * w[0] - u[0] * w[2],
                u[0] * w[1] - u[1] * w[0],
            ];
            0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
        })
        .sum()
}

/// Every face of the block draws with the area the kernel reports, wound outwards, and carries
/// the reference text a command field takes.
#[test]
fn block_faces_match_topology_and_names() {
    let (r, shown) = display(&block());
    assert_clean(&r);
    assert_eq!(shown.bodies.len(), 1);
    let body = &shown.bodies[0];
    let topo = &r.state.bodies[0].topology;
    assert_eq!(body.faces.len(), topo.faces.len());
    assert_eq!(body.edges.len(), topo.edges.len());
    assert_eq!(body.vertices.len(), topo.vertices.len());
    assert_eq!(body.reference.as_deref(), Some("feature.body.body"));
    for (face, t) in body.faces.iter().zip(&topo.faces) {
        let area = tri_area(&face.positions, &face.triangles);
        assert!(
            (area - t.geom.area).abs() < 1e-9,
            "face {} draws {area}, kernel says {}",
            face.index,
            t.geom.area
        );
        // Outward: the triangle normal points away from the body's centroid.
        let p = &face.positions;
        let i = |k: usize| face.triangles[k] as usize * 3;
        let a = [p[i(0)], p[i(0) + 1], p[i(0) + 2]];
        let b = [p[i(1)], p[i(1) + 1], p[i(1) + 2]];
        let c = [p[i(2)], p[i(2) + 1], p[i(2) + 2]];
        let n = [
            (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
            (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
        ];
        let centre = [0.04f32, 0.025, 0.005];
        let out = [a[0] - centre[0], a[1] - centre[1], a[2] - centre[2]];
        assert!(
            n[0] * out[0] + n[1] * out[1] + n[2] * out[2] > 0.0,
            "face {} winds inwards",
            face.index
        );
        assert_eq!(face.features, vec!["body".to_string()]);
    }
    let refs: Vec<&str> = body
        .faces
        .iter()
        .filter_map(|f| f.reference.as_deref())
        .collect();
    for expected in [
        "feature.body.face.end",
        "feature.body.face.start",
        "feature.body.face.side(s1.l1)",
    ] {
        assert!(refs.contains(&expected), "{expected} not in {refs:?}");
    }
    assert!(body.edges.iter().all(|e| e
        .reference
        .as_deref()
        .is_some_and(|r| r.starts_with("edges_between("))
        && e.points.len() == 6));
    assert!(body.vertices.iter().all(|v| v
        .reference
        .as_deref()
        .is_some_and(|r| r.starts_with("vertex_of("))));
    let b = shown.bounds.unwrap();
    assert_eq!(b.min, [0.0, 0.0, 0.0]);
    assert!(
        (b.max[0] - 0.08).abs() < 1e-6
            && (b.max[1] - 0.05).abs() < 1e-6
            && (b.max[2] - 0.01).abs() < 1e-6
    );
}

/// A fillet on a convex edge draws as a rounded strip between its tangent lines, and the faces
/// it meets are trimmed back to them: the shown area stays close to the filleted solid's.
#[test]
fn convex_fillet_draws_rounded_and_trims_neighbours() {
    let mut d = block();
    d.features.push(fillet(
        "round",
        vec![Reference::edges_between(
            side("body", "s1", "l1"),
            Reference::end_cap("body"),
        )],
        2.0,
    ));
    let (r, shown) = display(&d);
    assert_clean(&r);
    let body = &shown.bodies[0];
    let blend = body
        .faces
        .iter()
        .find(|f| f.features.contains(&"round".to_string()))
        .expect("the blend face draws");
    assert!(
        blend.triangles.len() / 3 >= 4,
        "the blend is curved, not one quad"
    );
    // A quarter cylinder of radius 2 mm along the 80 mm edge.
    let expected = std::f64::consts::FRAC_PI_2 * 0.002 * 0.08;
    let area = tri_area(&blend.positions, &blend.triangles);
    assert!(
        (area - expected).abs() / expected < 0.02,
        "blend area {area} vs {expected}"
    );
    let end = body
        .faces
        .iter()
        .find(|f| f.reference.as_deref() == Some("feature.body.face.end"))
        .unwrap();
    let end_area = tri_area(&end.positions, &end.triangles);
    assert!(
        (end_area - (0.08 * 0.05 - 0.08 * 0.002)).abs() < 1e-9,
        "end cap trimmed: {end_area}"
    );
    let tangent = body
        .edges
        .iter()
        .filter(|e| e.faces.contains(&blend.index))
        .count();
    assert_eq!(tangent, 2, "two tangent edges bound the blend");
}

/// The tessellator refuses a body another kernel made.
#[test]
fn foreign_body_is_refused() {
    let env = Env::new();
    let r = env.regen(&block());
    let mut handle = r.state.bodies[0].handle.clone();
    handle.kernel.name = "other".into();
    assert!(env
        .kernel
        .tessellate(&handle, &TessellationOptions::default())
        .is_err());
}

/// Sketch mode draws each solved sketch on its plane: entities with their solved points.
#[test]
fn sketches_draw_on_their_planes() {
    let d = block();
    let env = Env::new();
    let r = env.regen(&d);
    let sketches = sketch_display(&d, &r);
    assert_eq!(sketches.len(), 1);
    let s1 = &sketches[0];
    assert_eq!(s1.id, "s1");
    assert_eq!(s1.frame.z, [0.0, 0.0, 1.0]);
    assert_eq!(s1.entities.len(), 4);
    let l1 = s1.entities.iter().find(|e| e.id == "l1").unwrap();
    assert_eq!(l1.kind, "line");
    assert_eq!(l1.points["start"], [0.0, 0.0]);
    assert!((l1.points["end"][0] - 0.08).abs() < 1e-6);
}
