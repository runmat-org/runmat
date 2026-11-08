//! The advanced mates: width, symmetric, path, linear coupler, hinge and universal joint, each
//! solved in a small mechanism of published frames.

use runmat_cad::assembly::resolve::*;
use runmat_cad::assembly::source::*;
use runmat_cad::assembly::*;
use runmat_cad::ids::{FilePath, PublishName};
use runmat_cad::model::datum::Frame;
use runmat_cad::units::StubUnits;
use runmat_cad::value::{NoDynamicValues, ValueExpr};

fn run(src: &MemorySource) -> ResolvedAssembly {
    let (units, dynamic) = (StubUnits, NoDynamicValues);
    let ctx = ResolveContext {
        source: src,
        units: &units,
        dynamic: &dynamic,
        lock: None,
        catalog: None,
        tables: None,
        gauge: None,
    };
    resolve(&FilePath::new("m.assembly"), &ResolveRequest::default(), &ctx)
}

fn errors(r: &ResolvedAssembly) -> Vec<String> {
    r.diagnostics.errors().map(|d| format!("{} {}", d.code, d.message)).collect()
}

fn close(a: [f64; 3], b: [f64; 3]) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() < 1e-9)
}

fn axes(origin: [f64; 3], x: [f64; 3], z: [f64; 3]) -> Frame {
    let y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
    Frame { origin, x, y, z }
}

fn up(x: f64, y: f64, z: f64) -> Frame {
    axes([x, y, z], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
}

/// Z along world X (a guide along X).
fn along_x() -> Frame {
    axes([0.0; 3], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0])
}

/// `base` fixed, the other parts floating at the origin; each part publishes its frames (and
/// paths), and each frame is a connector `<part>.<frame>`.
fn mechanism(parts: &[(&str, Vec<(&str, Frame)>)], mates: Vec<(&str, MateKind, &str, &str)>) -> MemorySource {
    mechanism_with(parts, mates, &[])
}

/// [`mechanism`], with `paths` (part, name, points) published too.
fn mechanism_with(parts: &[(&str, Vec<(&str, Frame)>)], mates: Vec<(&str, MateKind, &str, &str)>, paths: &[(&str, &str, Vec<[f64; 3]>)]) -> MemorySource {
    let mut src = MemorySource::default();
    let mut top = AssemblyDocument::new("m.assembly", "m");
    for (name, frames) in parts {
        let file = format!("{name}.part");
        let mut s = DefinitionSummary::new(DefinitionRef::part(&file, None));
        s.part_number = Some((*name).into());
        let mut publishes: std::collections::BTreeMap<_, _> = frames.iter().map(|(n, f)| (PublishName::new(*n), PublishedGeometry::Frame { frame: *f })).collect();
        for (_, path, points) in paths.iter().filter(|(p, _, _)| p == name) {
            let chains = vec![PathChain { key: (*path).into(), points: points.clone(), normals: Vec::new() }];
            publishes.insert(PublishName::new(*path), PublishedGeometry::Path { chains });
        }
        src.add_part(s, ConfiguredSummary { bbox: Some(Aabb::new([0.0; 3], [0.01; 3])), publishes, ..Default::default() });
        let mut o = Occurrence::new(name, DefinitionRef::part(&file, None));
        o.fixed = *name == "base";
        top.occurrences.push(o);
        for (f, _) in frames {
            top.connectors.push(MateConnector {
                name: format!("{name}.{f}"),
                occurrence: RelPath::new(&[name]),
                anchor: AnchorRef::Published { name: (*f).into() },
                offset: None,
                flip: false,
                at: None,
                z: None,
                x: None,
            });
        }
    }
    for (name, kind, a, b) in mates {
        top.mates.push(MateDef { name: name.into(), kind, a: a.into(), b: b.into(), when: None, suppressed: false });
    }
    src.add_assembly(top);
    src
}

fn at(r: &ResolvedAssembly, name: &str) -> OccNode {
    r.node(r.find_display(name).unwrap()).clone()
}

fn slide(mm: f64) -> MateKind {
    MateKind::Slider { distance: Some(TargetExpr::Exact { value: ValueExpr::mm(mm) }) }
}

#[test]
fn a_width_mate_centres_the_tab_in_the_groove() {
    // A groove 20 mm wide (faces at z = 0 and 20 mm) and a tab 10 mm thick.
    let parts = [
        ("base", vec![("g1", up(0.0, 0.0, 0.0)), ("g2", up(0.0, 0.0, 0.02))]),
        ("tab", vec![("t1", up(0.0, 0.0, 0.0)), ("t2", up(0.0, 0.0, 0.01))]),
    ];
    let width = MateKind::Width { a2: "base.g2".into(), b2: "tab.t2".into() };
    let r = run(&mechanism(&parts, vec![("centre", width, "base.g1", "tab.t1")]));
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    let tab = at(&r, "tab");
    assert!((tab.world.translation[2] - 0.005).abs() < 1e-9, "{:?}", tab.world.translation);
    assert_eq!(tab.dof, Some(3), "the tab still slides in the groove's plane and turns in it");
    // The groove's faces on two components are refused by name.
    let parts = [
        ("base", vec![("g1", up(0.0, 0.0, 0.0))]),
        ("other", vec![("g2", up(0.0, 0.0, 0.02))]),
        ("tab", vec![("t1", up(0.0, 0.0, 0.0)), ("t2", up(0.0, 0.0, 0.01))]),
    ];
    let width = MateKind::Width { a2: "other.g2".into(), b2: "tab.t2".into() };
    let r = run(&mechanism(&parts, vec![("centre", width, "base.g1", "tab.t1")]));
    assert!(errors(&r).iter().any(|e| e.contains("groove's two faces are on different components")), "{:?}", errors(&r));
}

#[test]
fn a_symmetric_mate_mirrors_one_origin_into_the_other() {
    // The plane x = 50 mm; p slides 20 mm along X, so q sits at 80 mm.
    let plane = axes([0.05, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]);
    let parts = [
        ("base", vec![("guide", along_x()), ("mid", plane)]),
        ("p", vec![("rail", along_x()), ("o", up(0.0, 0.0, 0.0))]),
        ("q", vec![("o", up(0.0, 0.0, 0.0))]),
    ];
    let r = run(&mechanism(
        &parts,
        vec![("slide", slide(20.0), "base.guide", "p.rail"), ("mirror", MateKind::Symmetric { plane: "base.mid".into() }, "p.o", "q.o")],
    ));
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    let q = at(&r, "q");
    assert!(close(q.world.translation, [0.08, 0.0, 0.0]), "{:?}", q.world.translation);
    assert_eq!(q.dof, Some(3), "symmetry places q's origin and leaves its turn");
    // An undeclared plane is named.
    let r = run(&mechanism(&parts, vec![("mirror", MateKind::Symmetric { plane: "base.nowhere".into() }, "p.o", "q.o")]));
    assert!(errors(&r).iter().any(|e| e.contains("nowhere")), "{:?}", errors(&r));
}

#[test]
fn a_path_mate_keeps_a_point_on_the_published_path() {
    let rail = vec![[0.02, -0.05, 0.01], [0.04, -0.01, 0.0], [0.1, 0.05, 0.03]];
    let parts = [("base", vec![("frame", up(0.0, 0.0, 0.0))]), ("car", vec![("pin", up(0.0, 0.0, 0.0))])];
    let follow = MateKind::Path { path: PublishName::new("rail") };
    let r = run(&mechanism_with(&parts, vec![("follow", follow, "base.frame", "car.pin")], &[("base", "rail", rail.clone())]));
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    let car = at(&r, "car");
    let p = car.world.translation;
    // On one of the two segments.
    let off = rail
        .windows(2)
        .map(|w| {
            let e = [w[1][0] - w[0][0], w[1][1] - w[0][1], w[1][2] - w[0][2]];
            let q = [p[0] - w[0][0], p[1] - w[0][1], p[2] - w[0][2]];
            let t = ((q[0] * e[0] + q[1] * e[1] + q[2] * e[2]) / (e[0] * e[0] + e[1] * e[1] + e[2] * e[2])).clamp(0.0, 1.0);
            (0..3).map(|k| (q[k] - e[k] * t).powi(2)).sum::<f64>().sqrt()
        })
        .fold(f64::INFINITY, f64::min);
    assert!(off < 1e-9, "the car's pin is {off} m off the rail at {p:?}");
    assert_eq!(car.dof, Some(4), "it runs along the rail and turns freely");
    // A path the part doesn't publish says so.
    let src = mechanism(&parts, vec![("follow", MateKind::Path { path: PublishName::new("track") }, "base.frame", "car.pin")]);
    let r = run(&src);
    assert!(errors(&r).iter().any(|e| e.contains("publishes no `track`")), "{:?}", errors(&r));
}

#[test]
fn a_linear_coupler_moves_one_slider_by_a_ratio_of_the_other() {
    let along_y = axes([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let parts = [
        ("base", vec![("gx", along_x()), ("gy", along_y)]),
        ("p", vec![("rail", along_x())]),
        ("q", vec![("rail", along_y)]),
    ];
    let r = run(&mechanism(
        &parts,
        vec![
            ("sp", slide(30.0), "base.gx", "p.rail"),
            ("sq", MateKind::Slider { distance: None }, "base.gy", "q.rail"),
            ("couple", MateKind::LinearCoupler { ratio: ValueExpr::number(2.0) }, "p.rail", "q.rail"),
        ],
    ));
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    let q = at(&r, "q");
    assert!(close(q.world.translation, [0.0, 0.06, 0.0]), "{:?}", q.world.translation);
    assert_eq!(q.dof, Some(0));
}

#[test]
fn a_hinge_is_concentric_and_coincident_with_an_angle_range() {
    let parts = [("base", vec![("pin", up(0.0, 0.0, 0.0))]), ("door", vec![("eye", up(0.0, 0.0, 0.0))])];
    let r = run(&mechanism(&parts, vec![("hinge", MateKind::Hinge { angle: None }, "base.pin", "door.eye")]));
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    assert_eq!(at(&r, "door").dof, Some(1), "a hinge leaves the turn about its axis");
    // Driven to 90°: fully placed, its X along world Y.
    let driven = MateKind::Hinge { angle: Some(TargetExpr::Exact { value: ValueExpr::deg(90.0) }) };
    let r = run(&mechanism(&parts, vec![("hinge", driven, "base.pin", "door.eye")]));
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    let door = at(&r, "door");
    assert_eq!(door.dof, Some(0));
    assert!(close(door.world.apply_vector([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
    assert!(close(door.world.translation, [0.0; 3]));
}

#[test]
fn a_universal_joint_turns_the_output_shaft_by_the_cardan_law() {
    // The input turns 30° about Z; the output shaft is tilted 30° in XZ. tan φ = tan θ / cos β.
    let (theta, beta) = (30f64.to_radians(), 30f64.to_radians());
    let tilted = axes([0.0; 3], [0.0, 1.0, 0.0], [beta.sin(), 0.0, beta.cos()]);
    let parts = [
        ("base", vec![("input", up(0.0, 0.0, 0.0)), ("output", tilted)]),
        ("shaft_in", vec![("eye", up(0.0, 0.0, 0.0))]),
        ("shaft_out", vec![("eye", tilted)]),
    ];
    let driven = MateKind::Revolute { angle: Some(TargetExpr::Exact { value: ValueExpr::deg(30.0) }) };
    let r = run(&mechanism(
        &parts,
        vec![
            ("in", driven, "base.input", "shaft_in.eye"),
            ("out", MateKind::Revolute { angle: None }, "base.output", "shaft_out.eye"),
            ("joint", MateKind::UniversalJoint { joint_point: false }, "shaft_in.eye", "shaft_out.eye"),
        ],
    ));
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    let out = at(&r, "shaft_out");
    assert_eq!(out.dof, Some(0), "the joint takes the output shaft's turn");
    let phi = (theta.tan() / beta.cos()).atan();
    let arm = out.world.apply_vector([0.0, 1.0, 0.0]);
    let want = [-phi.sin() * beta.cos(), phi.cos(), phi.sin() * beta.sin()];
    assert!(close(arm, want), "arm {arm:?}, want {want:?} (φ = {}°)", phi.to_degrees());
    // With its joint point and no bearings, the output keeps two turns about the cross.
    let r = run(&mechanism(
        &parts,
        vec![
            ("in", MateKind::Revolute { angle: Some(TargetExpr::Exact { value: ValueExpr::deg(30.0) }) }, "base.input", "shaft_in.eye"),
            ("joint", MateKind::UniversalJoint { joint_point: true }, "shaft_in.eye", "shaft_out.eye"),
        ],
    ));
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    assert_eq!(at(&r, "shaft_out").dof, Some(2));
}
