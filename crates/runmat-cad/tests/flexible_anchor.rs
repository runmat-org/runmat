//! Flexible sub-assemblies whose own occurrence floats in the parent: the anchor (the
//! sub-assembly's fixed member) is fastened to the sub-assembly's frame, the frame floats
//! with the parent's mates, and a sub-assembly with no anchor keeps its placed frame.

use runmat_cad::assembly::resolve::*;
use runmat_cad::assembly::source::*;
use runmat_cad::assembly::*;
use runmat_cad::diagnostics::DiagCode;
use runmat_cad::ids::{FilePath, PublishName};
use runmat_cad::model::datum::Frame;
use runmat_cad::units::StubUnits;
use runmat_cad::value::{NoDynamicValues, ValueExpr};

fn frame_at(x: f64, y: f64, z: f64) -> PublishedGeometry {
    PublishedGeometry::Frame {
        frame: Frame {
            origin: [x, y, z],
            ..Frame::WORLD
        },
    }
}

fn add_part(src: &mut MemorySource, file: &str, publishes: Vec<(&str, PublishedGeometry)>) {
    let mut s = DefinitionSummary::new(DefinitionRef::part(file, None));
    s.part_number = Some(file.to_uppercase());
    src.add_part(
        s,
        ConfiguredSummary {
            bbox: Some(Aabb::new([0.0; 3], [0.01; 3])),
            publishes: publishes
                .into_iter()
                .map(|(k, v)| (PublishName::new(k), v))
                .collect(),
            ..Default::default()
        },
    );
}

fn connector(name: &str, path: &[&str], anchor: &str) -> MateConnector {
    MateConnector {
        name: name.into(),
        occurrence: RelPath::new(path),
        anchor: AnchorRef::Published {
            name: anchor.into(),
        },
        offset: None,
        flip: false,
        at: None,
        z: None,
        x: None,
    }
}

fn mate(name: &str, kind: MateKind, a: &str, b: &str) -> MateDef {
    MateDef {
        name: name.into(),
        kind,
        a: a.into(),
        b: b.into(),
        when: None,
        suppressed: false,
    }
}

fn at(x: f64, y: f64, z: f64) -> Placement {
    Placement::Offset {
        offset: OffsetExpr::translation([ValueExpr::mm(x), ValueExpr::mm(y), ValueExpr::mm(z)]),
    }
}

fn close(a: [f64; 3], b: [f64; 3]) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() < 1e-7)
}

/// `arm.assembly`: `shoulder` (the anchor unless `anchor_floats`) and `fore`, joined by the
/// arm's own revolute driven to 90°. The top fastens the shoulder to the base's pin, through
/// the arm, which floats and is flexible in the top.
fn rig(anchor_floats: bool, fasten_to_base: bool) -> MemorySource {
    let mut src = MemorySource::default();
    add_part(
        &mut src,
        "base.part",
        vec![("pin", frame_at(0.5, 0.0, 0.1))],
    );
    add_part(
        &mut src,
        "link.part",
        vec![
            ("eye", frame_at(0.0, 0.0, 0.0)),
            ("tip", frame_at(0.2, 0.0, 0.0)),
        ],
    );
    let mut arm = AssemblyDocument::new("arm.assembly", "arm");
    let mut shoulder = Occurrence::new("shoulder", DefinitionRef::part("link.part", None));
    shoulder.placement = at(100.0, 0.0, 0.0);
    shoulder.fixed = !anchor_floats;
    arm.occurrences.push(shoulder);
    let mut fore = Occurrence::new("fore", DefinitionRef::part("link.part", None));
    fore.fixed = false;
    fore.placement = at(300.0, 50.0, 0.0);
    arm.occurrences.push(fore);
    arm.connectors.push(connector("tip", &["shoulder"], "tip"));
    arm.connectors.push(connector("eye", &["fore"], "eye"));
    arm.mates.push(mate(
        "elbow",
        MateKind::Revolute {
            angle: Some(TargetExpr::Exact {
                value: ValueExpr::deg(90.0),
            }),
        },
        "tip",
        "eye",
    ));
    src.add_assembly(arm);

    let mut top = AssemblyDocument::new("top.assembly", "top");
    top.occurrences.push(Occurrence::new(
        "base",
        DefinitionRef::part("base.part", None),
    ));
    let mut a = Occurrence::new("arm", DefinitionRef::assembly("arm.assembly"));
    a.fixed = false;
    a.flexible = true;
    a.placement = at(1000.0, 2000.0, 3000.0);
    top.occurrences.push(a);
    top.connectors.push(connector("pin", &["base"], "pin"));
    top.connectors
        .push(connector("shoulder_eye", &["arm", "shoulder"], "eye"));
    if fasten_to_base {
        top.mates
            .push(mate("mount", MateKind::Fasten, "pin", "shoulder_eye"));
    }
    src.add_assembly(top);
    src
}

fn run(src: &MemorySource) -> ResolvedAssembly {
    let ctx = ResolveContext {
        source: src,
        units: &StubUnits,
        dynamic: &NoDynamicValues,
        lock: None,
        catalog: None,
        tables: None,
        gauge: None,
    };
    resolve(
        &FilePath::new("top.assembly"),
        &ResolveRequest::default(),
        &ctx,
    )
}

fn errors(r: &ResolvedAssembly) -> Vec<String> {
    r.diagnostics
        .errors()
        .map(|d| format!("{} {}", d.code, d.message))
        .collect()
}

#[test]
fn f01_anchor_of_a_floating_flexible_sub_assembly_follows_the_parent_mates() {
    let r = run(&rig(false, true));
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    let arm = r.find_display("arm").unwrap();
    let shoulder = r.find_display("arm/shoulder").unwrap();
    let fore = r.find_display("arm/fore").unwrap();
    // The anchor sits on the pin, and keeps its placement inside the arm.
    assert!(
        close(r.node(shoulder).world.translation, [0.5, 0.0, 0.1]),
        "{:?}",
        r.node(shoulder).world.translation
    );
    assert!(close(r.node(shoulder).local.translation, [0.1, 0.0, 0.0]));
    assert!(
        close(r.node(arm).world.translation, [0.4, 0.0, 0.1]),
        "the arm's frame moves with its anchor: {:?}",
        r.node(arm).world.translation
    );
    // The fore link hangs off the shoulder's tip, turned a quarter turn about Z.
    let f = r.node(fore).world;
    assert!(close(f.translation, [0.7, 0.0, 0.1]), "{:?}", f.translation);
    let x_axis = f.apply_vector([1.0, 0.0, 0.0]);
    assert!(
        (x_axis[1].abs() - 1.0).abs() < 1e-7,
        "fore's X is along ±Y: {:?}",
        x_axis
    );
    for n in [arm, shoulder, fore] {
        assert_eq!(r.node(n).pose, PoseStatus::Solved, "{}", r.path(n));
        assert_eq!(r.node(n).dof, Some(0), "{}", r.path(n));
    }
    assert!(r
        .diagnostics
        .with_code(DiagCode::AsmUnderDefined)
        .is_empty());
    assert_eq!(
        r.stats.mate_scopes, 1,
        "one solve for the top and the flexible arm"
    );
    let statuses: Vec<(&str, &MateStatus)> = r
        .mates
        .iter()
        .map(|m| (m.name.as_str(), &m.status))
        .collect();
    assert_eq!(
        statuses,
        vec![
            ("mount", &MateStatus::Satisfied),
            ("elbow", &MateStatus::Satisfied)
        ]
    );
}

#[test]
fn f02_a_floating_anchor_leaves_the_sub_assembly_frame_where_it_was_placed() {
    // Every member of the arm floats: nothing ties the arm's own frame to its members, so the
    // frame keeps its placement (it is not reported as a free body) and the members solve.
    let r = run(&rig(true, true));
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    let arm = r.find_display("arm").unwrap();
    let shoulder = r.find_display("arm/shoulder").unwrap();
    let fore = r.find_display("arm/fore").unwrap();
    assert!(close(r.node(arm).world.translation, [1.0, 2.0, 3.0]));
    assert_eq!(r.node(arm).pose, PoseStatus::Placed);
    assert!(close(r.node(shoulder).world.translation, [0.5, 0.0, 0.1]));
    assert!(close(r.node(fore).world.translation, [0.7, 0.0, 0.1]));
    // Local transforms are relative to the arm's placed frame.
    assert!(close(
        r.node(shoulder).local.translation,
        [-0.5, -2.0, -2.9]
    ));
    for n in [shoulder, fore] {
        assert_eq!(r.node(n).pose, PoseStatus::Solved, "{}", r.path(n));
    }
    assert!(r
        .diagnostics
        .with_code(DiagCode::AsmUnderDefined)
        .is_empty());
}

#[test]
fn f03_an_unmated_floating_flexible_sub_assembly_reports_the_frame_not_its_anchor() {
    // No parent mate: the arm (with its anchor fastened to it) still moves as one body. The
    // under-defined note names the arm and its floating member, not the anchor riding on it.
    let r = run(&rig(false, false));
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    let notes = r.diagnostics.with_code(DiagCode::AsmUnderDefined);
    assert_eq!(notes.len(), 1, "{:?}", notes);
    assert_eq!(
        notes[0].message,
        "2 occurrence(s) still move: arm (6 DOF), arm/fore (6 DOF)"
    );
    let arm = r.find_display("arm").unwrap();
    let shoulder = r.find_display("arm/shoulder").unwrap();
    assert_eq!(r.node(arm).pose, PoseStatus::Floating);
    // Wherever the free cluster settles, the anchor keeps its placement in the arm and the
    // fore link stays on the shoulder's tip.
    assert!(close(r.node(shoulder).local.translation, [0.1, 0.0, 0.0]));
    let fore = r.find_display("arm/fore").unwrap();
    let tip = r.node(shoulder).world.apply([0.2, 0.0, 0.0]);
    assert!(close(r.node(fore).world.translation, tip));
}
