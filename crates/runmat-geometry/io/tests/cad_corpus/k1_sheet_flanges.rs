//! K1 sheet metal on exact geometry: edge flanges on flat plates and their flat patterns.
//!
//! The plate is 80 × 40 mm, 2 mm thick, bend radius 3 mm, K = 0.44, so a 90° bend's allowance
//! is BA = π/2 (R + K t). A flange of flat length L grown from an edge develops as a strip of
//! BA + L along that edge, starting at the flange's setback from the edge.

use crate::common::{fid, mm, Env};
use crate::families::{plate_doc, plate_top};
use crate::k1_ops::{close, volume};
use runmat_cad::model::sheet_metal::*;
use runmat_cad::model::*;
use runmat_cad::refs::history::SurfaceKind;
use runmat_cad::refs::name::{GenerationName, PlaneSide, Role, SourceTag};
use runmat_cad::refs::{EntityName, Reference};
use runmat_cad::regen::kernel::KernelBody;
use runmat_cad::regen::records::FeatureRecord;
use runmat_cad::regen::RegenResult;
use runmat_cad::value::ValueExpr;

use crate::kernel_under_test::KernelUnderTest;

const T: f64 = 0.002;
const R: f64 = 0.003;
const K: f64 = 0.44;

fn ba() -> f64 {
    std::f64::consts::FRAC_PI_2 * (R + K * T)
}

fn face(feature: &str, role: Role, key: &str, ordinal: Option<u32>) -> Reference {
    Reference::face(EntityName::Generated(GenerationName {
        feature: fid(feature),
        role,
        tag: SourceTag::Key { key: key.into() },
        ordinal,
        step: None,
    }))
}

/// The plate's top edge along its side `side` (a sketch line of the plate's outline).
fn top_edge(side: &str) -> Reference {
    Reference::edges_between(plate_top(), face("base", Role::SheetEdge, side, None))
}

fn flange(id: &str, edge: Reference, length: f64, position: FlangePosition) -> Feature {
    Feature::new(
        id,
        FeatureKind::SheetMetal(SheetMetalFeature::EdgeFlange {
            edges: vec![edge],
            length: mm(length),
            angle: ValueExpr::deg(90.0),
            length_reference: LengthReference::Tangent,
            position,
            radius: None,
            relief: None,
        }),
    )
}

fn with_flat(mut d: PartDocument) -> PartDocument {
    d.features.push(Feature::new(
        "flat",
        FeatureKind::SheetMetal(SheetMetalFeature::FlatPattern {
            fixed_face: plate_top(),
            grain: None,
        }),
    ));
    d
}

fn flat_volume(out: &RegenResult) -> f64 {
    let flat = out
        .state
        .bodies
        .iter()
        .find(|b| b.names.body.iter().any(|n| n.maker() == &fid("flat")))
        .expect("the flat body");
    volume(
        &KernelUnderTest::new(),
        &KernelBody {
            handle: flat.handle.clone(),
            topology: flat.topology.clone(),
        },
    )
    .0
}

fn layout(out: &RegenResult) -> &FlatLayout {
    match out.state.records.get(&fid("flat")) {
        Some(FeatureRecord::FlatPattern { layout, .. }) => layout,
        other => panic!("no flat pattern: {other:?}"),
    }
}

fn unique_names(out: &RegenResult) {
    let mut seen = std::collections::BTreeSet::new();
    for n in out
        .state
        .bodies
        .iter()
        .flat_map(|b| b.names.faces.iter().flatten())
    {
        assert!(seen.insert(n.clone()), "{n} on two faces");
    }
}

/// A rectangle on the origin's YZ plane and a cut through everything along x.
fn window(d: &mut PartDocument, y0: f64, z0: f64, y1: f64, z1: f64) {
    use runmat_cad::model::datum::{DatumRef, OriginPlane, PlaneRef};
    d.features.push(crate::common::sketch(
        "s_window",
        crate::common::rect_names(
            PlaneRef::Datum {
                datum: DatumRef::OriginPlane {
                    plane: OriginPlane::Yz,
                },
            },
            y0,
            z0,
            y1,
            z1,
            ["w1", "w2", "w3", "w4"],
        ),
    ));
    d.features.push(Feature::new(
        "window",
        FeatureKind::Extrude(Extrude {
            profile: ProfileRef::sketch("s_window").into(),
            extent: Extent::Symmetric {
                distance: mm(400.0),
            },
            reverse: false,
            mode: BodyMode::cut(),
            from: Default::default(),
            second: None,
            draft: None,
            thin: None,
        }),
    ));
}

#[test]
fn k1_an_edge_flange_on_a_plate_develops_as_a_strip_of_its_allowance_and_flat() {
    let env = Env::new();
    let mut d = plate_doc();
    d.features.push(flange(
        "lip",
        top_edge("e2"),
        20.0,
        FlangePosition::BendOutside,
    ));
    // A 10 × 10 mm window through the flange's flat, 12 to 22 mm up.
    window(&mut d, 15.0, 12.0, 25.0, 22.0);
    let out = env.regen(&with_flat(d));
    assert!(out.status("lip").unwrap().ran(), "{:?}", out.diagnostics);
    assert!(out.status("flat").unwrap().ran(), "{:?}", out.diagnostics);
    // The plate, the strip BA + 20 mm along the 40 mm edge, less the window.
    let strip = ba() + 0.02;
    let want = T * (0.08 * 0.04 + strip * 0.04 - 0.01 * 0.01);
    let v = flat_volume(&out);
    assert!(close(v, want, 1e-9), "{v} vs {want}");
    let l = layout(&out);
    let (lo, hi) = l.extents();
    assert!(
        close(hi[0], 0.08 + strip, 1e-9) && close(lo[0], 0.0, 1e-9),
        "{lo:?} {hi:?}"
    );
    assert!(
        close(hi[1], 0.04, 1e-9) && close(lo[1], 0.0, 1e-9),
        "{lo:?} {hi:?}"
    );
    // The bend line runs across the strip's bend zone, mid-allowance; the flange turns up
    // (towards the top face, the layout's +z).
    assert_eq!(l.bend_lines.len(), 1, "{:?}", l.bend_lines);
    let b = &l.bend_lines[0];
    assert!(
        close(b.start[0], 0.08 + ba() / 2.0, 1e-9) && close(b.end[0], 0.08 + ba() / 2.0, 1e-9),
        "{b:?}"
    );
    assert!(close((b.end[1] - b.start[1]).abs(), 0.04, 1e-9), "{b:?}");
    assert!(close(b.allowance, ba(), 1e-12) && close(b.angle, std::f64::consts::FRAC_PI_2, 1e-12));
    assert_eq!(b.direction, BendDirection::Up);
    // The window lands on the strip: its sides across it sit at the flat's start (80 mm + BA)
    // plus their height above where the flange's flat starts (t + R).
    let flat = out
        .state
        .bodies
        .iter()
        .find(|b| b.names.body.iter().any(|n| n.maker() == &fid("flat")))
        .expect("the flat body");
    let mut across: Vec<f64> = flat
        .topology
        .faces
        .iter()
        .filter(
            |f| matches!(f.geom.surface, SurfaceKind::Plane { normal } if normal[0].abs() > 0.999),
        )
        .map(|f| f.geom.centroid[0])
        .filter(|x| *x > 0.08 + ba() + 1e-4 && *x < 0.08 + strip - 1e-4)
        .collect();
    across.sort_by(f64::total_cmp);
    across.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    let want = [0.08 + ba() + 0.012 - (T + R), 0.08 + ba() + 0.022 - (T + R)];
    assert_eq!(across.len(), 2, "{across:?}");
    for (got, w) in across.iter().zip(want) {
        assert!((got - w).abs() < 1e-9, "{across:?} vs {want:?}");
    }
    unique_names(&out);
}

#[test]
fn k1_flanges_on_two_sides_of_a_corner_develop_with_their_setbacks() {
    let env = Env::new();
    let mut d = plate_doc();
    // Material inside: the bend starts tan(45°)(R + t) = 5 mm inside the edge x = 80 mm.
    d.features.push(flange(
        "side",
        top_edge("e2"),
        20.0,
        FlangePosition::MaterialInside,
    ));
    d.features.push(flange(
        "back",
        top_edge("e3"),
        15.0,
        FlangePosition::BendOutside,
    ));
    let out = env.regen(&with_flat(d));
    assert!(out.status("back").unwrap().ran(), "{:?}", out.diagnostics);
    assert!(out.status("flat").unwrap().ran(), "{:?}", out.diagnostics);
    // The plate up to x = 75 mm, the side strip along x from 75 mm, the back strip along the
    // 75 mm the side flange left of the edge y = 40 mm.
    let (side, back) = (ba() + 0.02, ba() + 0.015);
    let want = T * (0.075 * 0.04 + side * 0.04 + back * 0.075);
    let v = flat_volume(&out);
    assert!(close(v, want, 1e-9), "{v} vs {want}");
    let l = layout(&out);
    let (_, hi) = l.extents();
    assert!(
        close(hi[0], 0.075 + side, 1e-9) && close(hi[1], 0.04 + back, 1e-9),
        "{hi:?}"
    );
    let mut at: Vec<(f64, f64)> = l
        .bend_lines
        .iter()
        .map(|b| (b.start[0], b.start[1]))
        .collect();
    at.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert_eq!(at.len(), 2, "{at:?}");
    assert!(
        l.bend_lines
            .iter()
            .any(|b| close(b.start[0], 0.075 + ba() / 2.0, 1e-9)
                && close(b.end[0], 0.075 + ba() / 2.0, 1e-9)),
        "{:?}",
        l.bend_lines
    );
    assert!(
        l.bend_lines
            .iter()
            .any(|b| close(b.start[1], 0.04 + ba() / 2.0, 1e-9)
                && close(b.end[1], 0.04 + ba() / 2.0, 1e-9)),
        "{:?}",
        l.bend_lines
    );
    unique_names(&out);
}

#[test]
fn k1_a_flange_on_a_bent_flat_develops_past_that_flats_edge() {
    let env = Env::new();
    let mut d = plate_doc();
    d.features.push(Feature::new(
        "bend",
        FeatureKind::SheetMetal(SheetMetalFeature::SketchedBend {
            face: plate_top(),
            line: (fid("s_line"), runmat_cad::ids::SketchEntityId::new("bl")),
            angle: ValueExpr::deg(90.0),
            radius: None,
            fixed: PlaneSide::Positive,
        }),
    ));
    // The turned-up flat's far edge (the plate's side x = 80 mm before the bend).
    let flat_side = face("bend", Role::SheetEdge, "e2", Some(0));
    let edge = (0..2)
        .map(|o| {
            Reference::edges_between(
                face("bend", Role::SheetFace, "bl.flat", Some(o)),
                flat_side.clone(),
            )
        })
        .collect::<Vec<_>>();
    d.features.push(flange(
        "lip",
        edge[1].clone(),
        20.0,
        FlangePosition::BendOutside,
    ));
    let out = env.regen(&with_flat(d));
    assert!(out.status("lip").unwrap().ran(), "{:?}", out.diagnostics);
    assert!(out.status("flat").unwrap().ran(), "{:?}", out.diagnostics);
    let strip = ba() + 0.02;
    let want = T * (0.08 * 0.04 + strip * 0.04);
    let v = flat_volume(&out);
    assert!(close(v, want, 1e-9), "{v} vs {want}");
    let l = layout(&out);
    let (_, hi) = l.extents();
    assert!(close(hi[0], 0.08 + strip, 1e-9), "{hi:?}");
    assert_eq!(l.bend_lines.len(), 2, "{:?}", l.bend_lines);
    assert!(
        l.bend_lines
            .iter()
            .any(|b| close(b.start[0], 0.08 + ba() / 2.0, 1e-9)),
        "{:?}",
        l.bend_lines
    );
    unique_names(&out);
}
