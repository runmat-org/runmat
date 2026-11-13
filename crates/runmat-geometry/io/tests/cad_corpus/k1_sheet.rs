//! K1 sheet metal on exact geometry: flat patterns of plates other features cut.

use crate::common::{fid, mm, rect_names, sketch, Env};
use crate::families::{plate_doc, plate_top};
use crate::k1_ops::{close, volume};
use crate::kernel_under_test::KernelUnderTest;
use runmat_cad::model::datum::{OriginPlane, PlaneRef, PointRef};
use runmat_cad::model::sheet_metal::*;
use runmat_cad::model::*;
use runmat_cad::refs::history::SurfaceKind;
use runmat_cad::refs::name::PlaneSide;
use runmat_cad::regen::kernel::KernelBody;
use runmat_cad::regen::RegenResult;
use runmat_cad::value::ValueExpr;

const T: f64 = 0.002;
const R: f64 = 0.003;
const K: f64 = 0.44;

/// The plate turned up 90° past x = 50 mm, a Ø8 hole through its unmoved part at (20, 20) mm
/// and a 10 × 10 mm window through the turned-up flat at y ∈ [15, 25], z ∈ [12, 22] mm.
fn cut_plate_doc() -> PartDocument {
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
    d.features.push(Feature::new(
        "hole",
        FeatureKind::Hole(Hole {
            face: plate_top(),
            positions: vec![HolePosition {
                key: "h1".into(),
                at: PointRef::Coords {
                    x: mm(20.0),
                    y: mm(20.0),
                    z: mm(2.0),
                },
            }],
            diameter: Some(mm(8.0)),
            end: HoleEnd::ThroughAll,
            form: HoleForm::Simple,
            spec: None,
            drill_point: None,
            series: None,
        }),
    ));
    d.features.push(window("s_window", 15.0, 12.0, 25.0, 22.0));
    d.features.push(window_cut("window", "s_window"));
    d
}

/// A rectangle on the origin's YZ plane (sketch x is world y, sketch y is world z).
fn window(id: &str, y0: f64, z0: f64, y1: f64, z1: f64) -> Feature {
    sketch(
        id,
        rect_names(
            PlaneRef::Datum {
                datum: runmat_cad::model::datum::DatumRef::OriginPlane {
                    plane: OriginPlane::Yz,
                },
            },
            y0,
            z0,
            y1,
            z1,
            ["w1", "w2", "w3", "w4"],
        ),
    )
}

/// A cut through everything along x.
fn window_cut(id: &str, sketch: &str) -> Feature {
    Feature::new(
        id,
        FeatureKind::Extrude(Extrude {
            profile: ProfileRef::sketch(sketch).into(),
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

fn flat_body(out: &RegenResult) -> &runmat_cad::regen::LiveBody {
    out.state
        .bodies
        .iter()
        .find(|b| b.names.body.iter().any(|n| n.maker() == &fid("flat")))
        .expect("the flat body")
}

#[test]
fn k1_a_cut_bent_plate_develops_its_holes_and_windows_onto_the_flat_pattern() {
    let env = Env::new();
    let out = env.regen(&with_flat(cut_plate_doc()));
    assert!(out.status("flat").unwrap().ran(), "{:?}", out.diagnostics);
    assert_eq!(out.state.bodies.len(), 2, "the part and its flat pattern");
    let flat = flat_body(&out);
    let k = KernelUnderTest::new();
    let v = volume(
        &k,
        &KernelBody {
            handle: flat.handle.clone(),
            topology: flat.topology.clone(),
        },
    )
    .0;
    // The plate less the hole and the window, each through the thickness.
    let want = T * (0.08 * 0.04 - std::f64::consts::PI * 0.004 * 0.004 - 0.01 * 0.01);
    assert!(close(v, want, 1e-9), "{v} vs {want}");
    // The hole stays where it was drilled: the unmoved part lies in place.
    let wall = flat
        .topology
        .faces
        .iter()
        .find_map(|f| match f.geom.surface {
            SurfaceKind::Cylinder {
                axis_point, radius, ..
            } => Some((axis_point, radius)),
            _ => None,
        })
        .expect("the hole's wall");
    assert!(
        close(wall.1, 0.004, 1e-9) && close(wall.0[0], 0.02, 1e-9) && close(wall.0[1], 0.02, 1e-9),
        "{wall:?}"
    );
    // The window lands on the developed flat: its edges across the strip sit at the moved
    // flat's start (50 mm + BA/2) plus their height above where the flat starts (t + R).
    let ba = std::f64::consts::FRAC_PI_2 * (R + K * T);
    let start = 0.05 + ba / 2.0;
    let mut across: Vec<f64> = flat
        .topology
        .faces
        .iter()
        .filter(
            |f| matches!(f.geom.surface, SurfaceKind::Plane { normal } if normal[0].abs() > 0.999),
        )
        .map(|f| f.geom.centroid[0])
        .filter(|x| *x > start + 1e-4 && *x < 0.08 - 1e-4)
        .collect();
    across.sort_by(f64::total_cmp);
    let want = [start + 0.012 - (T + R), start + 0.022 - (T + R)];
    assert_eq!(across.len(), 2, "{across:?}");
    for (got, w) in across.iter().zip(want) {
        assert!((got - w).abs() < 1e-9, "{across:?} vs {want:?}");
    }
    // Every face name is on one face only.
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

#[test]
fn k1_a_flat_pattern_refuses_material_cut_across_a_bend_zone() {
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
    // A window through the bend: z from 2.5 to 4.5 mm crosses the bend's arc and misses the
    // plate (z ≤ 2 mm) and the turned-up flat (z ≥ 5 mm).
    d.features.push(window("s_window", 15.0, 2.5, 25.0, 4.5));
    d.features.push(window_cut("window", "s_window"));
    let out = env.regen(&with_flat(d));
    assert!(out.status("window").unwrap().ran(), "{:?}", out.diagnostics);
    let refused =
        out.diagnostics.diagnostics.iter().any(|d| {
            d.message.contains("across a bend zone") && d.message.contains("lies on no flat")
        });
    assert!(refused, "{:?}", out.diagnostics);
}
