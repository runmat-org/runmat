//! K1 sheet metal on exact geometry: corner reliefs and closed corners where two edge flanges
//! of a flat plate meet.
//!
//! The plate is 80 × 40 mm, 2 mm thick, bend radius 3 mm, K = 0.44, so a 90° bend's allowance
//! is BA = π/2 (R + K t). The flanges stand on the edges x = 80 mm (`side`) and y = 40 mm
//! (`back`). Each check compares the part and its flat pattern with the same part without the
//! corner feature, against the closed-form change.

use crate::common::{fid, mm, Env};
use crate::families::{plate_doc, plate_top};
use crate::k1_ops::{close, volume};
use runmat_cad::model::sheet_metal::*;
use runmat_cad::model::*;
use runmat_cad::refs::name::{GenerationName, Role, SourceTag};
use runmat_cad::refs::{EntityName, Reference};
use runmat_cad::regen::kernel::KernelBody;
use runmat_cad::regen::RegenResult;
use runmat_cad::value::ValueExpr;
use std::f64::consts::{FRAC_PI_2, PI};

use crate::kernel_under_test::KernelUnderTest;

const T: f64 = 0.002;
const R: f64 = 0.003;
const K: f64 = 0.44;

fn ba() -> f64 {
    FRAC_PI_2 * (R + K * T)
}

fn top_edge(side: &str) -> Reference {
    Reference::edges_between(
        plate_top(),
        Reference::face(EntityName::Generated(GenerationName {
            feature: fid("base"),
            role: Role::SheetEdge,
            tag: SourceTag::Key { key: side.into() },
            ordinal: None,
            step: None,
        })),
    )
}

fn flange(id: &str, edge: &str, length: f64, position: FlangePosition) -> Feature {
    Feature::new(
        id,
        FeatureKind::SheetMetal(SheetMetalFeature::EdgeFlange {
            edges: vec![top_edge(edge)],
            length: mm(length),
            angle: ValueExpr::deg(90.0),
            length_reference: LengthReference::Tangent,
            position,
            radius: None,
            relief: None,
        }),
    )
}

/// The plate with `side` (placed `side_at`, 20 mm) and `back` (bend outside, 15 mm).
fn flanged(side_at: FlangePosition) -> PartDocument {
    let mut d = plate_doc();
    d.features.push(flange("side", "e2", 20.0, side_at));
    d.features
        .push(flange("back", "e3", 15.0, FlangePosition::BendOutside));
    d
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

/// The volumes of the part and of its flat pattern.
fn volumes(out: &RegenResult) -> (f64, f64) {
    let of = |flat: bool| {
        let b = out
            .state
            .bodies
            .iter()
            .find(|b| b.names.body.iter().any(|n| n.maker() == &fid("flat")) == flat)
            .expect("a body");
        volume(
            &KernelUnderTest::new(),
            &KernelBody {
                handle: b.handle.clone(),
                topology: b.topology.clone(),
            },
        )
        .0
    };
    (of(false), of(true))
}

/// A face `feature` made with `role` and `ordinal`, by the name the regeneration gave it.
fn named(out: &RegenResult, feature: &str, role: Role, ordinal: u32) -> Reference {
    out.state
        .bodies
        .iter()
        .flat_map(|b| b.names.faces.iter().flatten())
        .find_map(|n| match n {
            EntityName::Generated(g)
                if g.feature == fid(feature) && g.role == role && g.ordinal == Some(ordinal) =>
            {
                Some(Reference::face(n.clone()))
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("no {feature} {role:?} face {ordinal}"))
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

/// Regenerates `d` with and without `feature` (inserted before the flat pattern), checks
/// both ran and returns the volume changes (part, flat).
fn change(d: PartDocument, feature: impl Fn(&RegenResult) -> Feature) -> (f64, f64, RegenResult) {
    let env = Env::new();
    let plain = env.regen(&with_flat(d.clone()));
    assert!(
        plain.status("flat").unwrap().ran(),
        "{:?}",
        plain.diagnostics
    );
    let f = feature(&plain);
    let fid_ = f.id.clone();
    let mut with = d;
    with.features.push(f);
    let out = env.regen(&with_flat(with));
    assert!(out.status(&fid_.0).unwrap().ran(), "{:?}", out.diagnostics);
    assert!(out.status("flat").unwrap().ran(), "{:?}", out.diagnostics);
    let (p0, f0) = volumes(&plain);
    let (p1, f1) = volumes(&out);
    unique_names(&out);
    (p1 - p0, f1 - f0, out)
}

fn relief(kind: CornerReliefKind) -> impl Fn(&RegenResult) -> Feature {
    move |plain| {
        Feature::new(
            "relief",
            FeatureKind::SheetMetal(SheetMetalFeature::CornerRelief {
                corners: vec![named(plain, "side", Role::BendFace, 0)],
                kind: kind.clone(),
            }),
        )
    }
}

/// What a relief of `size` removes from each bend: the bend's annulus over `size` of its axis.
fn bend_piece(size: f64) -> f64 {
    FRAC_PI_2 * T * (R + T / 2.0) * size
}

#[test]
fn k1_a_rectangular_corner_relief_cuts_both_bend_zones_and_a_square_of_the_plate() {
    let d = 0.002;
    let (part, flat, out) = change(
        flanged(FlangePosition::MaterialInside),
        relief(CornerReliefKind::Rectangular { size: mm(2.0) }),
    );
    let flat_want = -T * (d * d + 2.0 * d * ba());
    assert!(close(flat, flat_want, 1e-9), "flat {flat} vs {flat_want}");
    let part_want = -(T * d * d + 2.0 * bend_piece(d));
    assert!(close(part, part_want, 1e-9), "part {part} vs {part_want}");
    // The notch on the outline: its inner corner (75 − d, 40 − d) mm is an outline vertex.
    let layout = match out.state.records.get(&fid("flat")) {
        Some(runmat_cad::regen::records::FeatureRecord::FlatPattern { layout, .. }) => layout,
        other => panic!("no flat pattern: {other:?}"),
    };
    let corner = [0.075 - d, 0.04 - d];
    assert!(
        layout.loops[0]
            .iter()
            .any(|s| (s.start[0] - corner[0]).hypot(s.start[1] - corner[1]) < 1e-9),
        "{:?}",
        layout.loops[0]
    );
}

#[test]
fn k1_a_round_corner_relief_rounds_the_plates_piece() {
    let d = 0.002;
    let (part, flat, _) = change(
        flanged(FlangePosition::BendOutside),
        relief(CornerReliefKind::Round { size: mm(2.0) }),
    );
    let flat_want = -T * (PI * d * d / 4.0 + 2.0 * d * ba());
    assert!(close(flat, flat_want, 1e-9), "flat {flat} vs {flat_want}");
    let part_want = -(T * PI * d * d / 4.0 + 2.0 * bend_piece(d));
    assert!(close(part, part_want, 1e-9), "part {part} vs {part_want}");
}

#[test]
fn k1_a_tear_corner_relief_removes_nothing() {
    let (part, flat, _) = change(
        flanged(FlangePosition::BendOutside),
        relief(CornerReliefKind::Tear),
    );
    assert!(part.abs() < 1e-15 && flat.abs() < 1e-15, "{part} {flat}");
}

fn closed(overlap: CornerOverlap) -> impl Fn(&RegenResult) -> Feature {
    move |plain| {
        Feature::new(
            "close",
            FeatureKind::SheetMetal(SheetMetalFeature::ClosedCorner {
                faces: vec![named(plain, "side", Role::SheetFace, 0)],
                gap: mm(0.5),
                overlap,
            }),
        )
    }
}

/// Closed corners: `side`'s 20 mm flat and `back`'s 15 mm flat extend towards each other by
/// the lengths [`CornerOverlap`] gives (R + t to the other's outer face, R − gap short of its
/// inner face). The part and the flat pattern both gain exactly those rectangles.
#[test]
fn k1_closed_corners_extend_both_flats_by_their_overlap() {
    let g = 0.0005;
    for (kind, side, back) in [
        (CornerOverlap::Overlap, R + T, R - g),
        (CornerOverlap::Underlap, R - g, R + T),
        (CornerOverlap::Butt, R - g, R - g),
    ] {
        let (part, flat, _) = change(flanged(FlangePosition::BendOutside), closed(kind));
        let want = T * (0.02 * side + 0.015 * back);
        assert!(close(part, want, 1e-9), "{kind:?}: part {part} vs {want}");
        assert!(close(flat, want, 1e-9), "{kind:?}: flat {flat} vs {want}");
    }
}
