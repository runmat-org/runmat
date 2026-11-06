//! The Features tab with its options, regenerated on the fake kernel: extrude end conditions
//! (second direction, up to next, up to a curved or tilted face, offset from face), draft and
//! thin; holes from the standard tables (tapped, modelled threads, slots, series);
//! variable, face and full-round fillets; chamfer forms and vertex chamfers; multi-thickness
//! shells; parting-line drafts; and curve, sketch, fill and variable patterns. Volumes are
//! checked against closed forms and generated faces are resolved by name.

mod common;
use common::*;
use runmat_cad::evaluate::*;
use runmat_cad::ids::{InputName, SketchEntityId};
use runmat_cad::model::hole_tables::*;
use runmat_cad::model::*;
use runmat_cad::refs::*;
use runmat_cad::regen::records::FeatureRecord;
use runmat_cad::regen::RegenResult;
use runmat_cad::value::{InputDecl, ValueExpr};
use std::collections::BTreeMap;

const MM3: f64 = 1e-9;

fn volume(env: &Env, r: &RegenResult) -> f64 {
    r.state
        .bodies
        .iter()
        .map(|b| {
            let AnalysisResult::MassProperties(m) = env
                .kernel
                .analyze(
                    &AnalysisRequest::MassProperties { density: 1.0 },
                    &[&b.handle],
                )
                .unwrap()
            else {
                panic!()
            };
            m.volume
        })
        .sum()
}

fn near(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1e-300)
}

fn top() -> PlaneRef {
    PlaneRef::Face {
        face: Reference::end_cap("body"),
    }
}

fn square(plane: PlaneRef, x0: f64, y0: f64, x1: f64, y1: f64) -> Sketch {
    rect(plane, mm(x0), mm(y0), mm(x1), mm(y1))
}

fn feature(id: &str, kind: FeatureKind) -> Feature {
    Feature::new(id, kind)
}

fn ext(sketch: &str, extent: Extent, mode: BodyMode) -> Extrude {
    Extrude::new(ProfileRef::sketch(sketch), extent, mode)
}

fn blind(d: f64) -> Extent {
    Extent::Blind { distance: mm(d) }
}

fn probe(d: &mut PartDocument, name: &str, kind: TopoKind, r: Reference) {
    d.regions
        .push(region(name, kind, vec![r], Cardinality::ExactlyOne));
}

fn gen_name(feature: &str, role: Role, tag: SourceTag) -> EntityName {
    EntityName::generated(feature, role, tag)
}

// ================================================================== extrude options

#[test]
fn second_direction_and_offset_from_face() {
    let env = Env::new();
    let mut d = block();
    d.features.push(sketch(
        "s2",
        square(PlaneRef::origin_xy(), 10.0, 10.0, 30.0, 30.0),
    ));
    let mut e = ext("s2", blind(15.0), BodyMode::New);
    e.second = Some(blind(5.0));
    d.features.push(feature("boss", FeatureKind::Extrude(e)));
    probe(
        &mut d,
        "boss_start",
        TopoKind::Face,
        Reference::start_cap("boss"),
    );
    probe(
        &mut d,
        "boss_end",
        TopoKind::Face,
        Reference::end_cap("boss"),
    );
    let r = env.regen(&d);
    assert_clean(&r);
    assert!(near(volume(&env, &r), (40_000.0 + 8_000.0) * MM3, 1e-9));
    assert!(approx(
        region_faces(&r, "boss_start")[0].centroid[2],
        -0.005
    ));
    assert!(approx(region_faces(&r, "boss_end")[0].centroid[2], 0.015));

    // Offset from the block's top face: 2 mm short of it.
    let mut d = block();
    d.features.push(sketch(
        "s2",
        square(PlaneRef::origin_xy(), 90.0, 0.0, 100.0, 10.0),
    ));
    d.features.push(feature(
        "post",
        FeatureKind::Extrude(ext(
            "s2",
            Extent::OffsetFromFace {
                face: Reference::end_cap("body"),
                offset: mm(2.0),
                beyond: false,
            },
            BodyMode::New,
        )),
    ));
    let r = env.regen(&d);
    assert_clean(&r);
    assert!(near(volume(&env, &r), (40_000.0 + 800.0) * MM3, 1e-9));

    // Mid-plane cannot take a second direction.
    let mut d = block();
    d.features.push(sketch(
        "s2",
        square(PlaneRef::origin_xy(), 10.0, 10.0, 30.0, 30.0),
    ));
    let mut e = ext("s2", Extent::Symmetric { distance: mm(4.0) }, BodyMode::New);
    e.second = Some(blind(1.0));
    d.features.push(feature("bad", FeatureKind::Extrude(e)));
    let r = env.regen(&d);
    assert!(has(&r, runmat_cad::diagnostics::DiagCode::RegenInvalid));
}

#[test]
fn cut_up_to_next_goes_through_the_material_it_starts_in() {
    let env = Env::new();
    let mut d = block();
    d.features
        .push(sketch("s2", square(top(), 10.0, 10.0, 30.0, 30.0)));
    let mut e = ext("s2", Extent::UpToNext, BodyMode::cut());
    e.reverse = true;
    d.features.push(feature("pocket", FeatureKind::Extrude(e)));
    let r = env.regen(&d);
    assert_clean(&r);
    assert!(near(volume(&env, &r), (40_000.0 - 4_000.0) * MM3, 1e-9));
    // Nothing above the top: a boss up to next has no face to end at.
    let mut d = block();
    d.features
        .push(sketch("s2", square(top(), 10.0, 10.0, 30.0, 30.0)));
    d.features.push(feature(
        "boss",
        FeatureKind::Extrude(ext("s2", Extent::UpToNext, BodyMode::add())),
    ));
    let r = env.regen(&d);
    assert!(has(&r, runmat_cad::diagnostics::DiagCode::KernelFailed));
}

/// A sloped roof over the block: z from 20 + x/4 (underside) along x, 50 deep in y.
fn roofed() -> PartDocument {
    let mut d = block();
    let zx = PlaneRef::Datum {
        datum: DatumRef::OriginPlane {
            plane: OriginPlane::Zx,
        },
    };
    // Sketch (u, v) = (z, x).
    d.features.push(sketch(
        "s_roof",
        polygon(
            zx,
            &[
                (mm(20.0), mm(0.0)),
                (mm(40.0), mm(80.0)),
                (mm(45.0), mm(80.0)),
                (mm(25.0), mm(0.0)),
            ],
            Some(&["under", "end", "over", "start"]),
        ),
    ));
    d.features.push(feature(
        "roof",
        FeatureKind::Extrude(ext("s_roof", blind(50.0), BodyMode::New)),
    ));
    d
}

#[test]
fn boss_up_to_a_tilted_face_follows_it() {
    let env = Env::new();
    let base = env.regen(&roofed());
    assert_clean(&base);
    let v0 = volume(&env, &base);
    let mut d = roofed();
    d.features
        .push(sketch("s2", square(top(), 20.0, 10.0, 40.0, 30.0)));
    d.features.push(feature(
        "boss",
        FeatureKind::Extrude(ext(
            "s2",
            Extent::UpTo {
                face: Reference::side("roof", "s_roof", "under"),
            },
            BodyMode::Add {
                targets: vec![Reference::feature_body("body")],
            },
        )),
    ));
    probe(
        &mut d,
        "boss_end",
        TopoKind::Face,
        Reference::end_cap("boss"),
    );
    let r = env.regen(&d);
    assert_clean(&r);
    // Heights 20 + x/4 − 10 over x 20..40: mean 17.5 mm over 20 × 20.
    let added = volume(&env, &r) - v0;
    assert!(near(added, 7_000.0 * MM3, 0.06), "added {}", added / MM3);
    assert_eq!(r.state.bodies.len(), 2, "the roof is read, not merged");
    assert!(region_ok(&r, "boss_end"));
}

#[test]
fn drafted_and_thin_extrudes() {
    let env = Env::new();
    let mut d = PartDocument::new("p.part");
    d.features.push(sketch(
        "s",
        square(PlaneRef::origin_xy(), 0.0, 0.0, 40.0, 40.0),
    ));
    let mut e = ext("s", blind(20.0), BodyMode::New);
    e.draft = Some(ExtrudeDraft {
        angle: ValueExpr::deg(5.0),
        outward: false,
        second_angle: None,
        second_outward: false,
    });
    d.features.push(feature("body", FeatureKind::Extrude(e)));
    probe(&mut d, "side", TopoKind::Face, side("body", "s", "l1"));
    let r = env.regen(&d);
    assert_clean(&r);
    let t = 5f64.to_radians().tan();
    let exact: f64 = (0..2000)
        .map(|k| {
            let h = (k as f64 + 0.5) * 0.01;
            (40.0 - 2.0 * h * t).powi(2) * 0.01
        })
        .sum();
    let v = volume(&env, &r);
    assert!(
        near(v, exact * MM3, 0.03),
        "volume {} vs {}",
        v / MM3,
        exact
    );
    let g = region_faces(&r, "side")[0];
    // The bottom side face leans inward: its normal has −y and +z parts.
    assert!(matches!(
        g.surface,
        SurfaceKind::Plane { .. } | SurfaceKind::Other
    ));

    // Thin: a 2 mm wall inside a 40 × 30 profile, 10 high, then closed with 2 mm caps.
    let thin = |cap: Option<f64>| {
        let mut d = PartDocument::new("p.part");
        d.features.push(sketch(
            "s",
            square(PlaneRef::origin_xy(), 0.0, 0.0, 40.0, 30.0),
        ));
        let mut e = ext("s", blind(10.0), BodyMode::New);
        e.thin = Some(ThinWall {
            thickness: mm(2.0),
            side: ThinSide::One,
            second_thickness: None,
            cap_ends: cap.map(mm),
        });
        d.features.push(feature("box", FeatureKind::Extrude(e)));
        probe(
            &mut d,
            "inner",
            TopoKind::Face,
            Reference::gen_face(
                "box",
                Role::ThinWall,
                SourceTag::Curve {
                    sketch: "s".into(),
                    entity: SketchEntityId::new("l1"),
                },
            ),
        );
        d
    };
    let r = env.regen(&thin(None));
    assert_clean(&r);
    assert!(near(volume(&env, &r), 2_640.0 * MM3, 1e-6));
    let inner = region_faces(&r, "inner");
    assert!(approx(inner[0].centroid[1], 0.002));
    assert!(normal(&inner[0])[1] > 0.99);
    let r = env.regen(&thin(Some(2.0)));
    assert_clean(&r);
    assert!(near(
        volume(&env, &r),
        (2_640.0 + 2.0 * 36.0 * 26.0 * 2.0) * MM3,
        1e-6
    ));
}

// ============================================================================== holes

fn hole_at(face: Reference, key: &str, x: f64, y: f64, z: ValueExpr) -> Hole {
    Hole {
        face,
        positions: vec![HolePosition {
            key: key.into(),
            at: PointRef::Coords {
                x: mm(x),
                y: mm(y),
                z,
            },
        }],
        diameter: None,
        end: HoleEnd::ThroughAll,
        form: HoleForm::Simple,
        spec: None,
        drill_point: None,
        series: None,
    }
}

fn m6() -> HoleSpec {
    HoleSpec {
        standard: HoleStandard::Iso,
        size: "M6".into(),
        fit: HoleFit::Normal,
        class: None,
    }
}

#[test]
fn tapped_holes_from_the_table_with_modelled_threads() {
    let env = Env::new();
    let mut d = block();
    let mut h = hole_at(
        Reference::end_cap("body"),
        "a",
        20.0,
        20.0,
        input("thickness"),
    );
    h.spec = Some(m6());
    h.end = HoleEnd::Blind { depth: mm(8.0) };
    h.drill_point = Some(ValueExpr::deg(118.0));
    h.form = HoleForm::Tapped {
        thread_depth: None,
        modelled: true,
        near_countersink: None,
    };
    d.features.push(feature("tap", FeatureKind::Hole(h)));
    let key = || SourceTag::Key { key: "a".into() };
    probe(
        &mut d,
        "thread",
        TopoKind::Face,
        Reference::gen_face("tap", Role::ThreadFace, key()),
    );
    probe(
        &mut d,
        "wall",
        TopoKind::Face,
        Reference::gen_face("tap", Role::HoleWall, key()),
    );
    probe(
        &mut d,
        "tip",
        TopoKind::Face,
        Reference::gen_face("tap", Role::HoleBottom, key()),
    );
    let r = env.regen(&d);
    assert_clean(&r);
    assert!(region_ok(&r, "thread") && region_ok(&r, "wall") && region_ok(&r, "tip"));
    let Some(FeatureRecord::Holes { holes }) = r.state.records.get(&"tap".into()) else {
        panic!("no hole record")
    };
    let h0 = &holes[0];
    assert!(approx(h0.diameter, 0.005));
    let t = h0.thread.as_ref().unwrap();
    assert_eq!(t.designation, "M6x1");
    assert_eq!(t.class.as_deref(), Some("6H"));
    assert!(approx(t.depth.unwrap(), 0.0065));
    // The thread face sits above the plain wall.
    let (th, w) = (region_faces(&r, "thread")[0], region_faces(&r, "wall")[0]);
    assert!(th.centroid[2] > w.centroid[2]);

    // A counterbore from the table: clearance 6.6, counterbore 11 × 6.5.
    let mut d = block();
    let mut h = hole_at(
        Reference::end_cap("body"),
        "c",
        40.0,
        25.0,
        input("thickness"),
    );
    h.spec = Some(m6());
    h.form = HoleForm::Counterbore {
        diameter: None,
        depth: None,
    };
    d.features.push(feature("cb", FeatureKind::Hole(h)));
    probe(
        &mut d,
        "floor",
        TopoKind::Face,
        Reference::gen_face(
            "cb",
            Role::HoleCounterbore,
            SourceTag::Key { key: "c".into() },
        ),
    );
    let r = env.regen(&d);
    assert_clean(&r);
    let Some(FeatureRecord::Holes { holes }) = r.state.records.get(&"cb".into()) else {
        panic!()
    };
    assert!(approx(holes[0].diameter, 0.0066));
    let (cd, ch) = holes[0].counterbore.unwrap();
    assert!(approx(cd, 0.011) && approx(ch, 0.0065));
    assert!(approx(region_faces(&r, "floor")[0].centroid[2], 0.0035));

    // A pipe size on a tapped hole is refused with the reason.
    let mut d = block();
    let mut h = hole_at(
        Reference::end_cap("body"),
        "p",
        40.0,
        25.0,
        input("thickness"),
    );
    h.spec = Some(HoleSpec {
        standard: HoleStandard::Npt,
        size: "1/8".into(),
        fit: HoleFit::Normal,
        class: None,
    });
    h.form = HoleForm::Tapped {
        thread_depth: None,
        modelled: false,
        near_countersink: None,
    };
    d.features.push(feature("p", FeatureKind::Hole(h.clone())));
    let r = env.regen(&d);
    assert!(diag(&r, runmat_cad::diagnostics::DiagCode::ValueFailed)
        .message
        .contains("pipe size"));
    // As a pipe tap it tapers.
    h.form = HoleForm::PipeTap {
        thread_depth: None,
        modelled: false,
    };
    let mut d = block();
    d.features.push(feature("p", FeatureKind::Hole(h)));
    let r = env.regen(&d);
    assert_clean(&r);
    let Some(FeatureRecord::Holes { holes }) = r.state.records.get(&"p".into()) else {
        panic!()
    };
    assert_eq!(holes[0].thread.as_ref().unwrap().designation, "1/8-27 NPT");
}

#[test]
fn slots_and_hole_series() {
    let env = Env::new();
    let mut d = block();
    let mut h = hole_at(
        Reference::end_cap("body"),
        "s",
        40.0,
        25.0,
        input("thickness"),
    );
    h.diameter = Some(mm(6.0));
    h.form = HoleForm::Slot {
        length: mm(20.0),
        angle: None,
        counterbore: None,
    };
    d.features.push(feature("slot", FeatureKind::Hole(h)));
    for k in 0..4u32 {
        d.regions.push(Region {
            name: format!("w{}", k).as_str().into(),
            kind: TopoKind::Face,
            select: vec![Reference::face(EntityName::Generated(GenerationName {
                ordinal: Some(k),
                ..match gen_name("slot", Role::HoleWall, SourceTag::Key { key: "s".into() }) {
                    EntityName::Generated(g) => g,
                    _ => unreachable!(),
                }
            }))],
            expect: Cardinality::ExactlyOne,
        });
    }
    let r = env.regen(&d);
    assert_clean(&r);
    for k in 0..4 {
        assert!(region_ok(&r, &format!("w{}", k)), "slot wall {}", k);
    }
    // 6 wide, 20 between centres, through 10: 6·20·10 + π·9·10.
    let v = 40_000.0 - (1_200.0 + std::f64::consts::PI * 90.0);
    assert!(near(volume(&env, &r), v * MM3, 0.03));

    // A series through two stacked bodies: clearance in the top, tapped in the bottom.
    let mut d = block();
    d.features.push(sketch(
        "s_low",
        square(PlaneRef::origin_xy(), 0.0, 0.0, 80.0, 50.0),
    ));
    d.features.push(feature(
        "low",
        FeatureKind::Extrude({
            let mut e = ext("s_low", blind(10.0), BodyMode::New);
            e.reverse = true;
            e
        }),
    ));
    let mut h = hole_at(
        Reference::end_cap("body"),
        "b",
        40.0,
        25.0,
        input("thickness"),
    );
    h.spec = Some(m6());
    h.series = Some(HoleSeries {
        bodies: Vec::new(),
        end: HoleForm::Tapped {
            thread_depth: None,
            modelled: true,
            near_countersink: None,
        },
        middle_diameter: None,
    });
    d.features.push(feature("bolt", FeatureKind::Hole(h)));
    let r = env.regen(&d);
    assert_clean(&r);
    assert_eq!(r.state.bodies.len(), 2);
    let removed = 80_000.0 * MM3 - volume(&env, &r);
    let expect = std::f64::consts::PI * (3.3f64.powi(2) + 2.5f64.powi(2)) * 10.0 * MM3;
    assert!(
        near(removed, expect, 0.1),
        "removed {} vs {}",
        removed / MM3,
        expect / MM3
    );
}

// ===================================================================== blends, chamfers

#[test]
fn variable_face_and_full_round_fillets() {
    let env = Env::new();
    let edge = || Reference::edges_between(Reference::end_cap("body"), side("body", "s1", "l1"));
    let mut d = block();
    let mut f = Fillet::constant(vec![edge()], mm(2.0));
    f.variable = vec![
        RadiusAt {
            at: ValueExpr::number(0.0),
            radius: mm(2.0),
        },
        RadiusAt {
            at: ValueExpr::number(1.0),
            radius: mm(4.0),
        },
    ];
    d.features.push(feature("vr", FeatureKind::Fillet(f)));
    let tag = SourceTag::edge_between(
        &gen_name("body", Role::EndCap, SourceTag::None),
        &gen_name(
            "body",
            Role::Side,
            SourceTag::Curve {
                sketch: "s1".into(),
                entity: SketchEntityId::new("l1"),
            },
        ),
    );
    probe(
        &mut d,
        "vr",
        TopoKind::Face,
        Reference::gen_face("vr", Role::Blend, tag.clone()),
    );
    let r = env.regen(&d);
    assert_clean(&r);
    assert!(matches!(
        region_faces(&r, "vr")[0].surface,
        SurfaceKind::Other
    ));

    // A face fillet between the top and that side blends their shared edge.
    let mut d = block();
    d.features.push(feature(
        "ff",
        FeatureKind::FaceFillet(FaceFillet {
            first: vec![Reference::end_cap("body")],
            second: vec![side("body", "s1", "l1")],
            radius: mm(3.0),
            profile: FilletProfile::Circular,
        }),
    ));
    probe(
        &mut d,
        "ff",
        TopoKind::Face,
        Reference::gen_face("ff", Role::Blend, tag),
    );
    let r = env.regen(&d);
    assert_clean(&r);
    assert!(region_ok(&r, "ff"));

    // A full round across a 10 mm rib: the top becomes a 5 mm blend.
    let mut d = PartDocument::new("rib.part");
    d.features.push(sketch(
        "s",
        square(PlaneRef::origin_xy(), 0.0, 0.0, 80.0, 10.0),
    ));
    d.features.push(feature(
        "rib",
        FeatureKind::Extrude(ext("s", blind(20.0), BodyMode::New)),
    ));
    d.features.push(feature(
        "fr",
        FeatureKind::FullRound(FullRound {
            first: vec![side("rib", "s", "l1")],
            center: vec![Reference::end_cap("rib")],
            second: vec![side("rib", "s", "l3")],
        }),
    ));
    let top_name = gen_name("rib", Role::EndCap, SourceTag::None);
    probe(
        &mut d,
        "round",
        TopoKind::Face,
        Reference::gen_face("fr", Role::Blend, SourceTag::face_of(&top_name)),
    );
    let r = env.regen(&d);
    assert_clean(&r);
    let g = region_faces(&r, "round")[0];
    assert!(matches!(g.surface, SurfaceKind::Cylinder { radius, .. } if approx(radius, 0.005)));
}

#[test]
fn chamfer_forms_and_vertex_chamfers() {
    let env = Env::new();
    let mut d = block();
    d.features.push(feature(
        "da",
        FeatureKind::Chamfer(Chamfer {
            edges: vec![Reference::edges_between(
                Reference::end_cap("body"),
                side("body", "s1", "l1"),
            )],
            form: ChamferForm::DistanceAngle {
                distance: mm(2.0),
                angle: ValueExpr::deg(30.0),
            },
            flip: true,
        }),
    ));
    let r = env.regen(&d);
    assert_clean(&r);

    // A corner cut 6 mm along each edge removes a 6·6·6/6 tetrahedron.
    let mut d = block();
    let corner = Reference::vertex_of(vec![
        Reference::end_cap("body"),
        side("body", "s1", "l1"),
        side("body", "s1", "l2"),
    ]);
    d.features.push(feature(
        "vc",
        FeatureKind::Chamfer(Chamfer {
            edges: vec![corner],
            form: ChamferForm::Vertex {
                distances: [mm(6.0), mm(6.0), mm(6.0)],
            },
            flip: false,
        }),
    ));
    let faces = [
        gen_name("body", Role::EndCap, SourceTag::None),
        gen_name(
            "body",
            Role::Side,
            SourceTag::Curve {
                sketch: "s1".into(),
                entity: SketchEntityId::new("l1"),
            },
        ),
        gen_name(
            "body",
            Role::Side,
            SourceTag::Curve {
                sketch: "s1".into(),
                entity: SketchEntityId::new("l2"),
            },
        ),
    ];
    probe(
        &mut d,
        "cut",
        TopoKind::Face,
        Reference::gen_face("vc", Role::Chamfer, SourceTag::vertex_of(&faces)),
    );
    let r = env.regen(&d);
    assert_clean(&r);
    let removed = 40_000.0 * MM3 - volume(&env, &r);
    assert!(near(removed, 36.0 * MM3, 0.25), "removed {}", removed / MM3);
    assert!(region_ok(&r, "cut"));
}

// ===================================================================== shells, drafts

#[test]
fn multi_thickness_shell() {
    let env = Env::new();
    let mut d = block();
    let mut s = Shell::new(
        Reference::feature_body("body"),
        vec![Reference::end_cap("body")],
        mm(2.0),
    );
    s.faces.push(FaceThickness {
        faces: vec![side("body", "s1", "l1")],
        thickness: mm(5.0),
    });
    d.features.push(feature("sh", FeatureKind::Shell(s)));
    let r = env.regen(&d);
    assert_clean(&r);
    assert!(near(
        volume(&env, &r),
        (40_000.0 - 76.0 * 43.0 * 8.0) * MM3,
        1e-9
    ));
}

#[test]
fn outward_shells_make_the_body_the_cavity() {
    let env = Env::new();
    let shell = |open: Vec<Reference>| {
        let mut d = block();
        let mut s = Shell::new(Reference::feature_body("body"), open, mm(2.0));
        s.outward = true;
        d.features.push(feature("sh", FeatureKind::Shell(s)));
        d
    };
    // Open at the top: walls 2 mm outside the 80 × 50 × 10 block, flush with the opening.
    let mut d = shell(vec![Reference::end_cap("body")]);
    let l1 = gen_name(
        "body",
        Role::Side,
        SourceTag::Curve {
            sketch: "s1".into(),
            entity: SketchEntityId::new("l1"),
        },
    );
    probe(
        &mut d,
        "outer",
        TopoKind::Face,
        Reference::gen_face("sh", Role::ShellOuter, SourceTag::face_of(&l1)),
    );
    let top = gen_name("body", Role::EndCap, SourceTag::None);
    probe(
        &mut d,
        "rim",
        TopoKind::Face,
        Reference::gen_face("sh", Role::ShellRim, SourceTag::face_of(&top)),
    );
    probe(
        &mut d,
        "floor",
        TopoKind::Face,
        Reference::start_cap("body"),
    );
    let r = env.regen(&d);
    assert_clean(&r);
    let v = volume(&env, &r);
    assert!(
        near(v, (84.0 * 54.0 * 12.0 - 40_000.0) * MM3, 1e-9),
        "volume {}",
        v / MM3
    );
    let outer = region_faces(&r, "outer")[0];
    assert!(approx(outer.centroid[1], -0.002));
    assert!(normal(&outer)[1] < -0.99);
    let rim = region_faces(&r, "rim")[0];
    assert!(approx(rim.centroid[2], 0.01) && normal(&rim)[2] > 0.99);
    // The old bottom face is now the cavity's floor, facing up into it.
    let floor = region_faces(&r, "floor")[0];
    assert!(approx(floor.centroid[2], 0.0) && normal(&floor)[2] > 0.99);

    // Closed: the block becomes a sealed void.
    let r = env.regen(&shell(Vec::new()));
    assert_clean(&r);
    assert!(near(
        volume(&env, &r),
        (84.0 * 54.0 * 14.0 - 40_000.0) * MM3,
        1e-9
    ));
}

fn tall() -> PartDocument {
    let mut d = PartDocument::new("p.part");
    d.features.push(sketch(
        "s",
        square(PlaneRef::origin_xy(), 0.0, 0.0, 40.0, 40.0),
    ));
    d.features.push(feature(
        "body",
        FeatureKind::Extrude(ext("s", blind(20.0), BodyMode::New)),
    ));
    d
}

fn sides() -> Vec<Reference> {
    ["l1", "l2", "l3", "l4"]
        .iter()
        .map(|c| side("body", "s", c))
        .collect()
}

#[test]
fn parting_line_drafts_hinge_at_their_edges() {
    let env = Env::new();
    let draft = |kind: DraftKind| {
        let mut d = tall();
        d.features.push(feature(
            "dr",
            FeatureKind::Draft(Draft {
                faces: sides(),
                neutral: PlaneRef::origin_xy(),
                angle: ValueExpr::deg(4.0),
                kind,
                flip: false,
            }),
        ));
        env.regen(&d)
    };
    let neutral = draft(DraftKind::NeutralPlane);
    assert_clean(&neutral);
    let edges = |cap: Reference| -> Vec<Reference> {
        sides()
            .into_iter()
            .map(|s| Reference::edges_between(cap.clone(), s))
            .collect()
    };
    let bottom = draft(DraftKind::PartingLine {
        edges: edges(Reference::start_cap("body")),
    });
    assert_clean(&bottom);
    assert!(near(volume(&env, &bottom), volume(&env, &neutral), 1e-9));
    let top = draft(DraftKind::PartingLine {
        edges: edges(Reference::end_cap("body")),
    });
    assert_clean(&top);
    // Hinged at the top, the faces lean out going down: more material than the block.
    assert!(volume(&env, &top) > 32_000.0 * MM3);
    assert!(volume(&env, &neutral) < 32_000.0 * MM3);
}

/// Core and cavity: a block 5 mm round the 40 x 40 x 20 mm part, less the part, split at the
/// parting plane 10 mm up. The core (below) and the cavity (above) each hold 50 x 50 x 15 mm of
/// block less 40 x 40 x 10 mm of part, 21,500 mm^3, and the part stays whole beside them.
#[test]
fn core_and_cavity_split_a_block_round_the_part_at_the_parting_plane() {
    let env = Env::new();
    let mut d = tall();
    d.features.push(feature(
        "parting",
        FeatureKind::Plane(PlaneDef::Offset { base: PlaneRef::origin_xy(), distance: mm(10.0) }),
    ));
    d.features.push(feature(
        "mold",
        FeatureKind::Mold(runmat_cad::model::mold::MoldFeature::CoreCavity {
            part: Reference::feature_body("body"),
            parting: PlaneRef::datum("parting"),
            margin: mm(5.0),
        }),
    ));
    let r = env.regen(&d);
    assert_clean(&r);
    assert_eq!(r.state.bodies.len(), 3, "core, cavity and the part");
    let each = |b: &runmat_cad::regen::LiveBody| {
        let AnalysisResult::MassProperties(m) =
            env.kernel.analyze(&AnalysisRequest::MassProperties { density: 1.0 }, &[&b.handle]).unwrap()
        else {
            panic!()
        };
        m.volume / MM3
    };
    let mut v: Vec<f64> = r.state.bodies.iter().map(each).collect();
    v.sort_by(f64::total_cmp);
    assert!(near(v[0], 21_500.0, 1e-6) && near(v[1], 21_500.0, 1e-6), "{v:?}");
    assert!(near(v[2], 32_000.0, 1e-6), "{v:?}");
    // A parting plane clear of the part has nothing to part.
    let mut off = tall();
    off.features.push(feature(
        "mold",
        FeatureKind::Mold(runmat_cad::model::mold::MoldFeature::CoreCavity {
            part: Reference::feature_body("body"),
            parting: PlaneRef::origin_xy(),
            margin: mm(5.0),
        }),
    ));
    assert!(has(&env.regen(&off), runmat_cad::diagnostics::DiagCode::RegenInvalid));
}

// =========================================================================== patterns

fn pocket_doc(depth: ValueExpr) -> PartDocument {
    let mut d = block();
    d.inputs.push(InputDecl::new("pd", "length", depth));
    d.features
        .push(sketch("sp", square(top(), 4.0, 4.0, 8.0, 8.0)));
    let mut e = ext(
        "sp",
        Extent::Blind {
            distance: input("pd"),
        },
        BodyMode::cut(),
    );
    e.reverse = true;
    d.features.push(feature("pocket", FeatureKind::Extrude(e)));
    d
}

fn pattern(layout: PatternLayout) -> Feature {
    feature(
        "pat",
        FeatureKind::Pattern(Pattern {
            seeds: PatternSeeds::Features {
                features: vec!["pocket".into()],
            },
            layout,
            skip: Vec::new(),
        }),
    )
}

fn pockets(env: &Env, r: &RegenResult) -> f64 {
    (40_000.0 * MM3 - volume(env, r)) / (16.0 * 2.0 * MM3)
}

#[test]
fn curve_sketch_fill_and_variable_patterns() {
    let env = Env::new();
    // Curve-driven along the front top edge (80 long): 4 instances equally spaced.
    let mut d = pocket_doc(mm(2.0));
    d.features.push(pattern(PatternLayout::Curve {
        curve: CurveRef::Edge {
            edge: Reference::edges_between(Reference::end_cap("body"), side("body", "s1", "l1")),
        },
        count: ValueExpr::number(3.0),
        spacing: Some(mm(30.0)),
        orientation: CurveOrientation::Translate,
    }));
    let r = env.regen(&d);
    assert_clean(&r);
    assert!(near(pockets(&env, &r), 3.0, 1e-6));

    // Sketch-driven: three points, the seed at the first.
    let mut d = pocket_doc(mm(2.0));
    let mut pts = square(top(), 0.0, 0.0, 1.0, 1.0);
    pts.entities = [("p1", 6.0, 6.0), ("p2", 36.0, 6.0), ("p3", 60.0, 30.0)]
        .iter()
        .map(|(n, x, y)| SketchEntity {
            id: SketchEntityId::new(*n),
            geometry: SketchGeometry::Point {
                at: PointExpr::Coords {
                    x: mm(*x),
                    y: mm(*y),
                },
            },
            construction: false,
        })
        .collect();
    d.features.push(sketch("pts", pts));
    d.features.push(pattern(PatternLayout::Sketch {
        sketch: "pts".into(),
        points: Vec::new(),
        reference: None,
    }));
    let r = env.regen(&d);
    assert_clean(&r);
    assert!(near(pockets(&env, &r), 3.0, 1e-6));

    // Fill a 50 × 30 region at 10 mm pitch around the seed, 1 mm in from the boundary.
    let mut d = pocket_doc(mm(2.0));
    d.features
        .push(sketch("fill", square(top(), 1.0, 1.0, 51.0, 31.0)));
    d.features.push(pattern(PatternLayout::Fill {
        boundary: ProfileRef::sketch("fill"),
        spacing: mm(10.0),
        lattice: FillLattice::Square,
        direction: None,
        margin: Some(mm(1.0)),
        seed: Some(PointRef::Coords {
            x: mm(6.0),
            y: mm(6.0),
            z: input("thickness"),
        }),
    }));
    let r = env.regen(&d);
    assert_clean(&r);
    // Lattice x 6, 16 … 46 (5) by y 6, 16, 26 (3).
    assert!(near(pockets(&env, &r), 15.0, 1e-6));

    // Variable: the same pocket deeper at each instance (2, 4, 6 mm).
    let mut d = pocket_doc(mm(2.0));
    let inst = |key: &str, dx: f64, depth: f64| VariableInstance {
        key: key.into(),
        dx: mm(dx),
        dy: mm(0.0),
        dz: mm(0.0),
        values: BTreeMap::from([(InputName::new("pd"), mm(depth))]),
    };
    d.features.push(pattern(PatternLayout::Variable {
        instances: vec![inst("b", 20.0, 4.0), inst("c", 40.0, 6.0)],
    }));
    let r = env.regen(&d);
    assert_clean(&r);
    assert!(near(pockets(&env, &r), 6.0, 1e-6));
}
