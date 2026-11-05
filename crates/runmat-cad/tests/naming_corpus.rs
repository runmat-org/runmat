//! The naming corpus: each scenario edits a design (parameters that change topology counts,
//! splits, merges, suppress/unsuppress, reorders, pattern counts) and checks that every
//! reference resolves to the same logical entity or yields the specific expected diagnostic.
//! No scenario may change a target silently, and none may resolve by geometry unless it asks.

mod common;
use common::*;
use runmat_cad::diagnostics::{DiagCode, Repair};
use runmat_cad::ids::FeatureId;
use runmat_cad::model::*;
use runmat_cad::refs::*;
use runmat_cad::regen::FeatureStatus;
use runmat_cad::value::ValueExpr;

fn top() -> Reference {
    Reference::end_cap("body")
}

fn front() -> Reference {
    side("body", "s1", "l1")
}

fn right() -> Reference {
    side("body", "s1", "l2")
}

/// The block with regions on its top, front and the front-top edge.
fn base() -> PartDocument {
    let mut d = block();
    d.regions.push(region(
        "top",
        TopoKind::Face,
        vec![top()],
        Cardinality::ExactlyOne,
    ));
    d.regions.push(region(
        "front",
        TopoKind::Face,
        vec![front()],
        Cardinality::ExactlyOne,
    ));
    d.regions.push(region(
        "right",
        TopoKind::Face,
        vec![right()],
        Cardinality::ExactlyOne,
    ));
    d.regions.push(region(
        "front_top",
        TopoKind::Edge,
        vec![Reference::edges_between(front(), top())],
        Cardinality::AtLeastOne,
    ));
    d
}

fn assert_base_regions(r: &runmat_cad::regen::RegenResult, width: f64, thickness: f64) {
    let t = region_faces(r, "top");
    assert_eq!(t.len(), 1);
    assert_eq!(normal(&t[0]), [0.0, 0.0, 1.0]);
    assert!(approx(t[0].centroid[2], thickness));
    let f = region_faces(r, "front");
    assert_eq!(f.len(), 1);
    assert_eq!(normal(&f[0]), [0.0, -1.0, 0.0]);
    let rt = region_faces(r, "right");
    assert_eq!(normal(&rt[0]), [1.0, 0.0, 0.0]);
    assert!(approx(rt[0].centroid[0], width));
    assert!(region_ok(r, "front_top"));
    assert_eq!(region_count(r, "front_top"), 1);
    assert_eq!(r.stats.geometric, 0);
}

/// A slot cut across the full depth, from the top, `x0..x1`, blind `depth` mm.
fn slot(d: &mut PartDocument, x0: f64, x1: f64, depth: f64, before: Option<&str>) {
    let s = rect_names(
        PlaneRef::datum("top_plane"),
        x0,
        -10.0,
        x1,
        60.0,
        ["k1", "k2", "k3", "k4"],
    );
    let plane = Feature::new(
        "top_plane",
        FeatureKind::Plane(PlaneDef::Offset {
            base: PlaneRef::origin_xy(),
            distance: input("thickness"),
        }),
    );
    let sk = sketch("s_slot", s);
    let cut = extrude_reverse("slot", "s_slot", mm(depth), BodyMode::cut());
    match before {
        None => d.features.extend([plane, sk, cut]),
        Some(b) => {
            let i = d.feature_index(&FeatureId::new(b)).expect("feature");
            d.features.splice(i..i, [plane, sk, cut]);
        }
    }
}

// ------------------------------------------------------------------------- parameter edits

#[test]
fn c01_width_edit_keeps_every_reference() {
    let env = Env::new();
    let mut d = base();
    let r1 = env.regen(&d);
    assert_clean(&r1);
    assert_base_regions(&r1, 0.08, 0.01);
    set_input(&mut d, "width", mm(120.0));
    let r2 = env.regen(&d);
    assert_clean(&r2);
    assert_base_regions(&r2, 0.12, 0.01);
}

#[test]
fn c02_thickness_edit_moves_top() {
    let env = Env::new();
    let mut d = base();
    set_input(&mut d, "thickness", mm(5.0));
    let r = env.regen(&d);
    assert_clean(&r);
    assert_base_regions(&r, 0.08, 0.005);
}

#[test]
fn c03_more_profile_sides_keeps_existing_names() {
    // The rectangle becomes an L: l1..l4 keep their roles, l5 and l6 are new sides.
    let env = Env::new();
    let mut d = base();
    let r1 = env.regen(&d);
    assert_eq!(total_faces(&r1), 6);
    let l = polygon(
        PlaneRef::origin_xy(),
        &[
            (mm(0.0), mm(0.0)),
            (input("width"), mm(0.0)),
            (input("width"), mm(20.0)),
            (mm(30.0), mm(20.0)),
            (mm(30.0), input("depth")),
            (mm(0.0), input("depth")),
        ],
        Some(&["l1", "l2", "l5", "l6", "l3", "l4"]),
    );
    d.features[0] = sketch("s1", l);
    let r2 = env.regen(&d);
    assert_clean(&r2);
    assert_eq!(total_faces(&r2), 8);
    assert_eq!(normal(&region_faces(&r2, "front")[0]), [0.0, -1.0, 0.0]);
    assert_eq!(normal(&region_faces(&r2, "right")[0]), [1.0, 0.0, 0.0]);
    assert!(approx(region_faces(&r2, "right")[0].area, 0.02 * 0.01));
}

#[test]
fn c04_removed_curve_reports_not_generated_with_candidates() {
    let env = Env::new();
    let mut d = base();
    d.regions.push(region(
        "back",
        TopoKind::Face,
        vec![side("body", "s1", "l3")],
        Cardinality::ExactlyOne,
    ));
    assert!(region_ok(&env.regen(&d), "back"));
    // l3 is replaced by two new lines: the back face is now two new sides.
    let notch = polygon(
        PlaneRef::origin_xy(),
        &[
            (mm(0.0), mm(0.0)),
            (input("width"), mm(0.0)),
            (input("width"), input("depth")),
            (mm(40.0), input("depth")),
            (mm(40.0), mm(40.0)),
            (mm(0.0), mm(40.0)),
        ],
        Some(&["l1", "l2", "l7", "l8", "l9", "l4"]),
    );
    d.features[0] = sketch("s1", notch);
    let r = env.regen(&d);
    assert!(!region_ok(&r, "back"));
    let dg = diag(&r, DiagCode::RefNotGenerated);
    match &dg.repairs[0] {
        Repair::RePick { candidates, .. } => {
            assert!(!candidates.is_empty(), "same-feature candidates offered")
        }
        other => panic!("unexpected repair {:?}", other),
    }
    // Never silently resolved by geometry.
    assert_eq!(r.stats.geometric, 0);
    assert!(region_ok(&r, "front") && region_ok(&r, "top"));
}

// ------------------------------------------------------------------------- splits

#[test]
fn c05_slot_splits_top_into_pieces_reported_as_ambiguity() {
    let env = Env::new();
    let mut d = base();
    d.regions.push(region(
        "top_all",
        TopoKind::Face,
        vec![top()],
        Cardinality::AtLeastOne,
    ));
    slot(&mut d, 35.0, 45.0, 4.0, None);
    let r = env.regen(&d);
    assert!(
        errors(&r)
            .iter()
            .all(|e| e.code == DiagCode::RegionExpectation),
        "{:#?}",
        errors(&r)
    );
    assert!(!region_ok(&r, "top"));
    let dg = diag(&r, DiagCode::RegionExpectation);
    match &dg.repairs[0] {
        Repair::Disambiguate { candidates, .. } => {
            assert_eq!(candidates.len(), 2, "each piece gets a qualified reference")
        }
        other => panic!("{:?}", other),
    }
    assert!(region_ok(&r, "top_all"));
    assert_eq!(region_count(&r, "top_all"), 2);
    // The front-top edge is now two runs.
    assert_eq!(region_count(&r, "front_top"), 2);
}

#[test]
fn c06_qualified_piece_survives_slot_move() {
    let env = Env::new();
    let mut d = base();
    slot(&mut d, 35.0, 45.0, 4.0, None);
    // The left piece: bounded by the left side l4.
    let left = Reference::piece(top(), vec![side("body", "s1", "l4")]);
    d.regions.push(region(
        "top_left",
        TopoKind::Face,
        vec![left],
        Cardinality::ExactlyOne,
    ));
    let r1 = env.regen(&d);
    assert!(region_ok(&r1, "top_left"));
    assert!(region_faces(&r1, "top_left")[0].centroid[0] < 0.035);
    // Move the slot.
    let i = d.feature_index(&fid("s_slot")).unwrap();
    d.features[i] = sketch(
        "s_slot",
        rect_names(
            PlaneRef::datum("top_plane"),
            50.0,
            -10.0,
            60.0,
            60.0,
            ["k1", "k2", "k3", "k4"],
        ),
    );
    let r2 = env.regen(&d);
    assert!(region_ok(&r2, "top_left"));
    let g = region_faces(&r2, "top_left")[0];
    assert!(g.bbox_max[0] <= 0.05 + 1e-9 && g.bbox_min[0] < 1e-9);
}

#[test]
fn c07_fillet_on_edge_interrupted_upstream_rounds_both_runs() {
    let env = Env::new();
    let mut d = base();
    d.features.push(fillet(
        "round",
        vec![Reference::edges_between(front(), top())],
        2.0,
    ));
    d.regions.retain(|r| r.name.0 != "front_top");
    let r1 = env.regen(&d);
    assert_clean(&r1);
    let blends1: usize = r1.state.bodies[0]
        .names
        .faces
        .iter()
        .filter(|s| s.iter().any(|n| n.maker().0 == "round"))
        .count();
    assert_eq!(blends1, 1);
    slot(&mut d, 35.0, 45.0, 4.0, Some("round"));
    let r2 = env.regen(&d);
    assert!(
        r2.status("round").unwrap().ran(),
        "{:?}",
        r2.status("round")
    );
    let blends2: usize = r2.state.bodies[0]
        .names
        .faces
        .iter()
        .filter(|s| s.iter().any(|n| n.maker().0 == "round"))
        .count();
    assert_eq!(blends2, 2, "both remaining runs are rounded");
}

#[test]
fn c08_slot_inserted_after_fillet_splits_blend_into_pieces() {
    let env = Env::new();
    let mut d = base();
    d.features.push(fillet(
        "round",
        vec![Reference::edges_between(front(), top())],
        2.0,
    ));
    slot(&mut d, 35.0, 45.0, 4.0, None);
    let blend = Reference::face(EntityName::Generated(GenerationName {
        feature: fid("round"),
        role: Role::Blend,
        tag: SourceTag::edge_between(&name_of(&front()), &name_of(&top())),
        ordinal: None,
        step: None,
    }));
    d.regions.push(region(
        "blend",
        TopoKind::Face,
        vec![blend],
        Cardinality::AtLeastOne,
    ));
    let r = env.regen(&d);
    assert!(region_ok(&r, "blend"));
    assert_eq!(region_count(&r, "blend"), 2);
}

fn name_of(r: &Reference) -> EntityName {
    match r {
        Reference::Name { name, .. } => name.clone(),
        _ => panic!("not a name"),
    }
}

#[test]
fn c09_split_body_makes_body_reference_ambiguous_and_sides_exact() {
    let env = Env::new();
    let mut d = base();
    d.features.push(Feature::new(
        "cutplane",
        FeatureKind::Plane(PlaneDef::Offset {
            base: PlaneRef::Datum {
                datum: DatumRef::OriginPlane {
                    plane: OriginPlane::Yz,
                },
            },
            distance: mm(40.0),
        }),
    ));
    d.features.push(Feature::new(
        "halve",
        FeatureKind::Split(Split {
            body: Reference::feature_body("body"),
            tool: PlaneRef::datum("cutplane"),
        }),
    ));
    let pos = Reference::body(EntityName::generated(
        "halve",
        Role::Body,
        SourceTag::Side {
            side: PlaneSide::Positive,
        },
    ));
    d.regions.push(region(
        "pos",
        TopoKind::Body,
        vec![pos],
        Cardinality::ExactlyOne,
    ));
    let r = env.regen(&d);
    assert_eq!(r.state.bodies.len(), 2);
    // The part output names the original body: both halves tie, so it is ambiguous, not guessed.
    let dg = diag(&r, DiagCode::RefAmbiguous);
    assert!(matches!(
        dg.site.subject,
        runmat_cad::diagnostics::Subject::Part { .. }
    ));
    assert!(region_ok(&r, "pos"));
    // The top is split into one piece per body.
    assert!(!region_ok(&r, "top"));
    assert!(has(&r, DiagCode::RegionExpectation));
}

#[test]
fn c10_split_plane_moves_side_names_stable() {
    let env = Env::new();
    let mut d = base();
    d.features.push(Feature::new(
        "cutplane",
        FeatureKind::Plane(PlaneDef::Offset {
            base: PlaneRef::Datum {
                datum: DatumRef::OriginPlane {
                    plane: OriginPlane::Yz,
                },
            },
            distance: mm(40.0),
        }),
    ));
    d.features.push(Feature::new(
        "halve",
        FeatureKind::Split(Split {
            body: Reference::feature_body("body"),
            tool: PlaneRef::datum("cutplane"),
        }),
    ));
    let face_pos = Reference::gen_face(
        "halve",
        Role::SplitFace,
        SourceTag::Side {
            side: PlaneSide::Positive,
        },
    );
    d.regions.push(region(
        "cut_face_pos",
        TopoKind::Face,
        vec![face_pos],
        Cardinality::ExactlyOne,
    ));
    let r1 = env.regen(&d);
    let g1 = region_faces(&r1, "cut_face_pos")[0];
    assert_eq!(
        normal(&g1),
        [-1.0, 0.0, 0.0],
        "the positive body's split face faces back towards the plane"
    );
    let i = d.feature_index(&fid("cutplane")).unwrap();
    d.features[i] = Feature::new(
        "cutplane",
        FeatureKind::Plane(PlaneDef::Offset {
            base: PlaneRef::Datum {
                datum: DatumRef::OriginPlane {
                    plane: OriginPlane::Yz,
                },
            },
            distance: mm(20.0),
        }),
    );
    let r2 = env.regen(&d);
    let g2 = region_faces(&r2, "cut_face_pos")[0];
    assert_eq!(normal(&g2), [-1.0, 0.0, 0.0]);
    assert!(approx(g2.centroid[0], 0.02));
}

#[test]
fn c11_delete_one_half_makes_top_unique_again() {
    let env = Env::new();
    let mut d = base();
    d.features.push(Feature::new(
        "cutplane",
        FeatureKind::Plane(PlaneDef::Offset {
            base: PlaneRef::Datum {
                datum: DatumRef::OriginPlane {
                    plane: OriginPlane::Yz,
                },
            },
            distance: mm(40.0),
        }),
    ));
    d.features.push(Feature::new(
        "halve",
        FeatureKind::Split(Split {
            body: Reference::feature_body("body"),
            tool: PlaneRef::datum("cutplane"),
        }),
    ));
    d.features.push(Feature::new(
        "drop",
        FeatureKind::BodyOp(BodyOp::Delete {
            bodies: vec![Reference::body(EntityName::generated(
                "halve",
                Role::Body,
                SourceTag::Side {
                    side: PlaneSide::Negative,
                },
            ))],
        }),
    ));
    let r = env.regen(&d);
    assert_eq!(r.state.bodies.len(), 1);
    assert!(region_ok(&r, "top"));
    assert!(region_faces(&r, "top")[0].centroid[0] > 0.04);
    // The front face lost its negative piece only.
    assert!(region_ok(&r, "front"));
    // The left side was entirely on the deleted half.
    let mut d2 = d.clone();
    d2.regions.push(region(
        "left",
        TopoKind::Face,
        vec![side("body", "s1", "l4")],
        Cardinality::ExactlyOne,
    ));
    let r2 = env.regen(&d2);
    let dg = diag(&r2, DiagCode::RefConsumed);
    assert!(dg.message.contains("drop"));
}

// ------------------------------------------------------------------------- merges

fn boss(d: &mut PartDocument, x0: f64, x1: f64, height: f64) {
    d.features.push(sketch(
        "s_boss",
        rect_names(
            PlaneRef::origin_xy(),
            x0,
            0.0,
            x1,
            50.0,
            ["m1", "m2", "m3", "m4"],
        ),
    ));
    d.features
        .push(extrude("boss", "s_boss", mm(height), BodyMode::add()));
}

#[test]
fn c12_flush_boss_merges_faces_and_both_names_resolve() {
    let env = Env::new();
    let mut d = base();
    boss(&mut d, 60.0, 80.0, 30.0);
    d.regions.push(region(
        "boss_right",
        TopoKind::Face,
        vec![side("boss", "s_boss", "m2")],
        Cardinality::ExactlyOne,
    ));
    let r = env.regen(&d);
    assert!(region_ok(&r, "right") && region_ok(&r, "boss_right"));
    let a = region_faces(&r, "right")[0];
    let b = region_faces(&r, "boss_right")[0];
    assert_eq!(a, b, "one merged face answers to both names");
    assert!(approx(a.area, 0.05 * 0.03));
    let rec = r.regions.get(&"right".into()).unwrap();
    match &rec.references[0].outcome {
        ResolveOutcome::Resolved(res) => assert_eq!(res.class, ResolutionClass::Merged),
        o => panic!("{:?}", o),
    }
}

#[test]
fn c13_boss_unflush_unmerges_and_keeps_targets() {
    let env = Env::new();
    let mut d = base();
    boss(&mut d, 50.0, 70.0, 30.0);
    d.regions.push(region(
        "boss_right",
        TopoKind::Face,
        vec![side("boss", "s_boss", "m2")],
        Cardinality::ExactlyOne,
    ));
    let r = env.regen(&d);
    assert!(region_ok(&r, "right") && region_ok(&r, "boss_right"));
    let a = region_faces(&r, "right")[0];
    let b = region_faces(&r, "boss_right")[0];
    assert!(approx(a.centroid[0], 0.08) && approx(b.centroid[0], 0.07));
    // The full-depth boss splits the block's top into two pieces: reported, not guessed.
    assert!(!region_ok(&r, "top"));
    assert!(has(&r, DiagCode::RegionExpectation));
}

#[test]
fn c14_boss_start_cap_was_never_made_visible() {
    let env = Env::new();
    let mut d = base();
    // The boss starts on the top plane and grows up, so its start cap lies inside the union.
    d.features.push(Feature::new(
        "top_plane",
        FeatureKind::Plane(PlaneDef::Offset {
            base: PlaneRef::origin_xy(),
            distance: input("thickness"),
        }),
    ));
    d.features.push(sketch(
        "s_boss",
        rect_names(
            PlaneRef::datum("top_plane"),
            10.0,
            10.0,
            30.0,
            30.0,
            ["m1", "m2", "m3", "m4"],
        ),
    ));
    d.features
        .push(extrude("boss", "s_boss", mm(10.0), BodyMode::add()));
    d.regions.push(region(
        "boss_bottom",
        TopoKind::Face,
        vec![Reference::start_cap("boss")],
        Cardinality::ExactlyOne,
    ));
    d.regions.push(region(
        "boss_top",
        TopoKind::Face,
        vec![Reference::end_cap("boss")],
        Cardinality::ExactlyOne,
    ));
    let r = env.regen(&d);
    assert!(region_ok(&r, "boss_top"));
    assert!(!region_ok(&r, "boss_bottom"));
    assert!(has(&r, DiagCode::RefNotGenerated));
    // Thickness edit: the boss follows the top plane.
    set_input(&mut d, "thickness", mm(4.0));
    let r2 = env.regen(&d);
    assert!(approx(region_faces(&r2, "boss_top")[0].centroid[2], 0.014));
}

#[test]
fn c15_boolean_union_merges_body_names() {
    let env = Env::new();
    let mut d = base();
    d.features.push(sketch(
        "s2",
        rect_names(
            PlaneRef::origin_xy(),
            80.0,
            0.0,
            100.0,
            50.0,
            ["n1", "n2", "n3", "n4"],
        ),
    ));
    d.features
        .push(extrude("ear", "s2", mm(10.0), BodyMode::New));
    let r0 = env.regen(&d);
    assert_eq!(
        r0.state.bodies.len(),
        2,
        "a new-body extrude flush against the block is its own body until merged"
    );
    d.features.pop();
    d.features
        .push(extrude("ear", "s2", mm(10.0), BodyMode::New));
    d.features.push(Feature::new(
        "join",
        FeatureKind::Boolean(Boolean {
            kind: BooleanKind::Union,
            target: Reference::feature_body("body"),
            tools: vec![Reference::feature_body("ear")],
            keep_tools: false,
        }),
    ));
    d.regions.push(region(
        "ear_body",
        TopoKind::Body,
        vec![Reference::feature_body("ear")],
        Cardinality::ExactlyOne,
    ));
    let r = env.regen(&d);
    assert_eq!(r.state.bodies.len(), 1);
    assert!(region_ok(&r, "ear_body"));
    assert!(r.parts.values().all(|p| p.ok));
    // The flush tops merged.
    assert!(region_ok(&r, "top"));
    assert!(approx(region_faces(&r, "top")[0].area, 0.1 * 0.05));
}

#[test]
fn c16_boolean_subtract_consumes_tool_body() {
    let env = Env::new();
    let mut d = base();
    d.features.push(sketch(
        "s2",
        rect_names(
            PlaneRef::origin_xy(),
            20.0,
            20.0,
            40.0,
            30.0,
            ["n1", "n2", "n3", "n4"],
        ),
    ));
    d.features
        .push(extrude("pocket_tool", "s2", mm(30.0), BodyMode::New));
    d.features.push(Feature::new(
        "pocket",
        FeatureKind::Boolean(Boolean {
            kind: BooleanKind::Subtract,
            target: Reference::feature_body("body"),
            tools: vec![Reference::feature_body("pocket_tool")],
            keep_tools: false,
        }),
    ));
    d.regions.push(region(
        "tool",
        TopoKind::Body,
        vec![Reference::feature_body("pocket_tool")],
        Cardinality::ExactlyOne,
    ));
    d.regions.push(region(
        "wall",
        TopoKind::Face,
        vec![side("pocket_tool", "s2", "n1")],
        Cardinality::ExactlyOne,
    ));
    let r = env.regen(&d);
    assert_eq!(r.state.bodies.len(), 1);
    let dg = diag(&r, DiagCode::RefConsumed);
    assert!(dg.message.contains("pocket"));
    // The cavity wall carries the tool face's name and faces into the cavity.
    assert!(region_ok(&r, "wall"));
    assert_eq!(normal(&region_faces(&r, "wall")[0]), [0.0, 1.0, 0.0]);
}

// ------------------------------------------------------------------------- suppress / unsuppress

fn with_cut_and_fillet() -> PartDocument {
    let mut d = base();
    d.features.push(sketch(
        "s_cut",
        rect_names(
            PlaneRef::origin_xy(),
            20.0,
            20.0,
            40.0,
            30.0,
            ["c1", "c2", "c3", "c4"],
        ),
    ));
    d.features.push(Feature::new(
        "cut",
        FeatureKind::Extrude(Extrude {
            profile: ProfileRef::sketch("s_cut"),
            extent: Extent::ThroughAll,
            reverse: false,
            mode: BodyMode::cut(),
            from: Default::default(),
            second: None,
            draft: None,
            thin: None,
        }),
    ));
    d.features.push(fillet(
        "round",
        vec![
            Reference::edges_between(front(), top()),
            Reference::edges_between(side("cut", "s_cut", "c1"), top()),
        ],
        1.0,
    ));
    d.regions.push(region(
        "cut_wall",
        TopoKind::Face,
        vec![side("cut", "s_cut", "c1")],
        Cardinality::ExactlyOne,
    ));
    // The fillet replaces the front-top edge, which `c33` covers on its own.
    d.regions.retain(|r| r.name.0 != "front_top");
    d
}

#[test]
fn c17_suppressed_maker_means_waiting_not_lost() {
    let env = Env::new();
    let mut d = with_cut_and_fillet();
    let r1 = env.regen(&d);
    assert_clean(&r1);
    let w1 = region_faces(&r1, "cut_wall")[0];
    d.feature_mut("cut").unwrap().suppressed = true;
    let r2 = env.regen(&d);
    // The fillet still runs on the edge that exists; the other member is dormant.
    assert!(r2.status("round").unwrap().ran());
    assert!(has(&r2, DiagCode::RefDormant));
    assert!(has(&r2, DiagCode::RefWaiting));
    assert!(!region_ok(&r2, "cut_wall"));
    assert!(
        errors(&r2).is_empty(),
        "waiting is not an error: {:#?}",
        errors(&r2)
    );
    d.feature_mut("cut").unwrap().suppressed = false;
    let r3 = env.regen(&d);
    assert_clean(&r3);
    assert_eq!(region_faces(&r3, "cut_wall")[0], w1);
}

#[test]
fn c18_suppress_base_puts_everything_downstream_waiting() {
    let env = Env::new();
    let mut d = with_cut_and_fillet();
    d.feature_mut("body").unwrap().suppressed = true;
    let r = env.regen(&d);
    assert!(errors(&r).is_empty(), "{:#?}", errors(&r));
    assert!(matches!(
        r.status("round"),
        Some(FeatureStatus::Waiting { .. })
    ));
    match r.status("round") {
        Some(FeatureStatus::Waiting { reason }) => assert_eq!(reason.maker, fid("body")),
        _ => unreachable!(),
    }
    d.feature_mut("body").unwrap().suppressed = false;
    assert_clean(&env.regen(&d));
}

#[test]
fn c19_configuration_suppression_waits() {
    let env = Env::new();
    let mut d = with_cut_and_fillet();
    d.configurations.push(Configuration {
        name: "solid".into(),
        suppress: [fid("cut")].into_iter().collect(),
        ..Default::default()
    });
    let r = env.regen_config(&d, "solid");
    let w = r
        .diagnostics
        .diagnostics
        .iter()
        .find(|x| x.code == DiagCode::RefWaiting)
        .expect("waiting");
    assert!(w.message.contains("configuration"));
    assert!(errors(&r).is_empty());
}

#[test]
fn c20_rollback_bar_makes_later_references_wait() {
    let env = Env::new();
    let mut d = with_cut_and_fillet();
    d.rollback_after = Some(fid("body"));
    let r = env.regen(&d);
    assert_eq!(r.status("cut"), Some(&FeatureStatus::RolledBack));
    assert!(!region_ok(&r, "cut_wall"));
    assert!(region_ok(&r, "top"));
    assert!(errors(&r).is_empty());
}

// ------------------------------------------------------------------------- reorders

#[test]
fn c21_reordering_independent_features_changes_nothing() {
    let env = Env::new();
    let mut d = with_cut_and_fillet();
    d.features.insert(
        2,
        Feature::new(
            "aux",
            FeatureKind::Plane(PlaneDef::Offset {
                base: PlaneRef::origin_xy(),
                distance: mm(3.0),
            }),
        ),
    );
    let r1 = env.regen(&d);
    assert!(d.reorder("aux", Some("round")));
    let r2 = env.regen(&d);
    assert_clean(&r2);
    for name in ["top", "front", "cut_wall"] {
        assert_eq!(region_faces(&r1, name), region_faces(&r2, name), "{}", name);
    }
}

#[test]
fn c22_reorder_before_maker_is_a_forward_reference() {
    let env = Env::new();
    let mut d = with_cut_and_fillet();
    assert!(d.reorder("round", Some("body")));
    let r = env.regen(&d);
    let dg = diag(&r, DiagCode::RefForward);
    assert!(dg.repairs.iter().any(|x| matches!(x, Repair::Reorder { feature, after } if feature.0 == "round" && after.0 == "cut")));
}

#[test]
fn c23_face_consumed_by_later_cut() {
    let env = Env::new();
    let mut d = base();
    d.features.push(sketch(
        "s_trim",
        rect_names(
            PlaneRef::origin_xy(),
            60.0,
            -10.0,
            100.0,
            60.0,
            ["t1", "t2", "t3", "t4"],
        ),
    ));
    d.features.push(Feature::new(
        "trim",
        FeatureKind::Extrude(Extrude {
            profile: ProfileRef::sketch("s_trim"),
            extent: Extent::ThroughAll,
            reverse: false,
            mode: BodyMode::cut(),
            from: Default::default(),
            second: None,
            draft: None,
            thin: None,
        }),
    ));
    let r = env.regen(&d);
    assert!(!region_ok(&r, "right"));
    let dg = diag(&r, DiagCode::RefConsumed);
    assert!(dg.message.contains("trim"));
    assert!(region_ok(&r, "top") && region_ok(&r, "front"));
}

#[test]
fn c24_consumer_moved_before_consumer_of_its_face() {
    let env = Env::new();
    let mut d = base();
    d.features.push(sketch(
        "s_trim",
        rect_names(
            PlaneRef::origin_xy(),
            60.0,
            -10.0,
            100.0,
            60.0,
            ["t1", "t2", "t3", "t4"],
        ),
    ));
    d.features.push(Feature::new(
        "trim",
        FeatureKind::Extrude(Extrude {
            profile: ProfileRef::sketch("s_trim"),
            extent: Extent::ThroughAll,
            reverse: false,
            mode: BodyMode::cut(),
            from: Default::default(),
            second: None,
            draft: None,
            thin: None,
        }),
    ));
    d.features.push(fillet(
        "round",
        vec![Reference::edges_between(right(), top())],
        1.0,
    ));
    let r = env.regen(&d);
    let dg = diag(&r, DiagCode::RefConsumed);
    assert!(dg.repairs.iter().any(|x| matches!(x, Repair::MoveBefore { feature, before } if feature.0 == "round" && before.0 == "trim")));
    assert!(d.reorder("round", Some("body")));
    let r2 = env.regen(&d);
    assert!(r2.status("round").unwrap().ran());
}

#[test]
fn c25_deleted_maker_reports_missing() {
    let env = Env::new();
    let mut d = with_cut_and_fillet();
    d.features.retain(|f| f.id.0 != "cut" && f.id.0 != "s_cut");
    let r = env.regen(&d);
    assert!(has(&r, DiagCode::RefMakerMissing));
}

// ------------------------------------------------------------------------- renames

#[test]
fn c26_rename_resolves_through_lock_alias() {
    let mut env = Env::new();
    let mut d = with_cut_and_fillet();
    let mut lock = LockSection::default();
    let mut minter = SeededMinter {
        seed: "test".into(),
    };
    for f in &d.features {
        lock.ensure(&ItemPath::feature(&d.path, &f.id), &mut minter);
    }
    let before = lock
        .lookup(&ItemPath::feature(&d.path, &fid("cut")))
        .unwrap()
        .uuid;
    lock.rename(
        &ItemPath::feature(&d.path, &fid("cut")),
        &ItemPath::feature(&d.path, &fid("pocket")),
    )
    .unwrap();
    d.feature_mut("cut").unwrap().id = fid("pocket");
    // References still say `cut`.
    let r0 = env.regen(&d);
    assert!(has(&r0, DiagCode::RefMakerMissing));
    env.lock = Some(lock.clone());
    let r = env.regen(&d);
    assert!(region_ok(&r, "cut_wall"));
    let dg = diag(&r, DiagCode::RefViaAlias);
    assert!(dg
        .repairs
        .iter()
        .any(|x| matches!(x, Repair::RewriteReference { .. })));
    assert_eq!(
        lock.lookup(&ItemPath::feature(&d.path, &fid("pocket")))
            .unwrap()
            .uuid,
        before
    );
    assert!(r.stats.via_alias > 0);
}

// ------------------------------------------------------------------------- patterns

fn holes(d: &mut PartDocument, count: u32, spacing: f64) {
    d.features.push(Feature::new(
        "h",
        FeatureKind::Hole(Hole {
            face: top(),
            positions: vec![HolePosition {
                key: "a".into(),
                at: PointRef::Coords {
                    x: mm(10.0),
                    y: mm(25.0),
                    z: input("thickness"),
                },
            }],
            diameter: Some(mm(4.0)),
            end: HoleEnd::ThroughAll,
            form: HoleForm::Simple,
            spec: None,
            drill_point: None,
            series: None,
        }),
    ));
    d.features.push(Feature::new(
        "row",
        FeatureKind::Pattern(Pattern {
            seeds: PatternSeeds::Features {
                features: vec![fid("h")],
            },
            layout: PatternLayout::Linear {
                direction: AxisRef::origin(OriginAxis::X),
                spacing: mm(spacing),
                count: ValueExpr::number(count as f64),
            },
            skip: Vec::new(),
        }),
    ));
}

fn wall(instance: u32, ordinal: Option<u32>) -> Reference {
    let base = EntityName::Generated(GenerationName {
        feature: fid("h"),
        role: Role::HoleWall,
        tag: SourceTag::Key { key: "a".into() },
        ordinal,
        step: None,
    });
    Reference::face(EntityName::copy(
        &base,
        &fid("row"),
        InstanceKey::Linear { step: instance },
    ))
}

#[test]
fn c27_pattern_count_increase_keeps_instances() {
    let env = Env::new();
    let mut d = base();
    holes(&mut d, 3, 15.0);
    d.regions.push(region(
        "w2",
        TopoKind::Face,
        vec![wall(2, None)],
        Cardinality::ExactlyOne,
    ));
    d.regions.push(region(
        "w2_all",
        TopoKind::Face,
        vec![wall(2, None)],
        Cardinality::AtLeastOne,
    ));
    let r1 = env.regen(&d);
    assert_clean(&r1);
    let g1 = region_faces(&r1, "w2")[0];
    // A round hole's wall is one cylindrical face about the instance's axis (x = 40 mm).
    assert!((g1.centroid[0] - 0.040).abs() < 1e-6, "{:?}", g1.centroid);
    assert!(matches!(g1.surface, SurfaceKind::Cylinder { .. }));
    assert_eq!(region_count(&r1, "w2_all"), 1);
    let i = d.feature_index(&fid("row")).unwrap();
    d.features.truncate(i - 1);
    holes(&mut d, 5, 15.0);
    let r2 = env.regen(&d);
    assert_clean(&r2);
    assert_eq!(region_faces(&r2, "w2")[0], g1);
}

#[test]
fn c28_pattern_count_decrease_reports_removed_instance() {
    let env = Env::new();
    let mut d = base();
    holes(&mut d, 4, 15.0);
    d.regions.push(region(
        "w3",
        TopoKind::Face,
        vec![wall(3, None)],
        Cardinality::ExactlyOne,
    ));
    d.regions.push(region(
        "w1",
        TopoKind::Face,
        vec![wall(1, None)],
        Cardinality::ExactlyOne,
    ));
    assert!(region_ok(&env.regen(&d), "w3"));
    let i = d.feature_index(&fid("h")).unwrap();
    d.features.truncate(i);
    holes(&mut d, 2, 15.0);
    let r = env.regen(&d);
    assert!(!region_ok(&r, "w3"));
    let dg = diag(&r, DiagCode::RefInstanceRemoved);
    assert!(dg.message.contains("1"));
    assert!(region_ok(&r, "w1"));
}

#[test]
fn c29_pattern_spacing_edit_moves_same_instance() {
    let env = Env::new();
    let mut d = base();
    holes(&mut d, 3, 15.0);
    d.regions.push(region(
        "w2",
        TopoKind::Face,
        vec![wall(2, None)],
        Cardinality::ExactlyOne,
    ));
    let i = d.feature_index(&fid("h")).unwrap();
    d.features.truncate(i);
    holes(&mut d, 3, 20.0);
    let r = env.regen(&d);
    assert!((region_faces(&r, "w2")[0].centroid[0] - 0.050).abs() < 1e-6);
}

#[test]
fn c30_pattern_skip_is_reported_by_key() {
    let env = Env::new();
    let mut d = base();
    holes(&mut d, 4, 15.0);
    d.regions.push(region(
        "w2",
        TopoKind::Face,
        vec![wall(2, None)],
        Cardinality::ExactlyOne,
    ));
    if let FeatureKind::Pattern(p) = &mut d.feature_mut("row").unwrap().kind {
        p.skip.push(InstanceKey::Linear { step: 2 });
    }
    let r = env.regen(&d);
    assert!(has(&r, DiagCode::RefInstanceRemoved));
}

#[test]
fn c31_mirror_image_named_by_instance() {
    let env = Env::new();
    let mut d = base();
    holes(&mut d, 1, 15.0);
    d.features.pop();
    d.features.push(Feature::new(
        "mid",
        FeatureKind::Plane(PlaneDef::Offset {
            base: PlaneRef::Datum {
                datum: DatumRef::OriginPlane {
                    plane: OriginPlane::Yz,
                },
            },
            distance: mm(40.0),
        }),
    ));
    d.features.push(Feature::new(
        "mir",
        FeatureKind::Mirror(Mirror {
            seeds: PatternSeeds::Features {
                features: vec![fid("h")],
            },
            plane: PlaneRef::datum("mid"),
        }),
    ));
    let base_wall = EntityName::Generated(GenerationName {
        feature: fid("h"),
        role: Role::HoleWall,
        tag: SourceTag::Key { key: "a".into() },
        ordinal: None,
        step: None,
    });
    let img = Reference::face(EntityName::copy(
        &base_wall,
        &fid("mir"),
        InstanceKey::Mirror,
    ));
    d.regions.push(region(
        "img",
        TopoKind::Face,
        vec![img],
        Cardinality::ExactlyOne,
    ));
    let r = env.regen(&d);
    assert_clean(&r);
    // The seed's wall about x = 10 mm mirrors to a wall about x = 70 mm.
    let g = region_faces(&r, "img")[0];
    assert!((g.centroid[0] - 0.070).abs() < 1e-6, "{:?}", g.centroid);
    match g.surface {
        SurfaceKind::Cylinder {
            axis_point, radius, ..
        } => {
            assert!((axis_point[0] - 0.070).abs() < 1e-9);
            assert!((radius - 0.002).abs() < 1e-12);
        }
        other => panic!("{:?}", other),
    }
}

// ------------------------------------------------------------------------- relational, imports, fallback

#[test]
fn c32_vertex_reference_survives_resize() {
    let env = Env::new();
    let mut d = base();
    d.regions.push(region(
        "corner",
        TopoKind::Vertex,
        vec![Reference::vertex_of(vec![front(), right(), top()])],
        Cardinality::ExactlyOne,
    ));
    assert!(region_ok(&env.regen(&d), "corner"));
    set_input(&mut d, "width", mm(90.0));
    set_input(&mut d, "thickness", mm(7.0));
    let r = env.regen(&d);
    assert!(region_ok(&r, "corner"));
    let v =
        r.state.bodies[0].topology.vertices.iter().find(|v| {
            approx(v.point[0], 0.09) && approx(v.point[1], 0.0) && approx(v.point[2], 0.007)
        });
    assert!(v.is_some());
}

#[test]
fn c33_edge_replaced_by_fillet_points_at_blend() {
    let env = Env::new();
    let mut d = base();
    d.features.push(fillet(
        "round",
        vec![Reference::edges_between(front(), top())],
        2.0,
    ));
    let r = env.regen(&d);
    assert!(!region_ok(&r, "front_top"));
    let dg = diag(&r, DiagCode::RefEdgeReplaced);
    assert!(dg
        .repairs
        .iter()
        .any(|x| matches!(x, Repair::UseBlendFace { .. })));
}

#[test]
fn c34_imported_cad_names_survive_source_edit() {
    let mut env = Env::new();
    let fixture = |w: f64| {
        vec![runmat_cad::regen::FakeImportBox {
            name: "housing".into(),
            min: [0.0, 0.0, 0.0],
            max: [w, 0.02, 0.01],
            face_names: [
                "mount_left".into(),
                "mount_right".into(),
                "front".into(),
                "back".into(),
                "base".into(),
                "lid".into(),
            ],
        }]
    };
    env.kernel
        .imports
        .insert("supplier/housing.step".into(), fixture(0.05));
    let mut d = PartDocument::new("housing.part");
    d.features.push(Feature::new(
        "step",
        FeatureKind::Import(Import {
            source: "supplier/housing.step".into(),
            digest: "d1".into(),
        }),
    ));
    let named = |n: &str| {
        Reference::face(EntityName::CadFile {
            import: fid("step"),
            name: n.into(),
        })
    };
    d.regions.push(region(
        "lid",
        TopoKind::Face,
        vec![named("lid")],
        Cardinality::ExactlyOne,
    ));
    d.regions.push(region(
        "right",
        TopoKind::Face,
        vec![named("mount_right")],
        Cardinality::ExactlyOne,
    ));
    d.parts.push(PartOutput {
        name: "housing".into(),
        body: Reference::body(EntityName::CadFile {
            import: fid("step"),
            name: "housing".into(),
        }),
        part_number: None,
        material: None,
        color: None,
        properties: Vec::new(),
    });
    let r1 = env.regen(&d);
    assert_clean(&r1);
    assert_eq!(normal(&region_faces(&r1, "lid")[0]), [0.0, 0.0, 1.0]);
    env.kernel
        .imports
        .insert("supplier/housing.step".into(), fixture(0.07));
    d.features[0] = Feature::new(
        "step",
        FeatureKind::Import(Import {
            source: "supplier/housing.step".into(),
            digest: "d2".into(),
        }),
    );
    let r2 = env.regen(&d);
    assert_clean(&r2);
    assert!(approx(region_faces(&r2, "right")[0].centroid[0], 0.07));
}

fn fallback_doc() -> (PartDocument, Reference) {
    let mut d = base();
    let fb = Reference::WithFallback {
        primary: Box::new(side("body", "s1", "l3")),
        fallback: GeometricFallback {
            maker: fid("body"),
            signature: GeometricSignature {
                kind: TopoKind::Face,
                normal: Some([0.0, 1.0, 0.0]),
                centroid: [0.04, 0.05, 0.005],
                measure: 0.08 * 0.01,
            },
            tolerance: 1e-6,
        },
    };
    d.regions.push(region(
        "back",
        TopoKind::Face,
        vec![fb.clone()],
        Cardinality::ExactlyOne,
    ));
    // Rename the back curve: the name is gone, the geometry is the same.
    if let FeatureKind::Sketch(s) = &mut d.features[0].kind {
        s.entities[2].id = "l3b".into();
    }
    (d, fb)
}

#[test]
fn c35_geometric_fallback_is_off_by_default() {
    let env = Env::new();
    let (d, _) = fallback_doc();
    let r = env.regen(&d);
    assert!(!region_ok(&r, "back"));
    assert!(has(&r, DiagCode::RefFallbackDisabled));
    assert_eq!(r.stats.geometric, 0);
}

#[test]
fn c36_geometric_fallback_when_allowed_is_diagnosed_and_counted() {
    let mut env = Env::new();
    env.policy.geometric_fallback = true;
    let (d, _) = fallback_doc();
    let r = env.regen(&d);
    assert!(region_ok(&r, "back"));
    assert!(has(&r, DiagCode::RefFallbackUsed));
    assert_eq!(r.stats.geometric, 1);
}

#[test]
fn c37_geometric_fallback_never_crosses_features_or_picks_nearest() {
    let mut env = Env::new();
    env.policy.geometric_fallback = true;
    let (mut d, _) = fallback_doc();
    // Move the face a little: an exact match no longer exists, and nothing nearest is taken.
    set_input(&mut d, "depth", mm(52.0));
    let r = env.regen(&d);
    assert!(!region_ok(&r, "back"));
    assert!(has(&r, DiagCode::RefFallbackFailed));
    // A signature that matches another feature's face does not resolve either.
    let mut d2 = base();
    boss(&mut d2, 60.0, 70.0, 30.0);
    d2.regions.push(region(
        "x",
        TopoKind::Face,
        vec![Reference::WithFallback {
            primary: Box::new(side("body", "s1", "zz")),
            fallback: GeometricFallback {
                maker: fid("body"),
                signature: GeometricSignature {
                    kind: TopoKind::Face,
                    normal: Some([0.0, 0.0, 1.0]),
                    centroid: [0.065, 0.025, 0.03],
                    measure: 0.01 * 0.05,
                },
                tolerance: 1e-6,
            },
        }],
        Cardinality::ExactlyOne,
    ));
    let r2 = env.regen(&d2);
    assert!(!region_ok(&r2, "x"));
    assert_eq!(r2.stats.geometric, 0);
}

#[test]
fn c38_configuration_override_keeps_references() {
    let env = Env::new();
    let mut d = with_cut_and_fillet();
    d.configurations.push(Configuration {
        name: "thin".into(),
        overrides: [("thickness".into(), mm(4.0))].into_iter().collect(),
        ..Default::default()
    });
    let r = env.regen_config(&d, "thin");
    assert_clean(&r);
    assert!(approx(region_faces(&r, "top")[0].centroid[2], 0.004));
    assert!(region_ok(&r, "cut_wall"));
}

#[test]
fn c39_configuration_table_rows_resolve_alike() {
    let env = Env::new();
    let mut d = base();
    d.configuration_tables.push(ConfigurationTable {
        name: "size".into(),
        columns: vec!["width".into(), "thickness".into()],
        rows: (1..=3)
            .map(|k| TableRow {
                key: format!("r{}", k),
                values: vec![mm(60.0 + 10.0 * k as f64), mm(4.0 + k as f64)],
            })
            .collect(),
    });
    for k in 1..=3 {
        let r = env.regen_config(&d, &format!("size.r{}", k));
        assert_clean(&r);
        assert_base_regions(
            &r,
            (60.0 + 10.0 * k as f64) / 1000.0,
            (4.0 + k as f64) / 1000.0,
        );
    }
}

#[test]
fn c40_translate_keeps_names() {
    let env = Env::new();
    let mut d = base();
    d.features.push(Feature::new(
        "shift",
        FeatureKind::BodyOp(BodyOp::Translate {
            bodies: vec![Reference::feature_body("body")],
            dx: mm(5.0),
            dy: mm(0.0),
            dz: mm(0.0),
        }),
    ));
    let r = env.regen(&d);
    assert_clean(&r);
    assert!(approx(region_faces(&r, "right")[0].centroid[0], 0.085));
}

#[test]
fn c41_sketch_on_face_follows_and_hole_by_sketch_point() {
    let env = Env::new();
    let mut d = base();
    let mut s = Sketch {
        plane: PlaneRef::Face { face: top() },
        entities: Vec::new(),
        constraints: Vec::new(),
        solved: SolvedCache::default(),
    };
    s.entities.push(SketchEntity {
        id: "p".into(),
        geometry: SketchGeometry::Point {
            at: PointExpr::mm(30.0, 20.0),
        },
        construction: false,
    });
    d.features.push(sketch("marks", s));
    d.features.push(Feature::new(
        "h",
        FeatureKind::Hole(Hole {
            face: top(),
            positions: vec![HolePosition {
                key: "p".into(),
                at: PointRef::Sketch {
                    sketch: fid("marks"),
                    entity: "p".into(),
                    role: PointRole::At,
                },
            }],
            diameter: Some(mm(6.0)),
            end: HoleEnd::Blind { depth: mm(3.0) },
            form: HoleForm::Simple,
            spec: None,
            drill_point: None,
            series: None,
        }),
    ));
    let bottom = Reference::gen_face("h", Role::HoleBottom, SourceTag::Key { key: "p".into() });
    d.regions.push(region(
        "hb",
        TopoKind::Face,
        vec![bottom],
        Cardinality::AtLeastOne,
    ));
    let r1 = env.regen(&d);
    assert_clean(&r1);
    assert!(approx(region_faces(&r1, "hb")[0].centroid[2], 0.007));
    set_input(&mut d, "thickness", mm(12.0));
    let r2 = env.regen(&d);
    assert_clean(&r2);
    let g = region_faces(&r2, "hb")[0];
    assert!(approx(g.centroid[2], 0.009) && approx(g.centroid[0], 0.03));
}

#[test]
fn c42_no_scenario_resolves_by_geometry() {
    // A sweep over the corpus documents: every resolution is by name.
    let env = Env::new();
    for d in [base(), with_cut_and_fillet()] {
        let r = env.regen(&d);
        assert_eq!(r.stats.geometric, 0);
        assert!(r.stats.exact > 0);
    }
}
