//! The part feature families: sheet metal (sections, flanges, flat patterns against K-factor
//! formulas), weldments (members, mitres, cut lists), direct edits with live relations,
//! surface bodies (planar, swept, thicken, boundary edges by name), mold analyses and shrink,
//! details (threads, embosses, ribs, library features), reference curves, datums and mate
//! connectors, and Instant3D dimension edits. Each family is regenerated on the fake kernel and
//! re-regenerated under topology-changing edits, with references checked by name.

mod common;
use common::*;
use runmat_cad::diagnostics::DiagCode;
use runmat_cad::ids::{FeatureId, InputName, SketchEntityId};
use runmat_cad::model::sheet_metal::*;
use runmat_cad::model::weldment::*;
use runmat_cad::model::*;
use runmat_cad::refs::*;
use runmat_cad::regen::records::FeatureRecord;
use runmat_cad::regen::RegenResult;
use runmat_cad::value::{InputDecl, ValueExpr};
use std::f64::consts::PI;

fn ids(v: &[&str]) -> Vec<SketchEntityId> {
    v.iter().map(|s| SketchEntityId::new(*s)).collect()
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

// =========================================================== flat-pattern correctness

#[test]
fn l_bracket_develops_by_the_k_factor_formula() {
    // Outside dimensions 50 × 30 drawn on the outer mold line (material on the left when
    // turning right): t = 2, R = 3, K = 0.44, 90°.
    let (t, r, k) = (2.0, 3.0, 0.44);
    let pts = [[0.0, 30.0], [0.0, 0.0], [50.0, 0.0]];
    let sec = section(&pts, &ids(&["up", "base"]), MaterialSide::Left, t, r).unwrap();
    assert_eq!(sec.bends.len(), 1);
    let b = &sec.bends[0];
    assert!(close(b.angle.abs(), PI / 2.0, 1e-12));
    assert!(close(b.inner_radius, r, 1e-12));
    // Flats end at the bend's tangent lines: OSSB = R + t from the outer virtual sharp.
    assert!(close(sec.flats[0].length(), 30.0 - (r + t), 1e-12));
    assert!(close(sec.flats[1].length(), 50.0 - (r + t), 1e-12));
    let rule = BendRuleValue::KFactor { k };
    let dev = develop(&sec, &rule).unwrap();
    let ba = PI / 2.0 * (r + k * t);
    assert!(close(dev.length, 25.0 + 45.0 + ba, 1e-9), "{}", dev.length);
    // Equivalent: the outside lengths minus the bend deduction.
    let bd = bend_deduction(&rule, t, r, PI / 2.0).unwrap();
    assert!(close(dev.length, 80.0 - bd, 1e-9));
    assert!(close(bd, 2.0 * (r + t) - ba, 1e-12));
    // The bend line sits at the centre of the bend zone.
    let line = &dev.bend_lines[0];
    assert!(close(line.position, 25.0 + ba / 2.0, 1e-9));
    assert!(close(line.allowance, ba, 1e-12));
}

#[test]
fn z_profile_with_obtuse_bends_and_the_other_rules() {
    // A Z: two bends of 120° and -120° (deviation 60° each), mid-plane material.
    let (t, r) = (1.5, 2.0);
    let a = 60f64.to_radians();
    let pts = [
        [0.0, 0.0],
        [40.0, 0.0],
        [40.0 + 30.0 * a.cos(), 30.0 * a.sin()],
        [80.0 + 30.0 * a.cos(), 30.0 * a.sin()],
    ];
    let sec = section(&pts, &ids(&["a", "b", "c"]), MaterialSide::Mid, t, r).unwrap();
    assert_eq!(sec.bends.len(), 2);
    assert!(sec.bends[0].angle > 0.0 && sec.bends[1].angle < 0.0);
    for b in &sec.bends {
        assert!(close(b.angle.abs(), a, 1e-12));
        assert!(close(b.inner_radius, r, 1e-12));
    }
    // Mid-plane chain: setback from the virtual sharp of the mid surface is (R + t/2)·tan(θ/2).
    let sb = (r + t / 2.0) * (a / 2.0).tan();
    assert!(close(sec.flats[0].length(), 40.0 - sb, 1e-9));
    assert!(close(sec.flats[1].length(), 30.0 - 2.0 * sb, 1e-9));
    for (rule, ba) in [
        (BendRuleValue::KFactor { k: 0.5 }, a * (r + 0.5 * t)),
        (
            BendRuleValue::Allowance { value: 4.0 },
            4.0 * a / (PI / 2.0),
        ),
        (
            BendRuleValue::Deduction { value: 2.0 },
            2.0 * (r + t) * (a / 2.0).tan() - 2.0 * a / (PI / 2.0),
        ),
    ] {
        let dev = develop(&sec, &rule).unwrap();
        let flats: f64 = sec.flats.iter().map(|f| f.length()).sum();
        assert!(close(dev.length, flats + 2.0 * ba, 1e-9), "{rule:?}");
        assert_eq!(dev.bend_lines[0].direction, BendDirection::Up);
        assert_eq!(dev.bend_lines[1].direction, BendDirection::Down);
    }
    // A bend table interpolates in angle for the matching thickness and radius.
    let table = BendTable {
        name: "shop".into(),
        rows: vec![
            BendTableRow {
                thickness: t,
                radius: r,
                angle: PI / 4.0,
                allowance: 2.0,
            },
            BendTableRow {
                thickness: t,
                radius: r,
                angle: PI / 2.0,
                allowance: 3.6,
            },
        ],
    };
    let ba = bend_allowance(
        &BendRuleValue::Table {
            table: table.clone(),
        },
        t,
        r,
        a,
    )
    .unwrap();
    let w = (a - PI / 4.0) / (PI / 4.0);
    assert!(close(ba, 2.0 + w * 1.6, 1e-12));
    assert!(bend_allowance(&BendRuleValue::Table { table }, 3.0, r, a).is_err());
    // Folding back on itself in one bend is refused (a hem is its own feature).
    assert!(section(
        &[[0.0, 0.0], [10.0, 0.0], [0.0, 0.0001]],
        &ids(&["x", "y"]),
        MaterialSide::Left,
        1.0,
        1.0
    )
    .is_err());
}

// ================================================================ sheet metal on the kernel

fn rules() -> SheetMetalRules {
    SheetMetalRules {
        thickness: mm(2.0),
        bend_radius: mm(3.0),
        rule: BendRule::KFactor {
            k: ValueExpr::number(0.44),
        },
        gauge: None,
        relief: ReliefKind::Tear,
    }
}

fn chain_sketch(id: &str, pts: &[(f64, f64)], names: &[&str]) -> Feature {
    let n = names.len();
    let mut s = Sketch {
        plane: PlaneRef::Datum {
            datum: DatumRef::OriginPlane {
                plane: OriginPlane::Zx,
            },
        },
        entities: Vec::new(),
        constraints: Vec::new(),
        solved: Default::default(),
    };
    for k in 0..n {
        s.entities.push(SketchEntity {
            id: SketchEntityId::new(names[k]),
            geometry: SketchGeometry::Line {
                start: PointExpr::mm(pts[k].0, pts[k].1),
                end: PointExpr::mm(pts[k + 1].0, pts[k + 1].1),
            },
            construction: false,
        });
    }
    sketch(id, s)
}

fn sheet_doc(chain: &[(f64, f64)], names: &[&str]) -> PartDocument {
    let mut d = PartDocument::new("bracket_sm.part");
    d.features.push(chain_sketch("s_chain", chain, names));
    d.features.push(Feature::new(
        "base",
        FeatureKind::SheetMetal(SheetMetalFeature::BaseFlange {
            profile: ProfileRef::sketch("s_chain"),
            depth: Some(mm(40.0)),
            side: MaterialSide::Left,
            rules: rules(),
        }),
    ));
    d
}

fn sheet_face(curve: &str, side: u32) -> Reference {
    Reference::face(EntityName::Generated(GenerationName {
        feature: FeatureId::new("base"),
        role: Role::SheetFace,
        tag: SourceTag::Key { key: curve.into() },
        ordinal: Some(side),
    }))
}

fn resolves(r: &RegenResult, site: &str) -> bool {
    r.parts.get(&site.into()).map(|o| o.ok).unwrap_or(false)
}

fn count_faces(r: &RegenResult, d: &PartDocument, refs: &[Reference]) -> Vec<usize> {
    let _ = d;
    refs.iter()
        .map(|x| {
            r.state
                .bodies
                .iter()
                .flat_map(|b| b.names.faces.iter())
                .filter(|names| match x {
                    Reference::Name { name, .. } => names.iter().any(|n| name.matches(n)),
                    _ => false,
                })
                .count()
        })
        .collect()
}

#[test]
fn base_flange_flat_pattern_and_edge_flange_on_the_kernel() {
    let env = Env::new();
    let mut d = sheet_doc(&[(0.0, 30.0), (0.0, 0.0), (50.0, 0.0)], &["up", "base"]);
    let r = env.regen(&d);
    assert_clean(&r);
    assert_eq!(r.state.bodies.len(), 1);
    // Every flat face and the bend's two faces by name.
    let bend = |o: u32| {
        Reference::face(EntityName::Generated(GenerationName {
            feature: FeatureId::new("base"),
            role: Role::BendFace,
            tag: SourceTag::Key {
                key: "up+base".into(),
            },
            ordinal: Some(o),
        }))
    };
    let refs = vec![
        sheet_face("up", 0),
        sheet_face("up", 1),
        sheet_face("base", 0),
        sheet_face("base", 1),
        bend(0),
        bend(1),
    ];
    assert_eq!(count_faces(&r, &d, &refs), vec![1; 6]);
    let Some(FeatureRecord::Sheet { rules, section, .. }) = r.state.records.get(&fid("base"))
    else {
        panic!("no sheet record")
    };
    assert!(close(rules.thickness, 0.002, 1e-15));
    assert_eq!(section.as_ref().unwrap().bends.len(), 1);

    // The flat pattern from the base flat.
    d.features.push(Feature::new(
        "flat",
        FeatureKind::SheetMetal(SheetMetalFeature::FlatPattern {
            fixed_face: sheet_face("base", 0),
            grain: None,
        }),
    ));
    let r = env.regen(&d);
    assert_clean(&r);
    let Some(FeatureRecord::FlatPattern { development, .. }) = r.state.records.get(&fid("flat"))
    else {
        panic!("no flat record")
    };
    let ba = PI / 2.0 * (0.003 + 0.44 * 0.002);
    assert!(close(development.length, 0.025 + 0.045 + ba, 1e-12));
    assert_eq!(r.state.bodies.len(), 2);

    // An edge flange on the far end of the base flat: the edge between its far side face and
    // its end edge face.
    let end_face = Reference::face(EntityName::Generated(GenerationName {
        feature: FeatureId::new("base"),
        role: Role::SheetEdge,
        tag: SourceTag::Key { key: "base".into() },
        ordinal: Some(1),
    }));
    d.features.push(Feature::new(
        "lip",
        FeatureKind::SheetMetal(SheetMetalFeature::EdgeFlange {
            edges: vec![Reference::edges_between(sheet_face("base", 0), end_face)],
            length: mm(12.0),
            angle: ValueExpr::deg(90.0),
            length_reference: LengthReference::Tangent,
            position: FlangePosition::BendOutside,
            radius: None,
            relief: None,
        }),
    ));
    let r = env.regen(&d);
    assert!(r.status("lip").unwrap().ran(), "{:?}", r.diagnostics);
    let lip_bends = r
        .state
        .bodies
        .iter()
        .flat_map(|b| b.names.faces.iter())
        .filter(|ns| {
            ns.iter().any(|n| {
                matches!(n, EntityName::Generated(g) if g.feature.0 == "lip" && g.role == Role::BendFace)
            })
        })
        .count();
    assert_eq!(lip_bends, 2);
}

#[test]
fn sheet_references_survive_a_segment_added_to_the_chain() {
    let env = Env::new();
    let mut d = sheet_doc(&[(0.0, 30.0), (0.0, 0.0), (50.0, 0.0)], &["up", "base"]);
    d.parts.push(PartOutput {
        name: "sm".into(),
        body: Reference::feature_body("base"),
        part_number: None,
        material: None,
        properties: Vec::new(),
    });
    let before = env.regen(&d);
    assert!(resolves(&before, "sm"));
    // A third segment turns the L into a U: one more flat, one more bend.
    let d2 = sheet_doc(
        &[(0.0, 30.0), (0.0, 0.0), (50.0, 0.0), (50.0, 30.0)],
        &["up", "base", "back"],
    );
    let r = env.regen(&d2);
    assert_clean(&r);
    let refs = vec![
        sheet_face("up", 0),
        sheet_face("base", 1),
        sheet_face("back", 0),
    ];
    assert_eq!(count_faces(&r, &d2, &refs), vec![1, 1, 1]);
    // A thicker gauge regenerates with the same names.
    let mut d3 = d2.clone();
    if let FeatureKind::SheetMetal(SheetMetalFeature::BaseFlange { rules, .. }) =
        &mut d3.features[1].kind
    {
        rules.thickness = mm(3.0);
    }
    let r = env.regen(&d3);
    assert_clean(&r);
    assert_eq!(count_faces(&r, &d3, &refs), vec![1, 1, 1]);
}

// ===================================================================== weldments

fn frame_doc(corner: CornerTreatment) -> PartDocument {
    let mut d = PartDocument::new("frame.part");
    let mut s = Sketch {
        plane: PlaneRef::origin_xy(),
        entities: Vec::new(),
        constraints: Vec::new(),
        solved: Default::default(),
    };
    let p = [(0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)];
    let names = ["bottom", "right", "top", "left"];
    for k in 0..4 {
        s.entities.push(SketchEntity {
            id: SketchEntityId::new(names[k]),
            geometry: SketchGeometry::Line {
                start: PointExpr::mm(p[k].0, p[k].1),
                end: PointExpr::mm(p[(k + 1) % 4].0, p[(k + 1) % 4].1),
            },
            construction: false,
        });
    }
    d.features.push(sketch("layout", s));
    d.features.push(Feature::new(
        "members",
        FeatureKind::Weldment(WeldmentFeature::StructuralMember {
            groups: vec![MemberGroup {
                sketch: fid("layout"),
                segments: ids(&names),
                profile: MemberProfile::RectTube {
                    width: 0.02,
                    height: 0.02,
                    wall: 0.002,
                },
                material: Some("S235".into()),
                corner,
                rotation: ValueExpr::deg(0.0),
            }],
        }),
    ));
    d
}

#[test]
fn a_mitred_frame_makes_four_members_and_a_two_row_cut_list() {
    let env = Env::new();
    let d = frame_doc(CornerTreatment::Miter);
    let r = env.regen(&d);
    assert!(r.status("members").unwrap().ran(), "{:?}", r.diagnostics);
    assert_eq!(r.state.bodies.len(), 4);
    let Some(FeatureRecord::Weldment { cut_list, cuts }) = r.state.records.get(&fid("members"))
    else {
        panic!()
    };
    assert_eq!(cuts.len(), 4);
    assert_eq!(cut_list.len(), 2);
    // Long side of each mitre: centre-line length plus half the depth at each end.
    assert!(close(cut_list[0].length, 0.2 + 0.02, 1e-12));
    assert!(close(cut_list[1].length, 0.1 + 0.02, 1e-12));
    for row in cut_list {
        assert_eq!(row.quantity, 2);
        assert!(close(row.start_angle, PI / 4.0, 1e-12) && close(row.end_angle, PI / 4.0, 1e-12));
    }
    // Member faces by member and profile curve.
    let side = Reference::face(EntityName::Generated(GenerationName {
        feature: fid("members"),
        role: Role::MemberSide,
        tag: SourceTag::Key {
            key: "top/top".into(),
        },
        ordinal: None,
    }));
    // The top member's top face (the mitred ends may leave it in pieces on the grid): every
    // face with the name lies on the member's top plane.
    assert!(count_faces(&r, &d, std::slice::from_ref(&side))[0] >= 1);
    let Reference::Name { name, .. } = &side else {
        unreachable!()
    };
    for b in &r.state.bodies {
        for (fi, ns) in b.names.faces.iter().enumerate() {
            if ns.iter().any(|n| name.matches(n)) {
                assert!(approx(b.topology.faces[fi].geom.centroid[2], 0.01));
            }
        }
    }
    // Butt joints: the through members carry the full outside length, square ends.
    let r = env.regen(&frame_doc(CornerTreatment::Butt1));
    let Some(FeatureRecord::Weldment { cut_list, .. }) = r.state.records.get(&fid("members"))
    else {
        panic!()
    };
    assert!(cut_list
        .iter()
        .all(|c| c.start_angle == 0.0 && c.end_angle == 0.0));
}

// ================================================================== direct edits

#[test]
fn offset_face_keeps_names_and_live_relations_fail_loudly() {
    let env = Env::new();
    let mut d = block();
    d.features.push(Feature::new(
        "pull",
        FeatureKind::Direct(DirectEdit::MoveFaces {
            faces: vec![Reference::end_cap("body")],
            motion: DirectMotion::Offset { distance: mm(5.0) },
            keep: vec![LiveRelation::Parallel {
                a: Reference::end_cap("body"),
                b: Reference::start_cap("body"),
            }],
        }),
    ));
    d.regions.push(region(
        "top",
        TopoKind::Face,
        vec![Reference::end_cap("body")],
        Cardinality::ExactlyOne,
    ));
    let r = env.regen(&d);
    assert_clean(&r);
    let top = region_faces(&r, "top");
    assert!(approx(top[0].centroid[2], 0.015));
    // An upstream edit still flows through the direct edit.
    let mut d2 = d.clone();
    set_input(&mut d2, "thickness", mm(20.0));
    assert!(approx(
        region_faces(&env.regen(&d2), "top")[0].centroid[2],
        0.025
    ));
    // Rotating the top face would break the kept parallelism.
    if let FeatureKind::Direct(DirectEdit::MoveFaces { motion, .. }) = &mut d.features[2].kind {
        *motion = DirectMotion::Rotate {
            axis: AxisRef::origin(OriginAxis::X),
            angle: ValueExpr::deg(5.0),
        };
    }
    let r = env.regen(&d);
    assert!(has(&r, DiagCode::DirectRelationBroken));
}

#[test]
fn delete_face_heals_a_pocket_and_a_dimension_edit_is_recorded() {
    let env = Env::new();
    let mut d = block();
    d.features.push(sketch(
        "s_pocket",
        rect_names(
            PlaneRef::Face {
                face: Reference::end_cap("body"),
            },
            20.0,
            10.0,
            40.0,
            30.0,
            ["p1", "p2", "p3", "p4"],
        ),
    ));
    d.features.push(Feature::new(
        "pocket",
        FeatureKind::Extrude(Extrude {
            profile: ProfileRef::sketch("s_pocket"),
            extent: Extent::Blind { distance: mm(4.0) },
            reverse: true,
            mode: BodyMode::cut(),
            from: Default::default(),
            second: None,
            draft: None,
            thin: None,
        }),
    ));
    let pocket_faces = total_faces(&env.regen(&d));
    d.features.push(Feature::new(
        "heal",
        FeatureKind::Direct(DirectEdit::DeleteFaces {
            faces: vec![
                side("pocket", "s_pocket", "p1"),
                side("pocket", "s_pocket", "p2"),
                side("pocket", "s_pocket", "p3"),
                side("pocket", "s_pocket", "p4"),
                Reference::end_cap("pocket"),
            ],
            heal: true,
        }),
    ));
    let r = env.regen(&d);
    assert_clean(&r);
    assert_eq!(total_faces(&r), pocket_faces - 5);
    // Instant3D: drag the block's thickness; the edit lands on the input.
    let edit = apply_dimension_edit(
        &mut d,
        DimensionTarget::Input {
            input: InputName::new("thickness"),
        },
        mm(12.0),
    )
    .unwrap();
    assert_eq!(edit.from, mm(10.0));
    assert!(apply_dimension_edit(
        &mut d,
        DimensionTarget::Feature {
            feature: fid("pocket"),
            param: "radius".into()
        },
        mm(1.0)
    )
    .is_err());
    let e2 = apply_dimension_edit(
        &mut d,
        DimensionTarget::Feature {
            feature: fid("pocket"),
            param: "extent.distance".into(),
        },
        mm(6.0),
    )
    .unwrap();
    assert_eq!(e2.from, mm(4.0));
}

// ================================================================= surface bodies

#[test]
fn planar_patch_thickens_and_boundary_edges_resolve_by_name() {
    let env = Env::new();
    let mut d = PartDocument::new("skin.part");
    d.features.push(sketch(
        "s",
        rect_names(
            PlaneRef::origin_xy(),
            0.0,
            0.0,
            40.0,
            20.0,
            ["a", "b", "c", "e"],
        ),
    ));
    d.features.push(Feature::new(
        "patch",
        FeatureKind::Surface(SurfaceFeature::Planar {
            profile: ProfileRef::sketch("s"),
        }),
    ));
    let r = env.regen(&d);
    assert_clean(&r);
    let topo = &r.state.bodies[0].topology;
    assert_eq!(topo.faces.len(), 1);
    assert_eq!(topo.edges.iter().filter(|e| e.laminar).count(), 4);
    // A boundary edge by its own name.
    let edge_b = Reference::Name {
        kind: TopoKind::Edge,
        name: EntityName::Generated(GenerationName {
            feature: fid("patch"),
            role: Role::Boundary,
            tag: SourceTag::Curve {
                sketch: fid("s"),
                entity: SketchEntityId::new("b"),
            },
            ordinal: None,
        }),
    };
    d.features.push(Feature::new(
        "grow",
        FeatureKind::Surface(SurfaceFeature::Extend {
            edges: vec![edge_b],
            distance: mm(10.0),
        }),
    ));
    d.features.push(Feature::new(
        "solid",
        FeatureKind::Surface(SurfaceFeature::Thicken {
            bodies: vec![Reference::feature_body("patch")],
            thickness: mm(3.0),
            both_sides: false,
        }),
    ));
    let r = env.regen(&d);
    assert!(r.status("grow").unwrap().ran(), "{:?}", r.diagnostics);
    assert!(r.status("solid").unwrap().ran(), "{:?}", r.diagnostics);
    let solid = &r.state.bodies[0];
    assert_eq!(solid.topology.edges.iter().filter(|e| e.laminar).count(), 0);
    let (lo, hi) =
        solid
            .topology
            .faces
            .iter()
            .fold(([f64::MAX; 3], [f64::MIN; 3]), |(mut lo, mut hi), f| {
                for a in 0..3 {
                    lo[a] = lo[a].min(f.geom.bbox_min[a]);
                    hi[a] = hi[a].max(f.geom.bbox_max[a]);
                }
                (lo, hi)
            });
    assert!(
        approx(hi[0] - lo[0], 0.05) && approx(hi[1] - lo[1], 0.02) && approx(hi[2] - lo[2], 0.003)
    );
}

#[test]
fn a_swept_open_sheet_names_a_face_per_curve() {
    let env = Env::new();
    let mut d = PartDocument::new("duct.part");
    d.features.push(chain_sketch(
        "c",
        &[(0.0, 0.0), (30.0, 0.0), (30.0, 20.0)],
        &["floor", "wall"],
    ));
    d.features.push(Feature::new(
        "skin",
        FeatureKind::Surface(SurfaceFeature::Extrude {
            profile: ProfileRef::sketch("c"),
            start: mm(0.0),
            end: mm(25.0),
        }),
    ));
    let r = env.regen(&d);
    assert_clean(&r);
    let t = &r.state.bodies[0].topology;
    assert_eq!(t.faces.len(), 2);
    // Two faces, one shared edge, six boundary edges.
    assert_eq!(t.edges.iter().filter(|e| !e.laminar).count(), 1);
    assert_eq!(t.edges.iter().filter(|e| e.laminar).count(), 6);
}

// ===================================================================== mold

#[test]
fn shrink_scales_and_the_parting_analysis_classifies_faces() {
    let env = Env::new();
    let mut d = block();
    d.features.push(Feature::new(
        "draft_check",
        FeatureKind::Mold(MoldFeature::PartingLine {
            body: Reference::feature_body("body"),
            pull: AxisRef::origin(OriginAxis::Z),
            edges: Vec::new(),
            min_draft: ValueExpr::deg(1.0),
        }),
    ));
    d.features.push(Feature::new(
        "shrink",
        FeatureKind::Mold(MoldFeature::Shrink {
            bodies: vec![Reference::feature_body("body")],
            x: ValueExpr::number(0.005),
            y: ValueExpr::number(0.005),
            z: ValueExpr::number(0.005),
        }),
    ));
    let r = env.regen(&d);
    assert_clean(&r);
    let Some(FeatureRecord::Mold {
        positive,
        negative,
        insufficient,
        ..
    }) = r.state.records.get(&fid("draft_check"))
    else {
        panic!()
    };
    assert_eq!(
        (positive.len(), negative.len(), insufficient.len()),
        (1, 1, 4)
    );
    let top = r.state.bodies[0]
        .topology
        .faces
        .iter()
        .map(|f| f.geom.bbox_max[0])
        .fold(f64::MIN, f64::max);
    assert!(approx(top, 0.08 * 1.005));
}

// ================================================================= details

#[test]
fn threads_embosses_ribs_curves_connectors_and_library_features() {
    let env = Env::new();
    let mut d = block();
    // An emboss on the top face.
    d.features.push(sketch(
        "s_logo",
        rect_names(
            PlaneRef::Face {
                face: Reference::end_cap("body"),
            },
            10.0,
            10.0,
            20.0,
            15.0,
            ["g1", "g2", "g3", "g4"],
        ),
    ));
    d.features.push(Feature::new(
        "logo",
        FeatureKind::Emboss(Emboss {
            profile: ProfileRef::sketch("s_logo"),
            face: Reference::end_cap("body"),
            height: mm(1.0),
            wrap: false,
        }),
    ));
    // A helix about Z through (10, 0, 0): 3 turns of 2 mm.
    d.features.push(Feature::new(
        "coil",
        FeatureKind::Curve(CurveDef::Helix {
            axis: AxisRef::origin(OriginAxis::Z),
            start: PointRef::Coords {
                x: mm(10.0),
                y: mm(0.0),
                z: mm(0.0),
            },
            pitch: mm(2.0),
            turns: ValueExpr::number(3.0),
            left_hand: false,
            taper: None,
        }),
    ));
    // A three-point plane and a mate connector on the top face.
    d.features.push(Feature::new(
        "tilted",
        FeatureKind::Plane(PlaneDef::ThreePoint {
            a: PointRef::Coords {
                x: mm(0.0),
                y: mm(0.0),
                z: mm(0.0),
            },
            b: PointRef::Coords {
                x: mm(10.0),
                y: mm(0.0),
                z: mm(0.0),
            },
            c: PointRef::Coords {
                x: mm(0.0),
                y: mm(10.0),
                z: mm(10.0),
            },
        }),
    ));
    d.features.push(Feature::new(
        "mc",
        FeatureKind::MateConnector(MateConnectorDef {
            placement: ConnectorPlacement::Frame {
                frame: FrameDef::OnFace {
                    face: Reference::end_cap("body"),
                },
            },
            offset: Some([mm(0.0), mm(0.0), mm(2.0)]),
            rotation: None,
            flip: false,
        }),
    ));
    let r = env.regen(&d);
    assert_clean(&r);
    let tops = r.state.bodies[0]
        .names
        .faces
        .iter()
        .filter(|ns| {
            ns.iter()
                .any(|n| matches!(n, EntityName::Generated(g) if g.role == Role::EmbossTop))
        })
        .count();
    assert_eq!(tops, 1);
    let Some(FeatureRecord::Curve(c)) = r.state.records.get(&fid("coil")) else {
        panic!()
    };
    let expect = 3.0 * ((2.0 * PI * 0.01).powi(2) + 0.002f64.powi(2)).sqrt();
    assert!(
        close(c.length(), expect, expect * 2e-3),
        "{} vs {}",
        c.length(),
        expect
    );
    let Some(DatumValue::Plane { frame }) = r.state.datums.get(&fid("tilted")) else {
        panic!()
    };
    let s = std::f64::consts::FRAC_1_SQRT_2;
    assert!(close(frame.z[1], -s, 1e-12) && close(frame.z[2], s, 1e-12));
    let Some(DatumValue::Frame { frame }) = r.state.datums.get(&fid("mc")) else {
        panic!()
    };
    assert!(approx(frame.origin[2], 0.012));

    // A cosmetic thread on a hole wall, and a library boss expanded from a feature def.
    let mut d2 = block();
    d2.features.push(Feature::new(
        "tap",
        FeatureKind::Hole(Hole {
            face: Reference::end_cap("body"),
            positions: vec![HolePosition {
                key: "m6".into(),
                at: PointRef::Coords {
                    x: mm(40.0),
                    y: mm(25.0),
                    z: mm(10.0),
                },
            }],
            diameter: Some(mm(5.0)),
            end: HoleEnd::ThroughAll,
            form: HoleForm::Simple,
            spec: None,
            drill_point: None,
            series: None,
        }),
    ));
    d2.features.push(Feature::new(
        "m6_thread",
        FeatureKind::Thread(Thread {
            face: Reference::gen_face("tap", Role::HoleWall, SourceTag::Key { key: "m6".into() }),
            spec: ThreadSpec {
                designation: "M6x1".into(),
                pitch: mm(1.0),
                depth: mm(0.54),
                left_hand: false,
                class: Some("6H".into()),
            },
            length: None,
            modelled: false,
        }),
    ));
    d2.feature_defs.push(FeatureDef {
        name: "boss".into(),
        params: vec![InputDecl::new("h", "length", mm(5.0))],
        slots: vec![],
        features: vec![
            sketch(
                "s",
                rect_names(
                    PlaneRef::origin_xy(),
                    60.0,
                    10.0,
                    70.0,
                    20.0,
                    ["q1", "q2", "q3", "q4"],
                ),
            ),
            Feature::new(
                "e",
                FeatureKind::Extrude(Extrude {
                    profile: ProfileRef::sketch("s"),
                    extent: Extent::Blind {
                        distance: input("h"),
                    },
                    reverse: true,
                    mode: BodyMode::add(),
                    from: Default::default(),
                    second: None,
                    draft: None,
                    thin: None,
                }),
            ),
        ],
    });
    d2.features.push(Feature::new(
        "boss1",
        FeatureKind::Library(LibraryInstance {
            def: "feature_def.boss".into(),
            args: [(InputName::new("h"), mm(8.0))].into_iter().collect(),
            bindings: Default::default(),
            mode: BodyMode::add(),
        }),
    ));
    let r = env.regen(&d2);
    assert_clean(&r);
    assert!(matches!(
        r.state.records.get(&fid("m6_thread")),
        Some(FeatureRecord::CosmeticThread { .. })
    ));
    assert!(r.status("boss1.e").unwrap().ran());
    let low = r.state.bodies[0]
        .topology
        .faces
        .iter()
        .map(|f| f.geom.bbox_min[2])
        .fold(f64::MAX, f64::min);
    assert!(approx(low, -0.008));
}

#[test]
fn not_lowered_cases_fail_with_the_named_case() {
    let env = Env::new();
    let mut d = sheet_doc(&[(0.0, 30.0), (0.0, 0.0), (50.0, 0.0)], &["up", "base"]);
    d.features.push(Feature::new(
        "jog",
        FeatureKind::SheetMetal(SheetMetalFeature::Fold { bends: Vec::new() }),
    ));
    let r = env.regen(&d);
    assert!(has(&r, DiagCode::KernelNotSupported));
    assert!(r.status("base").unwrap().ran());
}

// ============================================================ evaluate-tab hooks

#[test]
fn mass_properties_and_draft_through_the_analysis_seam() {
    use runmat_cad::evaluate::*;
    let env = Env::new();
    let r = env.regen(&block());
    let h = &r.state.bodies[0].handle;
    let rho = 2700.0;
    let AnalysisResult::MassProperties(m) = env
        .kernel
        .analyze(&AnalysisRequest::MassProperties { density: rho }, &[h])
        .unwrap()
    else {
        panic!()
    };
    let (a, b, c) = (0.08, 0.05, 0.01);
    assert!(close(m.volume, a * b * c, 1e-15));
    assert!(close(m.centroid[0], 0.04, 1e-12) && close(m.centroid[2], 0.005, 1e-12));
    let mass = rho * a * b * c;
    assert!(close(m.inertia[2][2], mass * (a * a + b * b) / 12.0, 1e-12));
    assert!(close(m.inertia[0][1], 0.0, 1e-15));
    let AnalysisResult::Draft(d) = env
        .kernel
        .analyze(
            &AnalysisRequest::Draft {
                pull: [0.0, 0.0, 1.0],
                min: 1f64.to_radians(),
            },
            &[h],
        )
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(d.insufficient.len(), 4);
    // A block's edges are all sharp (G0).
    let AnalysisResult::Zebra { edges } =
        env.kernel.analyze(&AnalysisRequest::Zebra, &[h]).unwrap()
    else {
        panic!()
    };
    assert_eq!(edges.len(), 12);
    assert!(edges.iter().all(|(_, c)| *c == EdgeContinuity::G0));
}
