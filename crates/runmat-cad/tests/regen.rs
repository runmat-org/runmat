//! Regeneration: cascades, continue-past-failure, the chained cache, freeze, dependency
//! analysis, values and the sketch solver adapter.

mod common;
use common::*;
use runmat_cad::diagnostics::{Cause, DiagCode, Severity, SourceSpan, SpanMap};
use runmat_cad::ids::FeatureId;
use runmat_cad::model::*;
use runmat_cad::refs::*;
use runmat_cad::regen::*;
use runmat_cad::sketch_solver::{ConstraintSketchSolver, SketchSolver};
use runmat_cad::units::StubUnits;
use runmat_cad::value::{Evaluator, InputDecl, LetDecl, NoDynamicValues, ValueEnv, ValueExpr};

fn revolve_doc() -> PartDocument {
    let mut d = block();
    d.features.push(sketch(
        "s_rev",
        rect_names(
            PlaneRef::origin_xy(),
            0.0,
            60.0,
            10.0,
            70.0,
            ["r1", "r2", "r3", "r4"],
        ),
    ));
    d.features.push(Feature::new(
        "spin",
        FeatureKind::Revolve(Revolve {
            profile: ProfileRef::sketch("s_rev"),
            // Normal to the profile plane: the kernel refuses it.
            axis: AxisRef::origin(OriginAxis::Z),
            angle: ValueExpr::deg(360.0),
            mode: BodyMode::New,
        }),
    ));
    // Depends on the failed revolve.
    d.features.push(fillet(
        "round_spin",
        vec![Reference::edges_between(
            side("spin", "s_rev", "r1"),
            Reference::start_cap("spin"),
        )],
        1.0,
    ));
    // Independent of it.
    d.features.push(fillet(
        "round_body",
        vec![Reference::edges_between(
            side("body", "s1", "l1"),
            Reference::end_cap("body"),
        )],
        1.0,
    ));
    // Depends on the cascade, two levels down.
    d.features.push(Feature::new(
        "after",
        FeatureKind::Chamfer(Chamfer {
            edges: vec![Reference::edges_between(
                Reference::gen_face("round_spin", Role::Blend, SourceTag::None),
                Reference::end_cap("body"),
            )],
            form: ChamferForm::Equal { distance: mm(0.5) },
            flip: false,
        }),
    ));
    d
}

#[test]
fn g01_failure_is_root_dependents_cascade_independents_regenerate() {
    let env = Env::new();
    let d = revolve_doc();
    let r = env.regen(&d);
    let spin = r.status("spin").unwrap().clone();
    let FeatureStatus::Failed { diagnostic } = spin else {
        panic!("{:?}", spin)
    };
    let root = &r.diagnostics.diagnostics[diagnostic as usize];
    assert_eq!(root.code, DiagCode::KernelFailed);
    assert!(root.message.contains("profile plane"), "{}", root.message);
    assert_eq!(root.cause, Cause::Root);
    // The dependent fillet is a cascade of that root.
    match r.status("round_spin").unwrap() {
        FeatureStatus::Blocked {
            root: rf,
            diagnostic,
        } => {
            assert_eq!(rf, &fid("spin"));
            assert_eq!(
                r.diagnostics.diagnostics[*diagnostic as usize].cause,
                Cause::Cascade { root: root.id }
            );
        }
        s => panic!("{:?}", s),
    }
    // Two levels down still points at the same root.
    match r.status("after").unwrap() {
        FeatureStatus::Blocked {
            root: rf,
            diagnostic,
        } => {
            assert_eq!(rf, &fid("spin"));
            assert_eq!(
                r.diagnostics.diagnostics[*diagnostic as usize].cause,
                Cause::Cascade { root: root.id }
            );
        }
        s => panic!("{:?}", s),
    }
    // The independent branch regenerated, and the part output still resolves.
    assert!(r.status("round_body").unwrap().ran());
    assert!(r.parts.values().all(|p| p.ok));
    // One root counted, two cascades under it.
    assert_eq!(r.diagnostics.roots().count(), 1);
    assert_eq!(r.diagnostics.cascades_of(root.id).count(), 2);
}

#[test]
fn g02_value_failure_cascades_to_readers_only() {
    let env = Env::new();
    let mut d = block();
    d.inputs
        .push(InputDecl::new("bad", "length", ValueExpr::qty(3.0, "kg")));
    d.lets.push(LetDecl {
        name: "twice".into(),
        kind: None,
        value: ValueExpr::binary(
            runmat_cad::value::BinaryOp::Mul,
            ValueExpr::number(2.0),
            ValueExpr::input("bad"),
        ),
    });
    d.features.push(fillet(
        "r_bad",
        vec![Reference::edges_between(
            side("body", "s1", "l1"),
            Reference::end_cap("body"),
        )],
        1.0,
    ));
    if let FeatureKind::Fillet(f) = &mut d.features.last_mut().unwrap().kind {
        f.radius = ValueExpr::let_("twice");
    }
    d.features.push(fillet(
        "r_ok",
        vec![Reference::edges_between(
            side("body", "s1", "l3"),
            Reference::end_cap("body"),
        )],
        1.0,
    ));
    let r = env.regen(&d);
    let root = diag(&r, DiagCode::ValueFailed);
    assert_eq!(root.cause, Cause::Root);
    assert!(matches!(
        r.status("r_bad"),
        Some(FeatureStatus::Blocked { .. })
    ));
    assert!(r.status("r_ok").unwrap().ran());
    let cascade = r
        .diagnostics
        .diagnostics
        .iter()
        .find(|x| x.code == DiagCode::RegenCascade)
        .unwrap();
    assert!(matches!(cascade.cause, Cause::Cascade { .. }));
}

#[test]
fn g03_chained_cache_reuses_upstream_and_reruns_downstream() {
    let env = Env::new();
    let mut d = block();
    d.features.push(fillet(
        "round",
        vec![Reference::edges_between(
            side("body", "s1", "l1"),
            Reference::end_cap("body"),
        )],
        1.0,
    ));
    let mut cache = RegenCache::default();
    let r1 = regenerate(&d, &env.ctx(), &RegenRequest::default(), &mut cache);
    assert_eq!((cache.hits, cache.misses), (0, 2));
    // The same document again: all hits.
    let r2 = regenerate(&d, &env.ctx(), &RegenRequest::default(), &mut cache);
    assert_eq!((cache.hits, cache.misses), (2, 2));
    assert_eq!(r1.final_key, r2.final_key);
    assert_eq!(r2.status("body"), Some(&FeatureStatus::CacheHit));
    // Edit the fillet: the extrude is a hit, the fillet reruns.
    if let FeatureKind::Fillet(f) = &mut d.feature_mut("round").unwrap().kind {
        f.radius = mm(2.0);
    }
    let r3 = regenerate(&d, &env.ctx(), &RegenRequest::default(), &mut cache);
    assert_eq!(r3.status("body"), Some(&FeatureStatus::CacheHit));
    assert_eq!(r3.status("round"), Some(&FeatureStatus::Regenerated));
    // Edit thickness: everything downstream reruns because the chain changed.
    set_input(&mut d, "thickness", mm(12.0));
    let r4 = regenerate(&d, &env.ctx(), &RegenRequest::default(), &mut cache);
    assert_eq!(r4.status("body"), Some(&FeatureStatus::Regenerated));
    assert_eq!(r4.status("round"), Some(&FeatureStatus::Regenerated));
    assert_ne!(
        r4.feature("round").unwrap().cache_key,
        r3.feature("round").unwrap().cache_key
    );
}

#[test]
fn g04_suppressed_feature_leaves_chain_as_if_absent() {
    let env = Env::new();
    let mut d = block();
    d.features.push(Feature::new(
        "aux",
        FeatureKind::Plane(PlaneDef::Offset {
            base: PlaneRef::origin_xy(),
            distance: mm(1.0),
        }),
    ));
    d.features.push(fillet(
        "round",
        vec![Reference::edges_between(
            side("body", "s1", "l1"),
            Reference::end_cap("body"),
        )],
        1.0,
    ));
    d.feature_mut("aux").unwrap().suppressed = true;
    let a = env.regen(&d);
    d.features.retain(|f| f.id.0 != "aux");
    let b = env.regen(&d);
    assert_eq!(a.final_key, b.final_key);
}

#[test]
fn g05_frozen_feature_keeps_its_cached_result() {
    let env = Env::new();
    let mut d = block();
    let mut cache = RegenCache::default();
    let r1 = regenerate(&d, &env.ctx(), &RegenRequest::default(), &mut cache);
    let key = r1.feature("body").unwrap().cache_key.clone().unwrap();
    d.feature_mut("body").unwrap().frozen = Some(key);
    set_input(&mut d, "thickness", mm(30.0));
    let r2 = regenerate(&d, &env.ctx(), &RegenRequest::default(), &mut cache);
    assert_eq!(r2.status("body"), Some(&FeatureStatus::Frozen));
    assert!(has(&r2, DiagCode::RegenFrozenStale));
    let top = r2.state.bodies[0]
        .topology
        .faces
        .iter()
        .map(|f| f.geom.bbox_max[2])
        .fold(0.0, f64::max);
    assert!(approx(top, 0.01), "frozen at the old thickness");
    // Without the cache entry the freeze is reported and the feature regenerates.
    let r3 = env.regen(&d);
    assert!(has(&r3, DiagCode::RegenFrozenMissing));
    assert_eq!(r3.status("body"), Some(&FeatureStatus::Regenerated));
}

#[test]
fn g06_dependency_analysis_by_input_and_configuration() {
    let mut d = block();
    d.inputs.push(InputDecl::new("r", "length", mm(1.0)));
    d.features.push(fillet(
        "round",
        vec![Reference::edges_between(
            side("body", "s1", "l1"),
            Reference::end_cap("body"),
        )],
        1.0,
    ));
    if let FeatureKind::Fillet(f) = &mut d.feature_mut("round").unwrap().kind {
        f.radius = ValueExpr::input("r");
    }
    // A second, independent body.
    d.features.push(sketch(
        "s2",
        rect_names(
            PlaneRef::origin_xy(),
            200.0,
            0.0,
            220.0,
            20.0,
            ["n1", "n2", "n3", "n4"],
        ),
    ));
    d.features
        .push(extrude("lug", "s2", mm(5.0), BodyMode::New));
    d.parts.push(PartOutput {
        name: "lug".into(),
        body: Reference::feature_body("lug"),
        part_number: None,
        material: None,
        color: None,
        properties: Vec::new(),
    });
    d.regions.push(region(
        "lug_top",
        TopoKind::Face,
        vec![Reference::end_cap("lug")],
        Cardinality::ExactlyOne,
    ));
    d.configurations.push(Configuration {
        name: "big".into(),
        overrides: [("r".into(), mm(2.0))].into_iter().collect(),
        ..Default::default()
    });
    let g = DependencyGraph::analyse(&d);
    let a = g.affected_by_inputs(&["r".into()]);
    assert_eq!(
        a.features.iter().map(|f| f.0.as_str()).collect::<Vec<_>>(),
        vec!["round"]
    );
    assert_eq!(
        a.parts.iter().map(|p| p.0.as_str()).collect::<Vec<_>>(),
        vec!["bracket"]
    );
    let w = g.affected_by_inputs(&["width".into()]);
    assert!(
        w.features.contains(&fid("s1"))
            && w.features.contains(&fid("body"))
            && w.features.contains(&fid("round"))
    );
    assert!(!w.features.contains(&fid("lug")));
    assert!(!w.parts.contains(&"lug".into()));
    assert!(!w.regions.contains(&"lug_top".into()));
    let c = g.affected_by_configuration(&d, &"big".into()).unwrap();
    assert_eq!(c.features.len(), 1);
}

#[test]
fn g07_unknown_configuration_and_unit_errors() {
    let env = Env::new();
    let d = block();
    let r = env.regen_config(&d, "nope");
    assert!(has(&r, DiagCode::RegenInvalid));
    let mut d2 = block();
    set_input(&mut d2, "width", ValueExpr::deg(3.0));
    let r2 = env.regen(&d2);
    let v = diag(&r2, DiagCode::ValueFailed);
    assert!(v.message.contains("unit mismatch"));
    assert!(matches!(
        r2.status("s1"),
        Some(FeatureStatus::Blocked { .. })
    ));
}

#[test]
fn g08_open_profile_and_kernel_not_supported_are_specific() {
    let env = Env::new();
    let mut d = block();
    let mut s = rect_names(
        PlaneRef::origin_xy(),
        0.0,
        100.0,
        10.0,
        110.0,
        ["a", "b", "c", "e"],
    );
    s.entities.pop();
    d.features.push(sketch("open", s));
    d.features
        .push(extrude("x", "open", mm(1.0), BodyMode::New));
    // A chamfer on an edge of a fillet face: the fake kernel blends straight edges between
    // planar faces only.
    d.features.push(fillet(
        "r1",
        vec![Reference::edges_between(
            side("body", "s1", "l1"),
            Reference::end_cap("body"),
        )],
        1.0,
    ));
    let side_name = EntityName::generated(
        "body",
        Role::Side,
        SourceTag::Curve {
            sketch: fid("s1"),
            entity: runmat_cad::ids::SketchEntityId::new("l1"),
        },
    );
    let end_name = EntityName::generated("body", Role::EndCap, SourceTag::None);
    d.features.push(Feature::new(
        "ch",
        FeatureKind::Chamfer(Chamfer {
            edges: vec![Reference::edges_between(
                Reference::gen_face(
                    "r1",
                    Role::Blend,
                    SourceTag::edge_between(&side_name, &end_name),
                ),
                Reference::end_cap("body"),
            )],
            form: ChamferForm::Equal { distance: mm(0.5) },
            flip: false,
        }),
    ));
    let r = env.regen(&d);
    assert!(has(&r, DiagCode::SketchOpenProfile));
    let ns = diag(&r, DiagCode::KernelNotSupported);
    assert!(ns.message.contains("blend"), "{}", ns.message);
    assert_eq!(ns.severity, Severity::Error);
    assert!(r.parts.values().all(|p| p.ok), "the model stays usable");
}

#[test]
fn g09_spans_attach_by_site_key() {
    let env = Env::new();
    let mut d = block();
    d.features
        .push(fillet("round", vec![side("body", "s1", "zz")], 1.0));
    let mut r = env.regen(&d);
    let mut spans = SpanMap::default();
    spans.spans.insert(
        "feature.round#edges[0]".into(),
        SourceSpan {
            file: d.path.clone(),
            start: 120,
            end: 141,
        },
    );
    r.diagnostics.attach_spans(&spans);
    let dg = r
        .diagnostics
        .diagnostics
        .iter()
        .find(|x| x.site.key() == "feature.round#edges[0]")
        .unwrap();
    assert_eq!(dg.span.as_ref().map(|s| s.start), Some(120));
}

#[test]
fn g10_constraint_solver_adapter_solves_and_maps_back() {
    // A rectangle with only its first corner fixed and dimensions driving the rest.
    use runmat_cad::model::SketchConstraintKind as K;
    let mut s = rect(PlaneRef::origin_xy(), mm(0.0), mm(0.0), mm(70.0), mm(45.0));
    // Share corners so the solver sees one closed chain.
    for i in 0..4 {
        let prev = format!("l{}", if i == 0 { 4 } else { i });
        if let SketchGeometry::Line { start, .. } = &mut s.entities[i].geometry {
            *start = PointExpr::of(&prev, PointRole::End);
        }
    }
    let t = SketchTarget::entity;
    s.constraints = vec![
        SketchConstraint {
            id: None,
            kind: K::Fix {
                target: SketchTarget::point("l1", PointRole::Start),
            },
        },
        SketchConstraint {
            id: None,
            kind: K::Horizontal { line: t("l1") },
        },
        SketchConstraint {
            id: None,
            kind: K::Vertical { line: t("l2") },
        },
        SketchConstraint {
            id: None,
            kind: K::Horizontal { line: t("l3") },
        },
        SketchConstraint {
            id: None,
            kind: K::Vertical { line: t("l4") },
        },
        SketchConstraint {
            id: None,
            kind: K::Length {
                line: t("l1"),
                value: mm(80.0),
            },
        },
        SketchConstraint {
            id: None,
            kind: K::Length {
                line: t("l2"),
                value: mm(50.0),
            },
        },
    ];
    let units = StubUnits;
    let ev = Evaluator {
        units: &units,
        dynamic: &NoDynamicValues,
    };
    let solved = ConstraintSketchSolver
        .solve(&s, &ev, &ValueEnv::default())
        .expect("solves");
    assert_eq!(solved.status, SketchSolveStatus::WellDefined);
    let p = solved.point(&"l2".into(), PointRole::End).unwrap();
    assert!(
        (p[0] - 0.08).abs() < 1e-9 && (p[1] - 0.05).abs() < 1e-9,
        "{:?}",
        p
    );
    assert_eq!(solved.dof, Some(0));
    // Writing `solved` back and re-solving starts from the cache and lands in the same place.
    let mut s2 = s.clone();
    s2.solved = solved.cache.clone();
    let again = ConstraintSketchSolver
        .solve(&s2, &ev, &ValueEnv::default())
        .unwrap();
    assert_eq!(again.points, solved.points);
    // A redundant constraint is reported against our index (the later one).
    s.constraints.push(SketchConstraint {
        id: None,
        kind: K::Horizontal { line: t("l1") },
    });
    let red = ConstraintSketchSolver
        .solve(&s, &ev, &ValueEnv::default())
        .unwrap();
    assert_eq!(red.redundant, vec![7]);
    // A conflicting one fails with its index in the set.
    s.constraints.pop();
    s.constraints.push(SketchConstraint {
        id: None,
        kind: K::Length {
            line: t("l1"),
            value: mm(90.0),
        },
    });
    let bad = ConstraintSketchSolver
        .solve(&s, &ev, &ValueEnv::default())
        .unwrap_err();
    assert!(bad.constraints.contains(&7), "{:?}", bad);
}

#[test]
fn g10b_sketch_range_limits_clamp_and_release() {
    // A rectangle 80 wide whose height is only limited to [10, 30] mm, drawn 45 high: the
    // limit pulls it to 30 and leaves it free (one DOF) inside the range; drawn 20 it stays.
    use runmat_cad::model::SketchConstraintKind as K;
    let build = |height: f64| {
        let mut s = rect(
            PlaneRef::origin_xy(),
            mm(0.0),
            mm(0.0),
            mm(80.0),
            mm(height),
        );
        for i in 0..4 {
            let prev = format!("l{}", if i == 0 { 4 } else { i });
            if let SketchGeometry::Line { start, .. } = &mut s.entities[i].geometry {
                *start = PointExpr::of(&prev, PointRole::End);
            }
        }
        let t = SketchTarget::entity;
        let c = |kind| SketchConstraint { id: None, kind };
        s.constraints = vec![
            c(K::Fix {
                target: SketchTarget::point("l1", PointRole::Start),
            }),
            c(K::Horizontal { line: t("l1") }),
            c(K::Vertical { line: t("l2") }),
            c(K::Horizontal { line: t("l3") }),
            c(K::Vertical { line: t("l4") }),
            c(K::Length {
                line: t("l1"),
                value: mm(80.0),
            }),
            c(K::Limit {
                dimension: Box::new(K::Length {
                    line: t("l2"),
                    value: mm(0.0),
                }),
                min: mm(10.0),
                max: mm(30.0),
            }),
        ];
        s
    };
    let units = StubUnits;
    let ev = Evaluator {
        units: &units,
        dynamic: &NoDynamicValues,
    };
    let tall = ConstraintSketchSolver
        .solve(&build(45.0), &ev, &ValueEnv::default())
        .expect("solves");
    let p = tall.point(&"l2".into(), PointRole::End).unwrap();
    assert!((p[1] - 0.03).abs() < 1e-7, "clamped to the max: {:?}", p);
    let mid = ConstraintSketchSolver
        .solve(&build(20.0), &ev, &ValueEnv::default())
        .expect("solves");
    let p = mid.point(&"l2".into(), PointRole::End).unwrap();
    assert!((p[1] - 0.02).abs() < 1e-7, "free inside the range: {:?}", p);
    assert_eq!(mid.status, SketchSolveStatus::UnderDefined);
    // A limit on something that is not a dimension is refused.
    let mut bad = build(20.0);
    bad.constraints.push(SketchConstraint {
        id: None,
        kind: K::Limit {
            dimension: Box::new(K::Horizontal {
                line: SketchTarget::entity("l1"),
            }),
            min: mm(0.0),
            max: mm(1.0),
        },
    });
    assert!(ConstraintSketchSolver
        .solve(&bad, &ev, &ValueEnv::default())
        .is_err());
}

#[test]
fn g11_regeneration_with_the_constraint_solver_end_to_end() {
    use runmat_cad::model::SketchConstraintKind as K;
    let mut env = Env::new();
    let solver = ConstraintSketchSolver;
    let mut d = block();
    if let FeatureKind::Sketch(s) = &mut d.features[0].kind {
        for i in 0..4 {
            let prev = format!("l{}", if i == 0 { 4 } else { i });
            if let SketchGeometry::Line { start, .. } = &mut s.entities[i].geometry {
                *start = PointExpr::of(&prev, PointRole::End);
            }
        }
        s.constraints = vec![
            SketchConstraint {
                id: None,
                kind: K::Fix {
                    target: SketchTarget::point("l1", PointRole::Start),
                },
            },
            SketchConstraint {
                id: None,
                kind: K::Horizontal {
                    line: SketchTarget::entity("l1"),
                },
            },
            SketchConstraint {
                id: None,
                kind: K::Vertical {
                    line: SketchTarget::entity("l2"),
                },
            },
            SketchConstraint {
                id: None,
                kind: K::Horizontal {
                    line: SketchTarget::entity("l3"),
                },
            },
            SketchConstraint {
                id: None,
                kind: K::Vertical {
                    line: SketchTarget::entity("l4"),
                },
            },
            SketchConstraint {
                id: None,
                kind: K::Length {
                    line: SketchTarget::entity("l1"),
                    value: ValueExpr::input("width"),
                },
            },
        ];
    }
    let ctx = runmat_cad::regen::RegenContext {
        solver: &solver,
        ..env.ctx()
    };
    let r = regenerate(
        &d,
        &ctx,
        &RegenRequest::default(),
        &mut RegenCache::default(),
    );
    assert!(r.status("s1").unwrap().ran());
    assert!(r.status("body").unwrap().ran());
    let _ = &mut env;
    match &r
        .state
        .sketches
        .get(&FeatureId::new("s1"))
        .unwrap()
        .0
        .status
    {
        SketchSolveStatus::UnderDefined => {}
        s => panic!("one height dimension is missing: {:?}", s),
    }
}

#[test]
fn g12_pattern_of_unsupported_seed_is_specific() {
    let env = Env::new();
    let mut d = block();
    d.features.push(fillet(
        "round",
        vec![Reference::edges_between(
            side("body", "s1", "l1"),
            Reference::end_cap("body"),
        )],
        1.0,
    ));
    d.features.push(Feature::new(
        "p",
        FeatureKind::Pattern(Pattern {
            seeds: PatternSeeds::Features {
                features: vec![fid("round")],
            },
            layout: PatternLayout::Linear {
                direction: AxisRef::origin(OriginAxis::X),
                spacing: mm(5.0),
                count: ValueExpr::number(2.0),
            },
            skip: Vec::new(),
        }),
    ));
    let r = env.regen(&d);
    assert!(diag(&r, DiagCode::KernelNotSupported)
        .message
        .contains("patterning a fillet"));
}

#[test]
fn g13_body_pattern_copies_are_named_by_instance() {
    let env = Env::new();
    let mut d = block();
    d.features.push(Feature::new(
        "copies",
        FeatureKind::Pattern(Pattern {
            seeds: PatternSeeds::Bodies {
                bodies: vec![Reference::feature_body("body")],
            },
            layout: PatternLayout::Table {
                rows: vec![PatternRow {
                    key: "aft".into(),
                    dx: mm(0.0),
                    dy: mm(100.0),
                    dz: mm(0.0),
                }],
            },
            skip: Vec::new(),
        }),
    ));
    let copy_top = Reference::copy_of(
        Reference::end_cap("body"),
        "copies",
        InstanceKey::Row { key: "aft".into() },
    );
    d.regions.push(region(
        "aft_top",
        TopoKind::Face,
        vec![copy_top],
        Cardinality::ExactlyOne,
    ));
    let copy_body = Reference::copy_of(
        Reference::feature_body("body"),
        "copies",
        InstanceKey::Row { key: "aft".into() },
    );
    d.regions.push(region(
        "aft_body",
        TopoKind::Body,
        vec![copy_body],
        Cardinality::ExactlyOne,
    ));
    let r = env.regen(&d);
    assert_clean(&r);
    assert_eq!(r.state.bodies.len(), 2);
    assert!(approx(region_faces(&r, "aft_top")[0].centroid[1], 0.125));
    assert!(region_ok(&r, "aft_body"));
    assert!(
        r.parts.values().all(|p| p.ok),
        "the original body keeps its name"
    );
}
