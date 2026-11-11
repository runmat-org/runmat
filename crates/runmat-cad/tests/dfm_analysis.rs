//! Evaluate-tab analyses on the fake kernel (section properties, thickness, curvature, zebra,
//! deviation, undercut) and the design-for-manufacture packs over measured parts.

mod common;
use common::*;
use runmat_cad::dfm::{self, DfmGraph, DfmOptions, Process};
use runmat_cad::evaluate::*;
use runmat_cad::model::*;
use runmat_cad::props::ItemOutcome;
use runmat_cad::refs::*;
use runmat_cad::units::StubUnits;
use runmat_cad::value::{Evaluator, NoDynamicValues, ValueExpr};

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1e-300)
}

fn run(env: &Env, req: AnalysisRequest, r: &runmat_cad::regen::RegenResult) -> AnalysisResult {
    let hs: Vec<_> = r.state.bodies.iter().map(|b| &b.handle).collect();
    env.kernel.analyze(&req, &hs).unwrap()
}

#[test]
fn section_thickness_curvature_and_deviation() {
    let env = Env::new();
    let r = env.regen(&block());
    // A mid-height section of the 80 × 50 block: 80·50 mm², centroid at (40, 25).
    let AnalysisResult::SectionProperties(s) = run(
        &env,
        AnalysisRequest::SectionProperties {
            origin: [0.0, 0.0, 0.005],
            normal: [0.0, 0.0, 1.0],
        },
        &r,
    ) else {
        panic!()
    };
    assert!(close(s.area, 0.08 * 0.05, 1e-9));
    assert!(close(s.centroid[0], 0.04, 1e-9) && close(s.centroid[1], 0.025, 1e-9));
    let (ia, ib) = (0.08 * 0.05f64.powi(3) / 12.0, 0.05 * 0.08f64.powi(3) / 12.0);
    assert!(close(s.ixx.max(s.iyy), ib, 1e-6) && close(s.ixx.min(s.iyy), ia, 1e-6));
    assert!(s.ixy.abs() < 1e-15);
    // The same section on a tilted plane through the centre is sampled: 80 × 50/cos θ.
    let th = 0.3f64;
    let AnalysisResult::SectionProperties(t) = run(
        &env,
        AnalysisRequest::SectionProperties {
            origin: [0.04, 0.025, 0.005],
            normal: [0.0, th.sin(), th.cos()],
        },
        &r,
    ) else {
        panic!()
    };
    // The plane leaves the 10 mm slab before reaching y = 0 and 50: 80 × 10/sin θ.
    assert!(
        close(t.area, 0.08 * 0.01 / th.sin(), 0.03),
        "tilted {}",
        t.area
    );

    // Wall thickness: every face of the block sees 10, 50 or 80 mm across.
    let AnalysisResult::Thickness { faces } = run(
        &env,
        AnalysisRequest::Thickness {
            min: 0.02,
            max: None,
        },
        &r,
    ) else {
        panic!()
    };
    assert_eq!(faces.len(), 2, "top and bottom are 10 mm apart");
    assert!(faces.iter().all(|f| close(f.min, 0.01, 1e-9)));

    // A fillet makes a cylinder of curvature 1/r and G1 edges.
    let mut d = block();
    d.features.push(fillet(
        "round",
        vec![Reference::edges_between(
            Reference::end_cap("body"),
            side("body", "s1", "l1"),
        )],
        4.0,
    ));
    let rf = env.regen(&d);
    assert_clean(&rf);
    let AnalysisResult::Curvature { faces } = run(&env, AnalysisRequest::Curvature, &rf) else {
        panic!()
    };
    assert!(faces.iter().any(|f| close(f.max, 250.0, 1e-9)));
    let AnalysisResult::Zebra { edges } = run(&env, AnalysisRequest::Zebra, &rf) else {
        panic!()
    };
    assert_eq!(
        edges
            .iter()
            .filter(|(_, c)| *c == EdgeContinuity::G1)
            .count(),
        2
    );

    // Deviation between the block and a copy 1 mm thicker: 1 mm at most.
    let mut d2 = block();
    set_input(&mut d2, "thickness", mm(11.0));
    let r2 = env.regen(&d2);
    let AnalysisResult::Deviation { max, rms } = env
        .kernel
        .analyze(
            &AnalysisRequest::Deviation,
            &[&r.state.bodies[0].handle, &r2.state.bodies[0].handle],
        )
        .unwrap()
    else {
        panic!()
    };
    assert!(close(max, 0.001, 1e-9) && rms > 0.0 && rms < max);
}

/// A block with a T-shaped lip: an overhang facing down over a pocket floor.
fn lipped() -> PartDocument {
    let mut d = block();
    let top = PlaneRef::Face {
        face: Reference::end_cap("body"),
    };
    d.features.push(sketch(
        "s_post",
        rect(top.clone(), mm(30.0), mm(10.0), mm(40.0), mm(40.0)),
    ));
    d.features.push(Feature::new(
        "post",
        FeatureKind::Extrude(Extrude::new(
            ProfileRef::sketch("s_post"),
            Extent::Blind { distance: mm(20.0) },
            BodyMode::add(),
        )),
    ));
    d.features.push(sketch(
        "s_lip",
        rect(
            PlaneRef::Face {
                face: Reference::end_cap("post"),
            },
            mm(30.0),
            mm(10.0),
            mm(60.0),
            mm(40.0),
        ),
    ));
    d.features.push(Feature::new(
        "lip",
        FeatureKind::Extrude(Extrude::new(
            ProfileRef::sketch("s_lip"),
            Extent::Blind { distance: mm(4.0) },
            BodyMode::add(),
        )),
    ));
    d
}

#[test]
fn undercuts_block_release() {
    let env = Env::new();
    let r = env.regen(&lipped());
    assert_clean(&r);
    let AnalysisResult::Undercut { faces } = run(
        &env,
        AnalysisRequest::Undercut {
            pull: [0.0, 0.0, 1.0],
        },
        &r,
    ) else {
        panic!()
    };
    // The lip's underside (facing down over the block) and the block's top under it.
    assert!(!faces.is_empty());
    let plain = env.regen(&block());
    let AnalysisResult::Undercut { faces } = run(
        &env,
        AnalysisRequest::Undercut {
            pull: [0.0, 0.0, 1.0],
        },
        &plain,
    ) else {
        panic!()
    };
    assert!(faces.is_empty());
}

fn evaluator() -> (StubUnits, NoDynamicValues) {
    (StubUnits, NoDynamicValues)
}

fn outcome<'a>(
    results: &'a [runmat_cad::props::CheckResult],
    check: &str,
) -> &'a [(String, ItemOutcome)] {
    &results
        .iter()
        .find(|c| c.check == check)
        .unwrap_or_else(|| panic!("no check {}", check))
        .items
}

#[test]
fn machining_and_molding_packs_on_a_pocketed_block() {
    let env = Env::new();
    let mut d = block();
    let top = PlaneRef::Face {
        face: Reference::end_cap("body"),
    };
    // A sharp-cornered pocket 8 deep and 30 × 1.5 wide: too deep for its width.
    d.features.push(sketch(
        "s_p",
        rect(top.clone(), mm(10.0), mm(10.0), mm(40.0), mm(11.5)),
    ));
    let mut e = Extrude::new(
        ProfileRef::sketch("s_p"),
        Extent::Blind { distance: mm(8.0) },
        BodyMode::cut(),
    );
    e.reverse = true;
    d.features
        .push(Feature::new("pocket", FeatureKind::Extrude(e)));
    // A 0.8 mm hole 9.5 deep (11.9 D) leaving a 0.5 mm floor under it.
    d.features.push(Feature::new(
        "pin",
        FeatureKind::Hole(Hole {
            face: Reference::end_cap("body"),
            positions: vec![HolePosition {
                key: "a".into(),
                at: PointRef::Coords {
                    x: mm(70.0),
                    y: mm(40.0),
                    z: input("thickness"),
                },
            }],
            diameter: Some(mm(0.8)),
            end: HoleEnd::Blind { depth: mm(9.5) },
            form: HoleForm::Simple,
            spec: None,
            drill_point: None,
            series: None,
        }),
    ));
    let r = env.regen(&d);
    assert_clean(&r);
    let (units, dynamic) = evaluator();
    let ev = Evaluator {
        units: &units,
        dynamic: &dynamic,
    };
    let g = DfmGraph::measure(&d, &r, &ev, Some(&env.kernel), &DfmOptions::default());
    let (res, diags) = dfm::run(&[dfm::pack(Process::Machining)], &g, &units);
    let fails = |c: &str| {
        outcome(&res, c)
            .iter()
            .filter(|(_, o)| matches!(o, ItemOutcome::Fail { .. }))
            .count()
    };
    assert_eq!(fails("hole_depth"), 1, "{:#?}", res);
    assert_eq!(fails("hole_diameter"), 1);
    assert_eq!(fails("pocket_corners"), 1);
    assert_eq!(fails("pocket_depth"), 1);
    assert_eq!(
        fails("min_wall"),
        1,
        "the pin hole leaves a 0.5 mm floor: {:#?}",
        outcome(&res, "min_wall")
    );
    assert!(diags
        .diagnostics
        .iter()
        .any(|x| x.code == runmat_cad::diagnostics::DiagCode::DfmCheckFailed));

    // The plain block passes molding except for draft: its side walls are vertical.
    let plain = env.regen(&block());
    let g = DfmGraph::measure(
        &block(),
        &plain,
        &ev,
        Some(&env.kernel),
        &DfmOptions::default(),
    );
    let (res, _) = dfm::run(&[dfm::pack(Process::Molding)], &g, &units);
    assert!(matches!(
        outcome(&res, "draft")[0].1,
        ItemOutcome::Fail { .. }
    ));
    assert!(matches!(outcome(&res, "undercuts")[0].1, ItemOutcome::Pass));
    // Without an analyzer the geometric checks are unknown, not failed.
    let g = DfmGraph::measure(&block(), &plain, &ev, None, &DfmOptions::default());
    let (res, _) = dfm::run(&[dfm::pack(Process::Molding)], &g, &units);
    assert!(matches!(outcome(&res, "draft")[0].1, ItemOutcome::Unknown));
}

#[test]
fn additive_pack_sees_overhangs_voids_and_build_size() {
    let env = Env::new();
    let d = lipped();
    let r = env.regen(&d);
    let (units, dynamic) = evaluator();
    let ev = Evaluator {
        units: &units,
        dynamic: &dynamic,
    };
    let g = DfmGraph::measure(&d, &r, &ev, Some(&env.kernel), &DfmOptions::default());
    let (res, _) = dfm::run(&[dfm::pack(Process::Additive)], &g, &units);
    assert!(matches!(
        outcome(&res, "overhangs")[0].1,
        ItemOutcome::Fail { .. }
    ));
    assert!(matches!(outcome(&res, "build_x")[0].1, ItemOutcome::Pass));
    // A closed shell traps powder.
    let mut d = block();
    d.features.push(Feature::new(
        "hollow",
        FeatureKind::Shell(Shell::new(
            Reference::feature_body("body"),
            Vec::new(),
            ValueExpr::mm(2.0),
        )),
    ));
    let r = env.regen(&d);
    assert_clean(&r);
    let g = DfmGraph::measure(&d, &r, &ev, Some(&env.kernel), &DfmOptions::default());
    let (res, _) = dfm::run(&[dfm::pack(Process::Additive)], &g, &units);
    assert!(matches!(
        outcome(&res, "enclosed_voids")[0].1,
        ItemOutcome::Fail { .. }
    ));
}
