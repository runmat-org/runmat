//! The usage-metering seam (sketch solves, feature and part regenerations, mate solves) and
//! the part's own PMI resolved by regeneration.

mod common;
use common::*;
use runmat_cad::assembly::source::*;
use runmat_cad::assembly::*;
use runmat_cad::diagnostics::{DiagCode, Severity, Subject};
use runmat_cad::ids::{FilePath, PublishName};
use runmat_cad::index::build_index;
use runmat_cad::metering::*;
use runmat_cad::model::datum::Frame;
use runmat_cad::model::*;
use runmat_cad::pmi::*;
use runmat_cad::refs::*;
use runmat_cad::regen::*;
use runmat_cad::units::StubUnits;
use runmat_cad::value::{NoDynamicValues, ValueExpr};
use std::cell::{Cell, RefCell};

/// The test recorder: keeps every event, with a clock that ticks 1 µs per reading.
#[derive(Default)]
struct Recorder {
    events: RefCell<Vec<UsageEvent>>,
    ticks: Cell<u64>,
    timed: bool,
}

impl MeterClock for Recorder {
    fn now_nanos(&self) -> u64 {
        let t = self.ticks.get() + 1_000;
        self.ticks.set(t);
        t
    }
}

impl UsageMeter for Recorder {
    fn record(&self, event: &UsageEvent) {
        self.events.borrow_mut().push(event.clone());
    }
    fn clock(&self) -> Option<&dyn MeterClock> {
        if self.timed {
            Some(self)
        } else {
            None
        }
    }
}

#[test]
fn u01_regeneration_reports_sketch_solves_features_and_the_part() {
    let env = Env::new();
    let mut doc = block();
    doc.features.push(fillet(
        "round",
        vec![Reference::edges_between(
            Reference::end_cap("body"),
            side("body", "s1", "l1"),
        )],
        2.0,
    ));
    let meter = Recorder {
        timed: true,
        ..Default::default()
    };
    let mut cache = RegenCache::default();
    let r = regenerate_metered(
        &doc,
        &env.ctx(),
        &RegenRequest::default(),
        &mut cache,
        &meter,
    );
    assert_clean(&r);
    let events = meter.events.borrow().clone();
    assert_eq!(events.len(), 5, "{:#?}", events);
    match &events[0] {
        UsageEvent::SketchSolve {
            file,
            feature,
            entities,
            constraints,
            outcome,
            duration_nanos,
            ..
        } => {
            assert_eq!(file.0, "bracket.part");
            assert_eq!(feature.0, "s1");
            assert_eq!((*entities, *constraints), (4, 0));
            assert_eq!(*outcome, SolveOutcome::Explicit);
            assert_eq!(*duration_nanos, Some(1_000));
        }
        e => panic!("{:?}", e),
    }
    let works: Vec<(String, FeatureWork)> = events
        .iter()
        .filter_map(|e| match e {
            UsageEvent::FeatureRegeneration { feature, work, .. } => {
                Some((feature.0.clone(), *work))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        works,
        vec![
            ("s1".to_string(), FeatureWork::Computed),
            ("body".to_string(), FeatureWork::Computed),
            ("round".to_string(), FeatureWork::Computed),
        ]
    );
    match events.last().unwrap() {
        UsageEvent::PartRegeneration {
            features,
            computed,
            cache_hits,
            failed,
            duration_nanos,
            ..
        } => {
            assert_eq!((*features, *computed, *cache_hits, *failed), (3, 3, 0, 0));
            assert!(duration_nanos.unwrap() > 0);
        }
        e => panic!("{:?}", e),
    }

    // Again with the warm cache and no clock: cache hits, and no durations at all.
    let meter = Recorder::default();
    regenerate_metered(
        &doc,
        &env.ctx(),
        &RegenRequest::default(),
        &mut cache,
        &meter,
    );
    let events = meter.events.borrow();
    match events.last().unwrap() {
        UsageEvent::PartRegeneration {
            computed,
            cache_hits,
            duration_nanos,
            ..
        } => {
            assert_eq!((*computed, *cache_hits), (1, 2), "the sketch solves again");
            assert_eq!(*duration_nanos, None);
        }
        e => panic!("{:?}", e),
    }
    assert!(events.iter().all(|e| !matches!(
        e,
        UsageEvent::FeatureRegeneration {
            duration_nanos: Some(_),
            ..
        } | UsageEvent::SketchSolve {
            duration_nanos: Some(_),
            ..
        }
    )));
    assert_eq!(meter.ticks.get(), 0, "no clock is read without one");
}

#[test]
fn u02_failed_and_skipped_features_are_metered_as_such() {
    let env = Env::new();
    let mut doc = block();
    doc.features
        .push(fillet("round", vec![Reference::end_cap("missing")], 2.0));
    doc.features
        .push(extrude("ghost", "s1", mm(5.0), BodyMode::New));
    doc.feature_mut("ghost").unwrap().suppressed = true;
    let meter = Recorder::default();
    let mut cache = RegenCache::default();
    regenerate_metered(
        &doc,
        &env.ctx(),
        &RegenRequest::default(),
        &mut cache,
        &meter,
    );
    let events = meter.events.borrow();
    match events.last().unwrap() {
        UsageEvent::PartRegeneration {
            features,
            computed,
            failed,
            ..
        } => assert_eq!((*features, *computed, *failed), (4, 2, 1)),
        e => panic!("{:?}", e),
    }
    assert!(events.iter().any(|e| matches!(
        e,
        UsageEvent::FeatureRegeneration { feature, work: FeatureWork::Skipped, .. } if feature.0 == "ghost"
    )));
}

#[test]
fn u03_mate_solves_are_metered_per_scope_and_pass() {
    let mut src = MemorySource::default();
    for (file, pn, publish) in [
        ("base.part", "B", ("top", [0.0, 0.0, 0.01])),
        ("lid.part", "L", ("bottom", [0.0, 0.0, 0.0])),
    ] {
        let mut s = DefinitionSummary::new(DefinitionRef::part(file, None));
        s.part_number = Some(pn.into());
        src.add_part(
            s,
            ConfiguredSummary {
                bbox: Some(Aabb::new([0.0; 3], [0.01; 3])),
                publishes: [(
                    PublishName::new(publish.0),
                    PublishedGeometry::Frame {
                        frame: Frame {
                            origin: publish.1,
                            ..Frame::WORLD
                        },
                    },
                )]
                .into_iter()
                .collect(),
                ..Default::default()
            },
        );
    }
    let mut top = AssemblyDocument::new("box.assembly", "box");
    top.occurrences.push(Occurrence::new(
        "base",
        DefinitionRef::part("base.part", None),
    ));
    let mut lid = Occurrence::new("lid", DefinitionRef::part("lid.part", None));
    lid.fixed = false;
    top.occurrences.push(lid);
    for (name, occ, anchor) in [("top", "base", "top"), ("bottom", "lid", "bottom")] {
        top.connectors.push(MateConnector {
            name: name.into(),
            occurrence: RelPath::new(&[occ]),
            anchor: AnchorRef::Published {
                name: anchor.into(),
            },
            offset: None,
            flip: false,
            at: None,
            z: None,
            x: None,
        });
    }
    top.mates.push(MateDef {
        name: "close".into(),
        kind: MateKind::Fasten,
        a: "top".into(),
        b: "bottom".into(),
        when: None,
        suppressed: false,
    });
    src.add_assembly(top);
    let ctx = ResolveContext {
        source: &src,
        units: &StubUnits,
        dynamic: &NoDynamicValues,
        lock: None,
        catalog: None,
        tables: None,
        gauge: None,
    };
    let meter = Recorder {
        timed: true,
        ..Default::default()
    };
    let r = resolve_metered(
        &FilePath::new("box.assembly"),
        &ResolveRequest::default(),
        &ctx,
        &meter,
    );
    assert!(
        r.diagnostics.errors().next().is_none(),
        "{:?}",
        r.diagnostics
    );
    let events = meter.events.borrow();
    assert_eq!(events.len(), 1, "{:#?}", events);
    match &events[0] {
        UsageEvent::MateSolve {
            assembly,
            scope,
            pass,
            bodies,
            grounded,
            mates,
            variables,
            converged,
            dof,
            duration_nanos,
            ..
        } => {
            assert_eq!(assembly.0, "box.assembly");
            assert_eq!(scope, "");
            assert_eq!((*pass, *bodies, *grounded, *mates, *dof), (1, 2, 1, 1, 0));
            assert!(*converged);
            assert!(*variables >= 7, "one free body: 3 + 4 quaternion variables");
            assert_eq!(*duration_nanos, Some(1_000));
        }
        e => panic!("{:?}", e),
    }
    // The unmetered entry point is the same resolve.
    let plain = resolve(
        &FilePath::new("box.assembly"),
        &ResolveRequest::default(),
        &ctx,
    );
    assert_eq!(plain.nodes.len(), r.nodes.len());
}

fn pmi() -> PmiSet {
    PmiSet {
        datums: vec![DatumFeature {
            label: "A".into(),
            on: Reference::start_cap("body"),
        }],
        tolerances: vec![GeometricTolerance {
            name: "top_parallel".into(),
            characteristic: Characteristic::Parallelism,
            on: vec![Reference::end_cap("body")],
            tolerance: mm(0.1),
            diametral: false,
            modifier: MaterialCondition::Rfs,
            projected: None,
            datums: vec![DatumReference {
                label: "A".into(),
                modifier: MaterialCondition::Rfs,
            }],
        }],
        sizes: vec![SizeTolerance {
            name: "thickness".into(),
            on: vec![Reference::start_cap("body"), Reference::end_cap("body")],
            size: ValueExpr::Toleranced {
                nominal: Box::new(input("thickness")),
                minus: Box::new(mm(0.1)),
                plus: Box::new(mm(0.1)),
            },
            fit: None,
        }],
        ..Default::default()
    }
}

#[test]
fn p01_part_pmi_is_resolved_by_regeneration() {
    let env = Env::new();
    let mut doc = block();
    doc.pmi = pmi();
    let r = env.regen(&doc);
    assert_clean(&r);
    assert_eq!(
        r.pmi.tolerances["top_parallel"],
        vec![vec![EntityName::generated(
            "body",
            Role::EndCap,
            SourceTag::None
        )]]
    );
    assert_eq!(r.pmi.sizes["thickness"].len(), 2);
    assert_eq!(r.pmi.datums["A"].len(), 1);
    // The projection carries it; a part without PMI carries none.
    let idx = build_index(&doc, &r, None);
    assert_eq!(idx.pmi.tolerances.len(), 1);
    let json = serde_json::to_string(&idx).unwrap();
    assert!(json.contains("\"top_parallel\""));
    let plain = block();
    let pj = serde_json::to_string(&build_index(&plain, &env.regen(&plain), None)).unwrap();
    assert!(!pj.contains("\"pmi\""));
    // The document round-trips with its PMI, and omits an empty set.
    let back: PartDocument = serde_json::from_str(&serde_json::to_string(&doc).unwrap()).unwrap();
    assert_eq!(back.pmi, doc.pmi);
    assert!(!serde_json::to_string(&plain).unwrap().contains("\"pmi\""));
}

#[test]
fn p02_pmi_diagnostics_join_the_regeneration_report() {
    let env = Env::new();
    let mut doc = block();
    let mut p = pmi();
    p.tolerances[0].datums.clear();
    p.datums.push(DatumFeature {
        label: "B".into(),
        on: Reference::end_cap("cut9"),
    });
    doc.pmi = p;
    let r = env.regen(&doc);
    let invalid = r.diagnostics.with_code(DiagCode::PmiInvalid);
    assert_eq!(invalid.len(), 1, "{:#?}", r.diagnostics);
    assert_eq!(invalid[0].severity, Severity::Error);
    assert_eq!(
        invalid[0].site.subject,
        Subject::Value {
            name: "pmi.top_parallel".into()
        }
    );
    let refs = r.diagnostics.with_code(DiagCode::PmiReference);
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].site.key(), "pmi.datum.B#on");
    // Ids stay unique and dense after the PMI report is appended.
    for (i, d) in r.diagnostics.diagnostics.iter().enumerate() {
        assert_eq!(d.id as usize, i);
    }
    assert_eq!(r.diagnostics, r.diagnostics.clone());
    assert_eq!(r.pmi.diagnostics.diagnostics.len(), 2);
    // A suppressed maker makes the PMI wait rather than fail.
    let mut doc = block();
    doc.pmi = pmi();
    doc.feature_mut("body").unwrap().suppressed = true;
    let r = env.regen(&doc);
    assert!(r
        .diagnostics
        .with_code(DiagCode::PmiReference)
        .iter()
        .all(|d| d.severity == Severity::Info));
}
