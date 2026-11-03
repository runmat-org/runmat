//! Determinism: identical hashes and JSON across runs, independent of construction order of
//! unrelated state, plus round trips of the model and index.

mod common;
use common::*;
use runmat_cad::index::{build_index, to_json, PartIndex, INDEX_FORMAT_VERSION};
use runmat_cad::model::*;
use runmat_cad::refs::*;
use runmat_cad::value::ValueExpr;

fn rich() -> PartDocument {
    let mut d = block();
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
        }),
    ));
    d.features.push(Feature::new(
        "h",
        FeatureKind::Hole(Hole {
            face: Reference::end_cap("body"),
            positions: vec![HolePosition {
                key: "a".into(),
                at: PointRef::Coords {
                    x: mm(60.0),
                    y: mm(10.0),
                    z: mm(10.0),
                },
            }],
            diameter: mm(4.0),
            depth: None,
            form: HoleForm::Simple,
        }),
    ));
    d.features.push(Feature::new(
        "row",
        FeatureKind::Pattern(Pattern {
            seeds: PatternSeeds::Features {
                features: vec![fid("h")],
            },
            layout: PatternLayout::Grid {
                first: AxisRef::origin(OriginAxis::X),
                first_spacing: mm(6.0),
                first_count: ValueExpr::number(3.0),
                second: AxisRef::origin(OriginAxis::Y),
                second_spacing: mm(8.0),
                second_count: ValueExpr::number(2.0),
            },
            skip: Vec::new(),
        }),
    ));
    d.features.push(fillet(
        "round",
        vec![Reference::edges_between(
            side("body", "s1", "l1"),
            Reference::end_cap("body"),
        )],
        1.5,
    ));
    d.regions.push(region(
        "top",
        TopoKind::Face,
        vec![Reference::end_cap("body")],
        Cardinality::AtLeastOne,
    ));
    d.regions.push(region(
        "cut_wall",
        TopoKind::Face,
        vec![side("cut", "s_cut", "c1")],
        Cardinality::ExactlyOne,
    ));
    d.publishes.push(PublishEntry {
        name: "top".into(),
        target: PublishTarget::Region {
            region: "top".into(),
        },
        auto: false,
    });
    d.links.push(Link {
        kind: LinkKind::Verifies,
        from: LocalItem::Part {
            part: "bracket".into(),
        },
        to: "checks/mass.fea#check.mass_budget".into(),
    });
    d
}

#[test]
fn d01_same_inputs_same_keys_and_json() {
    let env = Env::new();
    let d = rich();
    let a = env.regen(&d);
    let b = env.regen(&d.clone());
    assert_eq!(a.final_key, b.final_key);
    let ka: Vec<_> = a.features.iter().map(|f| f.cache_key.clone()).collect();
    let kb: Vec<_> = b.features.iter().map(|f| f.cache_key.clone()).collect();
    assert_eq!(ka, kb);
    let ja = to_json(&build_index(&d, &a, None));
    let jb = to_json(&build_index(&d, &b, None));
    assert_eq!(ja, jb);
    // Ten more runs, byte for byte.
    for _ in 0..10 {
        assert_eq!(to_json(&build_index(&d, &env.regen(&d), None)), ja);
    }
}

#[test]
fn d02_keys_are_pinned_to_their_inputs() {
    let env = Env::new();
    let d = rich();
    let a = env.regen(&d);
    let mut d2 = d.clone();
    set_input(&mut d2, "depth", mm(51.0));
    let b = env.regen(&d2);
    assert_ne!(a.final_key, b.final_key);
    // Changing an unused configuration does not change the default keys.
    let mut d3 = d.clone();
    d3.configurations.push(Configuration {
        name: "x".into(),
        ..Default::default()
    });
    assert_eq!(env.regen(&d3).final_key, a.final_key);
    // The document path seeds the root key.
    let mut d4 = d.clone();
    d4.path = "other.part".into();
    assert_ne!(env.regen(&d4).final_key, a.final_key);
}

#[test]
fn d03_index_shape_and_version() {
    let env = Env::new();
    let d = rich();
    let r = env.regen(&d);
    let idx = build_index(&d, &r, None);
    assert_eq!(idx.format_version, INDEX_FORMAT_VERSION);
    assert_eq!(idx.features.len(), d.features.len());
    assert_eq!(idx.parts.len(), 1);
    assert!(idx.parts[0].ok);
    assert_eq!(
        idx.publishes[0].entities.len(),
        idx.regions[0].entities.len()
    );
    let json = to_json(&idx);
    let back: PartIndex = serde_json::from_str(&json).expect("the projection reads back");
    assert_eq!(back, idx);
    // References carry a display form and the structured value.
    let round = idx.features.iter().find(|f| f.id.0 == "round").unwrap();
    assert!(round.references[0].display.contains("edges_between"));
    assert!(json.contains("\"format_version\": 1"));
}

#[test]
fn d04_document_round_trips_through_serde() {
    let d = rich();
    let json = serde_json::to_string_pretty(&d).unwrap();
    let back: PartDocument = serde_json::from_str(&json).unwrap();
    assert_eq!(back, d);
    let again = serde_json::to_string_pretty(&back).unwrap();
    assert_eq!(json, again);
}

#[test]
fn d05_lock_ids_appear_in_index() {
    let env = Env::new();
    let d = rich();
    let mut lock = LockSection::default();
    let mut m = SeededMinter {
        seed: "proj".into(),
    };
    let u = lock.ensure(&ItemPath::part(&d.path, &"bracket".into()), &mut m);
    let r = env.regen(&d);
    let idx = build_index(&d, &r, Some(&lock));
    assert_eq!(idx.parts[0].uuid, Some(u));
    assert_eq!(u.to_string().len(), 32);
}

#[test]
fn d06_names_never_depend_on_kernel_order() {
    // Two documents that build the same geometry with features in a different but equivalent
    // order give the same names on every face.
    let env = Env::new();
    let mut a = block();
    a.features.push(Feature::new(
        "p1",
        FeatureKind::Plane(PlaneDef::Offset {
            base: PlaneRef::origin_xy(),
            distance: mm(2.0),
        }),
    ));
    a.features.push(Feature::new(
        "p2",
        FeatureKind::Plane(PlaneDef::Offset {
            base: PlaneRef::origin_xy(),
            distance: mm(3.0),
        }),
    ));
    let mut b = a.clone();
    b.reorder("p2", Some("s1"));
    let names = |r: &runmat_cad::regen::RegenResult| -> Vec<Vec<EntityName>> {
        let mut v: Vec<Vec<EntityName>> = r.state.bodies[0]
            .names
            .faces
            .iter()
            .map(|s| s.iter().cloned().collect())
            .collect();
        v.sort();
        v
    };
    assert_eq!(names(&env.regen(&a)), names(&env.regen(&b)));
}
