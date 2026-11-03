//! Cross-file publish/consume, auto-publish, and pinned vs live contexts.

mod common;
use common::*;
use runmat_cad::diagnostics::{DiagCode, Repair};
use runmat_cad::ids::{FilePath, PublishName, RevisionId};
use runmat_cad::model::*;
use runmat_cad::refs::*;
use runmat_cad::regen::RegenResult;

/// The skeleton: a block publishing its top face and a datum.
fn skeleton() -> PartDocument {
    let mut d = block();
    d.path = FilePath::new("interfaces/skeleton.part");
    d.features.push(Feature::new(
        "station",
        FeatureKind::Plane(PlaneDef::Offset {
            base: PlaneRef::origin_xy(),
            distance: mm(25.0),
        }),
    ));
    d.publishes.push(PublishEntry {
        name: PublishName::new("mount_face"),
        target: PublishTarget::Entities {
            kind: TopoKind::Face,
            references: vec![Reference::end_cap("body")],
        },
        auto: false,
    });
    d.publishes.push(PublishEntry {
        name: PublishName::new("station"),
        target: PublishTarget::Datum {
            datum: DatumRef::Feature {
                feature: fid("station"),
            },
        },
        auto: false,
    });
    d.publishes.push(PublishEntry {
        name: PublishName::new("width"),
        target: PublishTarget::Value {
            input: "width".into(),
        },
        auto: false,
    });
    d
}

fn consumer(file: &str, reference: Reference) -> PartDocument {
    let mut d = block();
    d.path = FilePath::new(file);
    d.regions.push(region(
        "mount",
        TopoKind::Face,
        vec![reference],
        Cardinality::ExactlyOne,
    ));
    d
}

fn catalog_with(
    env: &Env,
    sk: &PartDocument,
    releases: &[(&str, f64)],
) -> (MemoryCatalog, RegenResult) {
    let mut cat = MemoryCatalog::default();
    cat.set_partition(&sk.path, "structures");
    for (rev, width) in releases {
        let mut v = sk.clone();
        set_input(&mut v, "width", mm(*width));
        cat.release(&sk.path, rev, env.regen(&v).publishes);
    }
    let live = env.regen(sk);
    cat.set_live(&sk.path, live.publishes.clone());
    (cat, live)
}

fn mount_outcome(r: &RegenResult) -> &ResolveOutcome {
    &r.regions.get(&"mount".into()).unwrap().references[0].outcome
}

#[test]
fn x01_live_reference_within_partition() {
    let mut env = Env::new();
    let sk = skeleton();
    let (mut cat, _) = catalog_with(&env, &sk, &[]);
    cat.set_partition(&FilePath::new("frames/frame12.part"), "structures");
    env.catalog = Some(cat);
    let d = consumer(
        "frames/frame12.part",
        Reference::published(
            Some("interfaces/skeleton.part"),
            "mount_face",
            XrefContext::Live,
        ),
    );
    let r = env.regen(&d);
    assert!(region_ok(&r, "mount"), "{:#?}", r.diagnostics);
    match mount_outcome(&r) {
        ResolveOutcome::Resolved(res) => {
            assert_eq!(res.class, ResolutionClass::Published);
            match &res.target {
                Resolved::Foreign { target } => {
                    assert_eq!(target.revision, None);
                    assert_eq!(target.entities.len(), 1);
                    assert_eq!(
                        target.entities[0].names,
                        vec![EntityName::generated("body", Role::EndCap, SourceTag::None)]
                    );
                }
                _ => panic!("expected a foreign target"),
            }
        }
        o => panic!("{:?}", o),
    }
}

#[test]
fn x02_live_reference_across_partitions_must_be_pinned() {
    let mut env = Env::new();
    let sk = skeleton();
    let (mut cat, _) = catalog_with(&env, &sk, &[("A", 80.0)]);
    cat.set_partition(&FilePath::new("wing/rib.part"), "wing");
    env.catalog = Some(cat);
    let d = consumer(
        "wing/rib.part",
        Reference::published(
            Some("interfaces/skeleton.part"),
            "mount_face",
            XrefContext::Live,
        ),
    );
    let r = env.regen(&d);
    assert!(!region_ok(&r, "mount"));
    let dg = diag(&r, DiagCode::XrefLiveAcrossPartition);
    assert!(dg
        .repairs
        .iter()
        .any(|x| matches!(x, Repair::PinToRevision { revision, .. } if revision.0 == "A")));
}

#[test]
fn x03_pinned_reference_reads_the_pinned_revision_and_reports_newer() {
    let mut env = Env::new();
    let sk = skeleton();
    let (mut cat, _) = catalog_with(&env, &sk, &[("A", 80.0), ("B", 90.0)]);
    cat.set_partition(&FilePath::new("wing/rib.part"), "wing");
    env.catalog = Some(cat);
    let pin = |rev: &str| XrefContext::Pinned {
        revision: RevisionId::new(rev),
    };
    let d = consumer(
        "wing/rib.part",
        Reference::published(Some("interfaces/skeleton.part"), "mount_face", pin("A")),
    );
    let r = env.regen(&d);
    assert!(region_ok(&r, "mount"));
    match mount_outcome(&r) {
        ResolveOutcome::Resolved(res) => match &res.target {
            Resolved::Foreign { target } => assert_eq!(target.revision, Some(RevisionId::new("A"))),
            _ => panic!(),
        },
        o => panic!("{:?}", o),
    }
    let dg = diag(&r, DiagCode::XrefPinBehind);
    assert!(dg
        .repairs
        .iter()
        .any(|x| matches!(x, Repair::UpdatePin { to, .. } if to.0 == "B")));
    // A missing revision is specific.
    let d2 = consumer(
        "wing/rib.part",
        Reference::published(Some("interfaces/skeleton.part"), "mount_face", pin("Z")),
    );
    assert!(has(&env.regen(&d2), DiagCode::XrefRevisionMissing));
}

#[test]
fn x04_unpublished_and_broken_names() {
    let mut env = Env::new();
    let mut sk = skeleton();
    sk.publishes.push(PublishEntry {
        name: PublishName::new("gone"),
        target: PublishTarget::Entities {
            kind: TopoKind::Face,
            references: vec![side("body", "s1", "nope")],
        },
        auto: false,
    });
    let (mut cat, live) = catalog_with(&env, &sk, &[]);
    // The publishing file reports its own broken entry.
    assert!(has(&live, DiagCode::RefNotGenerated));
    cat.set_partition(&FilePath::new("frames/a.part"), "structures");
    env.catalog = Some(cat);
    let r = env.regen(&consumer(
        "frames/a.part",
        Reference::published(
            Some("interfaces/skeleton.part"),
            "secret",
            XrefContext::Live,
        ),
    ));
    assert!(has(&r, DiagCode::XrefNotPublished));
    let r2 = env.regen(&consumer(
        "frames/a.part",
        Reference::published(Some("interfaces/skeleton.part"), "gone", XrefContext::Live),
    ));
    assert!(has(&r2, DiagCode::XrefPublisherBroken));
    let r3 = env.regen(&consumer(
        "frames/a.part",
        Reference::published(Some("nowhere.part"), "x", XrefContext::Live),
    ));
    assert!(has(&r3, DiagCode::XrefUnknownFile));
}

#[test]
fn x05_publish_survives_publisher_edits() {
    let mut env = Env::new();
    let mut sk = skeleton();
    let (mut cat, _) = catalog_with(&env, &sk, &[]);
    cat.set_partition(&FilePath::new("frames/a.part"), "structures");
    set_input(&mut sk, "thickness", mm(20.0));
    cat.set_live(&sk.path, env.regen(&sk).publishes);
    env.catalog = Some(cat);
    let r = env.regen(&consumer(
        "frames/a.part",
        Reference::published(
            Some("interfaces/skeleton.part"),
            "mount_face",
            XrefContext::Live,
        ),
    ));
    assert!(region_ok(&r, "mount"));
}

#[test]
fn x06_datum_and_value_publishes() {
    let env = Env::new();
    let sk = skeleton();
    let live = env.regen(&sk);
    let st = live
        .publishes
        .get(&PublishName::new("station"))
        .unwrap()
        .as_ref()
        .unwrap();
    match st.datum {
        Some(DatumValue::Plane { frame }) => assert!(approx(frame.origin[2], 0.025)),
        other => panic!("{:?}", other),
    }
    let w = live
        .publishes
        .get(&PublishName::new("width"))
        .unwrap()
        .as_ref()
        .unwrap();
    assert!(approx(w.value.unwrap().si, 0.08));
}

#[test]
fn x07_auto_publish_adds_entry_and_reference() {
    let env = Env::new();
    let mut sk = skeleton();
    let mut cat = MemoryCatalog::default();
    cat.set_partition(&sk.path, "structures");
    cat.set_partition(&FilePath::new("frames/a.part"), "structures");
    let pick = ForeignPick {
        file: sk.path.clone(),
        kind: TopoKind::Face,
        reference: side("body", "s1", "l2"),
    };
    let edit = auto_publish(&sk, &pick, &FilePath::new("frames/a.part"), &cat).unwrap();
    let entry = edit.add.clone().expect("a new entry");
    assert_eq!(entry.name.0, "body_side_l2");
    assert!(entry.auto);
    assert_eq!(
        edit.reference,
        Reference::published(
            Some("interfaces/skeleton.part"),
            "body_side_l2",
            XrefContext::Live
        )
    );
    apply_publish(&mut sk, &edit);
    // The consumer resolves through the new entry.
    let live = env.regen(&sk);
    assert!(live.publishes.get(&entry.name).unwrap().is_ok());
    cat.set_live(&sk.path, live.publishes);
    let mut env2 = Env::new();
    env2.catalog = Some(cat.clone());
    let r = env2.regen(&consumer("frames/a.part", edit.reference.clone()));
    assert!(region_ok(&r, "mount"));
    // Picking the same thing again reuses the entry.
    let again = auto_publish(&sk, &pick, &FilePath::new("frames/a.part"), &cat).unwrap();
    assert!(again.add.is_none());
    assert_eq!(again.reference, edit.reference);
    // A different pick with a clashing name gets a suffix.
    let mut sk2 = sk.clone();
    sk2.publishes.push(PublishEntry {
        name: PublishName::new("body_side_l4"),
        target: PublishTarget::Value {
            input: "width".into(),
        },
        auto: false,
    });
    let e3 = auto_publish(
        &sk2,
        &ForeignPick {
            file: sk.path.clone(),
            kind: TopoKind::Face,
            reference: side("body", "s1", "l4"),
        },
        &FilePath::new("frames/a.part"),
        &cat,
    )
    .unwrap();
    assert_eq!(e3.add.unwrap().name.0, "body_side_l4_2");
}

#[test]
fn x08_auto_publish_across_partitions_pins_or_refuses() {
    let sk = skeleton();
    let mut cat = MemoryCatalog::default();
    cat.set_partition(&sk.path, "structures");
    cat.set_partition(&FilePath::new("wing/rib.part"), "wing");
    let pick = ForeignPick {
        file: sk.path.clone(),
        kind: TopoKind::Face,
        reference: Reference::end_cap("body"),
    };
    let err = auto_publish(&sk, &pick, &FilePath::new("wing/rib.part"), &cat).unwrap_err();
    assert_eq!(
        err,
        AutoPublishError::NoRelease {
            file: sk.path.clone()
        }
    );
    cat.release(&sk.path, "C", Default::default());
    let edit = auto_publish(&sk, &pick, &FilePath::new("wing/rib.part"), &cat).unwrap();
    // The end cap is already published as `mount_face`: reused, and pinned to C.
    assert!(edit.add.is_none());
    assert_eq!(
        edit.reference,
        Reference::published(
            Some("interfaces/skeleton.part"),
            "mount_face",
            XrefContext::Pinned {
                revision: RevisionId::new("C")
            }
        )
    );
}

#[test]
fn x09_closed_files_refuse_auto_publish() {
    let mut sk = skeleton();
    sk.closed = true;
    let mut cat = MemoryCatalog::default();
    cat.set_partition(&sk.path, "s");
    cat.set_partition(&FilePath::new("b.part"), "s");
    let pick = ForeignPick {
        file: sk.path.clone(),
        kind: TopoKind::Face,
        reference: side("body", "s1", "l3"),
    };
    assert_eq!(
        auto_publish(&sk, &pick, &FilePath::new("b.part"), &cat).unwrap_err(),
        AutoPublishError::Closed {
            file: sk.path.clone()
        }
    );
    // Picking something already published still works: it adds nothing.
    let ok = auto_publish(
        &sk,
        &ForeignPick {
            file: sk.path.clone(),
            kind: TopoKind::Face,
            reference: Reference::end_cap("body"),
        },
        &FilePath::new("b.part"),
        &cat,
    );
    assert!(ok.unwrap().add.is_none());
    assert_eq!(
        auto_publish(&sk, &pick, &sk.path.clone(), &cat).unwrap_err(),
        AutoPublishError::SameFile
    );
}

#[test]
fn x10_local_published_names_resolve_in_file() {
    let env = Env::new();
    let mut d = skeleton();
    d.regions.push(region(
        "m",
        TopoKind::Face,
        vec![Reference::published(None, "mount_face", XrefContext::Live)],
        Cardinality::ExactlyOne,
    ));
    d.regions.push(region(
        "bad",
        TopoKind::Face,
        vec![Reference::published(None, "nope", XrefContext::Live)],
        Cardinality::ExactlyOne,
    ));
    let r = env.regen(&d);
    assert!(region_ok(&r, "m"));
    assert!(has(&r, DiagCode::RefUnknownPublish));
}

#[test]
fn x11_no_catalog_is_specific() {
    let env = Env::new();
    let r = env.regen(&consumer(
        "a.part",
        Reference::published(Some("b.part"), "x", XrefContext::Live),
    ));
    assert!(has(&r, DiagCode::XrefNoCatalog));
}

#[test]
fn x12_lock_merge_and_conflicts() {
    let f = FilePath::new("a.part");
    let mut m = SeededMinter { seed: "p".into() };
    let mut a = LockSection::default();
    let mut b = LockSection::default();
    let ua = a.ensure(&ItemPath::part(&f, &"gasket".into()), &mut m);
    let ub = b.ensure(&ItemPath::part(&f, &"gasket".into()), &mut m);
    assert_eq!(ua, ub, "seeded ids are deterministic");
    b.ensure(&ItemPath::occurrence(&f, &["wing.left", "box"]), &mut m);
    let merged = a.merge(&b).unwrap();
    assert_eq!(merged.len(), 2);
    let mut c = LockSection::default();
    c.insert_entry(
        ItemPath::part(&f, &"gasket".into()),
        LockEntry {
            uuid: runmat_cad::ids::ItemUuid(7),
            aliases: Default::default(),
        },
    );
    assert!(matches!(a.merge(&c), Err(LockError::PathConflict { .. })));
    // Renames keep the id and resolve the old path as an alias.
    let mut r = merged.clone();
    r.rename(
        &ItemPath::part(&f, &"gasket".into()),
        &ItemPath::part(&f, &"seal".into()),
    )
    .unwrap();
    let l = r.lookup(&ItemPath::part(&f, &"gasket".into())).unwrap();
    assert!(l.via_alias);
    assert_eq!(l.uuid, ua);
    // One id at two paths after a bad merge is a conflict.
    assert!(matches!(
        r.merge(&merged),
        Err(LockError::UuidConflict { .. })
    ));
    // Ids serialise as 32 hex digits and round-trip.
    let json = serde_json::to_string(&r).unwrap();
    let back: LockSection = serde_json::from_str(&json).unwrap();
    assert_eq!(back, r);
}
