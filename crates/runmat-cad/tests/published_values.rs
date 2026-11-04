//! Published plain values read across files: live within a partition, pinned across
//! partitions, renames through the lock, and the specific diagnostics.

mod common;
use common::*;
use runmat_cad::diagnostics::{DiagCode, Repair, Severity, Subject};
use runmat_cad::ids::{FilePath, InputName, LetName, PublishName, RevisionId};
use runmat_cad::model::*;
use runmat_cad::refs::*;
use runmat_cad::regen::RegenResult;
use runmat_cad::units::QuantityKindId;
use runmat_cad::value::{BinaryOp, InputDecl, LetDecl, ValueExpr};

const SKELETON: &str = "interfaces/skeleton.part";

/// A skeleton publishing its width (an input), a derived pitch (a let) and a face.
fn skeleton(width: f64) -> PartDocument {
    let mut d = block();
    d.path = FilePath::new(SKELETON);
    set_input(&mut d, "width", mm(width));
    d.lets.push(LetDecl {
        name: LetName::new("pitch"),
        kind: Some(QuantityKindId::new("length")),
        value: ValueExpr::binary(BinaryOp::Div, input("width"), ValueExpr::number(4.0)),
    });
    d.publishes.push(PublishEntry {
        name: PublishName::new("width"),
        target: PublishTarget::Value {
            input: "width".into(),
        },
        auto: false,
    });
    d.publishes.push(PublishEntry {
        name: PublishName::new("pitch"),
        target: PublishTarget::Let {
            name: LetName::new("pitch"),
        },
        auto: false,
    });
    d.publishes.push(PublishEntry {
        name: PublishName::new("mount_face"),
        target: PublishTarget::Entities {
            kind: TopoKind::Face,
            references: vec![Reference::end_cap("body")],
        },
        auto: false,
    });
    d
}

/// A consumer whose `width` input reads the skeleton's published width.
fn consumer(file: &str, name: &str, context: XrefContext) -> PartDocument {
    let mut d = block();
    d.path = FilePath::new(file);
    set_input(
        &mut d,
        "width",
        ValueExpr::published(SKELETON, name, context),
    );
    d
}

fn catalog(env: &Env, sk: &PartDocument, releases: &[(&str, f64)]) -> MemoryCatalog {
    let mut cat = MemoryCatalog::default();
    cat.set_partition(&sk.path, "structures");
    for (rev, width) in releases {
        let mut v = sk.clone();
        set_input(&mut v, "width", mm(*width));
        cat.release(&sk.path, rev, env.regen(&v).publishes);
    }
    cat.set_live(&sk.path, env.regen(sk).publishes);
    cat
}

fn width(r: &RegenResult) -> f64 {
    r.values.inputs[&InputName::new("width")].si
}

fn pin(rev: &str) -> XrefContext {
    XrefContext::Pinned {
        revision: RevisionId::new(rev),
    }
}

#[test]
fn v01_live_value_within_partition_follows_the_publisher() {
    let mut env = Env::new();
    let mut cat = catalog(&env, &skeleton(80.0), &[]);
    cat.set_partition(&FilePath::new("frames/frame12.part"), "structures");
    env.catalog = Some(cat);
    let d = consumer("frames/frame12.part", "width", XrefContext::Live);
    let r = env.regen(&d);
    assert_clean(&r);
    assert!(approx(width(&r), 0.080));
    let key = r.final_key.clone();

    // The publisher changes: the consumer reads the new value and regenerates.
    let mut cat = catalog(&env, &skeleton(95.0), &[]);
    cat.set_partition(&FilePath::new("frames/frame12.part"), "structures");
    env.catalog = Some(cat);
    let r = env.regen(&d);
    assert_clean(&r);
    assert!(approx(width(&r), 0.095));
    assert_ne!(
        r.final_key, key,
        "a published value is part of the chained key"
    );
    assert!(r.diagnostics.diagnostics.is_empty(), "{:#?}", r.diagnostics);
}

#[test]
fn v02_derived_values_publish_through_lets() {
    let mut env = Env::new();
    let mut cat = catalog(&env, &skeleton(80.0), &[]);
    cat.set_partition(&FilePath::new("frames/frame12.part"), "structures");
    env.catalog = Some(cat);
    let mut d = consumer("frames/frame12.part", "pitch", XrefContext::Live);
    // A let in the consumer reads the published let, too.
    d.lets.push(LetDecl {
        name: LetName::new("twice"),
        kind: Some(QuantityKindId::new("length")),
        value: ValueExpr::binary(
            BinaryOp::Mul,
            ValueExpr::published(SKELETON, "pitch", XrefContext::Live),
            ValueExpr::number(2.0),
        ),
    });
    let r = env.regen(&d);
    assert_clean(&r);
    assert!(approx(width(&r), 0.020));
    assert!(approx(r.values.lets[&LetName::new("twice")].si, 0.040));
}

#[test]
fn v03_live_value_across_partitions_must_be_pinned_and_cascades() {
    let mut env = Env::new();
    let mut cat = catalog(&env, &skeleton(80.0), &[("A", 80.0)]);
    cat.set_partition(&FilePath::new("wing/rib.part"), "wing");
    env.catalog = Some(cat);
    let r = env.regen(&consumer("wing/rib.part", "width", XrefContext::Live));
    let dg = diag(&r, DiagCode::XrefLiveAcrossPartition);
    assert_eq!(dg.severity, Severity::Error);
    assert_eq!(
        dg.site.subject,
        Subject::Value {
            name: "input.width".into()
        }
    );
    assert_eq!(
        dg.message,
        "published value `width`: `interfaces/skeleton.part` is in another partition; references to it must be pinned to a released revision"
    );
    assert!(dg.repairs.iter().any(|x| matches!(
        x,
        Repair::PinToRevision { revision, file, .. }
            if revision.0 == "A" && file.0 == SKELETON
    )));
    // The sketch that reads `width` is a cascade of the failed value, not a second root.
    assert!(has(&r, DiagCode::RegenCascade));
    assert_eq!(r.diagnostics.roots().count(), 1, "{:#?}", r.diagnostics);
}

#[test]
fn v04_pinned_value_reads_the_pinned_revision_and_reports_newer() {
    let mut env = Env::new();
    let mut cat = catalog(&env, &skeleton(100.0), &[("A", 80.0), ("B", 90.0)]);
    cat.set_partition(&FilePath::new("wing/rib.part"), "wing");
    env.catalog = Some(cat);
    let r = env.regen(&consumer("wing/rib.part", "width", pin("A")));
    assert_clean(&r);
    assert!(approx(width(&r), 0.080), "the pin, not the live 100 mm");
    let dg = diag(&r, DiagCode::XrefPinBehind);
    assert_eq!(dg.severity, Severity::Info);
    assert_eq!(
        dg.message,
        "published value `width` is pinned to `interfaces/skeleton.part` revision A; B is released"
    );
    assert!(dg.repairs.iter().any(|x| matches!(
        x,
        Repair::UpdatePin { from, to, .. } if from.0 == "A" && to.0 == "B"
    )));
    // Pinned to the latest: no note.
    let r = env.regen(&consumer("wing/rib.part", "width", pin("B")));
    assert!(approx(width(&r), 0.090));
    assert!(!has(&r, DiagCode::XrefPinBehind));
    // A missing revision is specific.
    let r = env.regen(&consumer("wing/rib.part", "width", pin("Z")));
    let dg = diag(&r, DiagCode::XrefRevisionMissing);
    assert!(dg
        .repairs
        .iter()
        .any(|x| matches!(x, Repair::ReleaseFile { file } if file.0 == SKELETON)));
}

#[test]
fn v05_renamed_values_resolve_through_the_lock_and_ask_for_a_rewrite() {
    let mut env = Env::new();
    let mut sk = skeleton(80.0);
    let mut lock = LockSection::default();
    let mut m = SeededMinter { seed: "v".into() };
    rename_publish(
        &mut sk,
        &PublishName::new("width"),
        &PublishName::new("frame_width"),
        &mut lock,
        &mut m,
    )
    .unwrap();
    let mut cat = catalog(&env, &sk, &[]);
    cat.set_partition(&FilePath::new("frames/frame12.part"), "structures");
    env.catalog = Some(cat);
    let d = consumer("frames/frame12.part", "width", XrefContext::Live);
    let r = env.regen(&d);
    let dg = diag(&r, DiagCode::XrefNotPublished);
    assert_eq!(
        dg.message,
        "published value `width`: `interfaces/skeleton.part` publishes no `width`"
    );
    env.lock = Some(lock);
    let r = env.regen(&d);
    assert_clean(&r);
    assert!(approx(width(&r), 0.080));
    let n = diag(&r, DiagCode::XrefViaAlias);
    assert_eq!(n.severity, Severity::Warning);
    assert_eq!(
        n.site.subject,
        Subject::Value {
            name: "input.width".into()
        }
    );
    match &n.repairs[..] {
        [Repair::RewriteReference {
            reference:
                Reference::Published {
                    file,
                    name,
                    context,
                },
            ..
        }] => {
            assert_eq!(name.0, "frame_width");
            assert_eq!(file.as_ref().unwrap().0, SKELETON);
            assert_eq!(context, &XrefContext::Live);
        }
        other => panic!("{:?}", other),
    }
}

#[test]
fn v06_file_moves_resolve_through_the_lock() {
    let mut env = Env::new();
    let mut sk = skeleton(80.0);
    let mut lock = LockSection::default();
    let mut m = SeededMinter { seed: "v".into() };
    lock.ensure(
        &ItemPath::publish(&sk.path, &PublishName::new("width")),
        &mut m,
    );
    let moved = FilePath::new("interfaces/fuselage/skeleton.part");
    lock.rename_file(&sk.path.clone(), &moved, &mut m).unwrap();
    sk.path = moved;
    let mut cat = catalog(&env, &sk, &[]);
    cat.set_partition(&FilePath::new("frames/frame12.part"), "structures");
    env.catalog = Some(cat);
    env.lock = Some(lock);
    let r = env.regen(&consumer("frames/frame12.part", "width", XrefContext::Live));
    assert_clean(&r);
    assert!(approx(width(&r), 0.080));
    let n = diag(&r, DiagCode::XrefViaAlias);
    assert_eq!(
        n.message,
        "published value `width` from `interfaces/skeleton.part` was renamed to `width` in `interfaces/fuselage/skeleton.part`; update the reference"
    );
}

#[test]
fn v07_geometry_is_not_a_value_and_units_are_checked() {
    let mut env = Env::new();
    let mut cat = catalog(&env, &skeleton(80.0), &[]);
    cat.set_partition(&FilePath::new("frames/frame12.part"), "structures");
    env.catalog = Some(cat);
    let r = env.regen(&consumer(
        "frames/frame12.part",
        "mount_face",
        XrefContext::Live,
    ));
    let dg = diag(&r, DiagCode::XrefNotAValue);
    assert_eq!(
        dg.message,
        "`interfaces/skeleton.part` publishes `mount_face` as geometry, not as a value"
    );
    // A length read into an angle input is a unit mismatch, never a conversion.
    let mut d = block();
    d.path = FilePath::new("frames/frame12.part");
    d.inputs.push(InputDecl::new(
        "sweep",
        "angle",
        ValueExpr::published(SKELETON, "width", XrefContext::Live),
    ));
    let r = env.regen(&d);
    let dg = diag(&r, DiagCode::ValueFailed);
    assert_eq!(
        dg.site.subject,
        Subject::Value {
            name: "input.sweep".into()
        }
    );
    assert!(dg.message.starts_with("unit mismatch"), "{}", dg.message);
}

#[test]
fn v08_feature_parameters_read_published_values_and_no_catalog_is_specific() {
    let mut env = Env::new();
    let mut cat = catalog(&env, &skeleton(80.0), &[]);
    cat.set_partition(&FilePath::new("frames/frame12.part"), "structures");
    let mut d = block();
    d.path = FilePath::new("frames/frame12.part");
    let body = d.feature_mut("body").unwrap();
    if let FeatureKind::Extrude(e) = &mut body.kind {
        e.extent = Extent::Blind {
            distance: ValueExpr::published(SKELETON, "pitch", XrefContext::Live),
        };
    }
    let r = env.regen(&d);
    let dg = diag(&r, DiagCode::XrefNoCatalog);
    assert_eq!(dg.site.key(), "feature.body#distance");
    env.catalog = Some(cat);
    let r = env.regen(&d);
    assert_clean(&r);
    // The extrude is 20 mm (80 / 4) tall: its top cap sits at z = 0.02.
    let top = r
        .state
        .bodies
        .iter()
        .flat_map(|b| b.topology.faces.iter())
        .filter_map(|f| match f.geom.surface {
            SurfaceKind::Plane { normal } if normal[2] > 0.5 => Some(f.geom.centroid[2]),
            _ => None,
        })
        .fold(f64::MIN, f64::max);
    assert!(approx(top, 0.020), "{}", top);
}

// ================================================== dependency graph and change impact

#[test]
fn published_reads_enter_the_dependency_graph_and_change_impact() {
    use runmat_cad::config::{affected_by_published, Because};
    use runmat_cad::ids::ConfigurationName;
    use runmat_cad::regen::DependencyGraph;
    use runmat_cad::value::ValueXref;
    let mut d = consumer("frames/a.part", "width", XrefContext::Live);
    // A hole pitch read straight from the skeleton, through a let.
    d.lets.push(LetDecl {
        name: LetName::new("pitch"),
        kind: Some(QuantityKindId::new("length")),
        value: ValueExpr::published(SKELETON, "pitch", XrefContext::Live),
    });
    d.features.push(Feature::new(
        "deep",
        FeatureKind::Extrude(Extrude {
            profile: ProfileRef::sketch("s1"),
            extent: Extent::Blind {
                distance: ValueExpr::let_("pitch"),
            },
            reverse: true,
            mode: BodyMode::add(),
            from: Default::default(),
            second: None,
            draft: None,
            thin: None,
        }),
    ));
    // `wide` overrides the published width; `lite` suppresses the let reader.
    let mut wide = Configuration {
        name: ConfigurationName::new("wide"),
        ..Default::default()
    };
    wide.overrides.insert(InputName::new("width"), mm(120.0));
    let mut lite = Configuration {
        name: ConfigurationName::new("lite"),
        ..Default::default()
    };
    lite.suppress.insert(fid("deep"));
    lite.overrides.insert(InputName::new("width"), mm(90.0));
    d.configurations.push(wide);
    d.configurations.push(lite);

    let g = DependencyGraph::analyse(&d);
    let x_width = ValueXref {
        file: FilePath::new(SKELETON),
        name: PublishName::new("width"),
        context: XrefContext::Live,
    };
    let x_pitch = ValueXref {
        file: FilePath::new(SKELETON),
        name: PublishName::new("pitch"),
        context: XrefContext::Live,
    };
    assert!(g.input_published[&InputName::new("width")].contains(&x_width));
    assert!(g.feature_published[&fid("deep")].contains(&x_pitch));
    // The published width reaches the sketch through the input, and everything after it.
    let a = g.affected_by_published(std::slice::from_ref(&x_width));
    assert!(a.features.contains(&fid("s1")) && a.features.contains(&fid("body")));
    assert!(a.parts.contains(&"bracket".into()));

    // Width moved upstream: the default configuration changes; `wide` and `lite` override it.
    let imp = affected_by_published(&d, std::slice::from_ref(&x_width));
    assert!(imp.affected[&None].contains(&Because::Published {
        value: x_width.clone()
    }));
    assert!(imp.unaffected.contains(&ConfigurationName::new("wide")));
    assert!(imp.unaffected.contains(&ConfigurationName::new("lite")));
    // Pitch moved: `wide` reads it through `deep`; `lite` suppresses `deep`.
    let imp = affected_by_published(&d, std::slice::from_ref(&x_pitch));
    let wide_why = &imp.affected[&Some(ConfigurationName::new("wide"))];
    assert!(wide_why.contains(&Because::Feature {
        feature: fid("deep")
    }));
    assert!(imp.unaffected.contains(&ConfigurationName::new("lite")));
}
