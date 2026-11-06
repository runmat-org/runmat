//! Assemblies: composition, variants and their proof, overrides, typed configurations, mates,
//! flexible sub-assemblies, patterns and mirrors, fasteners, BOM, where-used, interference,
//! working sets, the index and summary cards, identity through renames, and the scale gate.

use runmat_cad::assembly::bom::*;
use runmat_cad::assembly::fasteners::*;
use runmat_cad::assembly::index::*;
use runmat_cad::assembly::interference::*;
use runmat_cad::assembly::options::*;
use runmat_cad::assembly::representation::*;
use runmat_cad::assembly::resolve::*;
use runmat_cad::assembly::source::*;
use runmat_cad::assembly::synth::{generate, SynthSpec};
use runmat_cad::assembly::validate::validate_product;
use runmat_cad::assembly::*;
use runmat_cad::diagnostics::{DiagCode, Severity};
use runmat_cad::ids::{ConfigurationName, FilePath, InputName, PublishName};
use runmat_cad::model::datum::Frame;
use runmat_cad::refs::lock::{LockSection, SeededMinter};
use runmat_cad::refs::{InstanceKey, XrefContext};
use runmat_cad::regen::kernel::Transform;
use runmat_cad::units::{Dimension, QuantityKindId, StubUnits};
use runmat_cad::value::{NoDynamicValues, Quantity, ValueExpr};
use std::collections::BTreeMap;
use std::time::Instant;

fn mm(v: f64) -> ValueExpr {
    ValueExpr::mm(v)
}

fn at(x: f64, y: f64, z: f64) -> Placement {
    Placement::Offset {
        offset: OffsetExpr::translation([mm(x), mm(y), mm(z)]),
    }
}

fn add_part(
    src: &mut MemorySource,
    file: &str,
    pn: &str,
    bbox: Aabb,
    publishes: Vec<(&str, PublishedGeometry)>,
) {
    let mut s = DefinitionSummary::new(DefinitionRef::part(file, None));
    s.part_number = Some(pn.into());
    src.add_part(
        s,
        ConfiguredSummary {
            bbox: Some(bbox),
            publishes: publishes
                .into_iter()
                .map(|(k, v)| (PublishName::new(k), v))
                .collect(),
            ..Default::default()
        },
    );
}

fn unit_box() -> Aabb {
    Aabb::new([0.0; 3], [0.01, 0.01, 0.01])
}

fn frame_at(x: f64, y: f64, z: f64) -> PublishedGeometry {
    PublishedGeometry::Frame {
        frame: Frame {
            origin: [x, y, z],
            ..Frame::WORLD
        },
    }
}

struct Env {
    units: StubUnits,
    dynamic: NoDynamicValues,
}

impl Env {
    fn new() -> Self {
        Env {
            units: StubUnits,
            dynamic: NoDynamicValues,
        }
    }
}

fn run(src: &MemorySource, top: &str, req: &ResolveRequest) -> ResolvedAssembly {
    run_with(src, top, req, None, None, None, None)
}

fn run_with(
    src: &MemorySource,
    top: &str,
    req: &ResolveRequest,
    lock: Option<&LockSection>,
    catalog: Option<&dyn FastenerCatalog>,
    tables: Option<&dyn TableSource>,
    gauge: Option<&dyn StackGauge>,
) -> ResolvedAssembly {
    let e = Env::new();
    let ctx = ResolveContext {
        source: src,
        units: &e.units,
        dynamic: &e.dynamic,
        lock,
        catalog,
        tables,
        gauge,
    };
    resolve(&FilePath::new(top), req, &ctx)
}

fn codes(r: &ResolvedAssembly) -> Vec<DiagCode> {
    r.diagnostics.diagnostics.iter().map(|d| d.code).collect()
}

fn errors(r: &ResolvedAssembly) -> Vec<String> {
    r.diagnostics
        .errors()
        .map(|d| format!("{} {}", d.code, d.message))
        .collect()
}

fn close(a: [f64; 3], b: [f64; 3]) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() < 1e-9)
}

fn values(pairs: &[(&str, AxisValue)]) -> VariantSelection {
    VariantSelection::Values {
        values: pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    }
}

// ------------------------------------------------------------------------------ structure

/// A bracket on a plate, a sub-assembly of two bolts, typed configurations.
fn bracket_project() -> MemorySource {
    let mut src = MemorySource::default();
    add_part(
        &mut src,
        "plate.part",
        "PL-1",
        Aabb::new([0.0; 3], [0.2, 0.1, 0.01]),
        vec![("top", frame_at(0.1, 0.05, 0.01))],
    );
    let mut s = DefinitionSummary::new(DefinitionRef::part("bracket.part", None));
    s.part_number = Some("BR-1".into());
    s.configurations = vec![ConfigurationName::new("light")];
    s.inputs = vec![(InputName::new("thickness"), QuantityKindId::new("length"))];
    src.add_part(
        s,
        ConfiguredSummary {
            bbox: Some(Aabb::new([0.0; 3], [0.08, 0.05, 0.008])),
            publishes: [(PublishName::new("base"), frame_at(0.04, 0.025, 0.0))]
                .into_iter()
                .collect(),
            ..Default::default()
        },
    );
    let light = ConfiguredSummary {
        bbox: Some(Aabb::new([0.0; 3], [0.08, 0.05, 0.005])),
        part_number: Some("BR-1-L".into()),
        publishes: [(PublishName::new("base"), frame_at(0.04, 0.025, 0.0))]
            .into_iter()
            .collect(),
        ..Default::default()
    };
    src.add_configuration(
        &DefinitionRef::part("bracket.part", None),
        &ConfigKey::named("light"),
        light,
    );
    src.set_configure(&DefinitionRef::part("bracket.part", None), |_, key| {
        let t = key
            .inputs
            .get(&InputName::new("thickness"))
            .copied()
            .unwrap_or(0.008);
        Ok(ConfiguredSummary {
            bbox: Some(Aabb::new([0.0; 3], [0.08, 0.05, t])),
            publishes: [(PublishName::new("base"), frame_at(0.04, 0.025, 0.0))]
                .into_iter()
                .collect(),
            ..Default::default()
        })
    });
    add_part(
        &mut src,
        "bolt.part",
        "BT-6",
        Aabb::new([-0.003, -0.003, 0.0], [0.003, 0.003, 0.02]),
        vec![],
    );

    let mut bolts = AssemblyDocument::new("bolts.assembly", "bolts");
    bolts.part_number = Some("BA-1".into());
    let mut b1 = Occurrence::new("b1", DefinitionRef::part("bolt.part", None));
    b1.placement = at(0.0, 0.0, 0.0);
    let mut b2 = Occurrence::new("b2", DefinitionRef::part("bolt.part", None));
    b2.placement = at(30.0, 0.0, 0.0);
    bolts.occurrences = vec![b1, b2];
    src.add_assembly(bolts);

    let mut top = AssemblyDocument::new("rig.assembly", "rig");
    top.part_number = Some("RIG-1".into());
    top.occurrences.push(Occurrence::new(
        "plate",
        DefinitionRef::part("plate.part", None),
    ));
    let mut br = Occurrence::new("bracket", DefinitionRef::part("bracket.part", None));
    br.configuration = ConfigurationSelection::Named {
        name: "light".into(),
    };
    br.placement = Placement::Attach {
        own: AnchorRef::Published {
            name: "base".into(),
        },
        to: FrameRef::Connector {
            connector: "plate_top".into(),
        },
        offset: None,
        flip: false,
    };
    top.occurrences.push(br);
    let mut bo = Occurrence::new("bolts", DefinitionRef::assembly("bolts.assembly"));
    bo.placement = at(50.0, 20.0, 10.0);
    top.occurrences.push(bo);
    top.connectors.push(MateConnector {
        name: "plate_top".into(),
        occurrence: RelPath::new(&["plate"]),
        anchor: AnchorRef::Published { name: "top".into() },
        offset: None,
        flip: false,
    });
    src.add_assembly(top);
    src
}

#[test]
fn a01_composes_places_and_attaches() {
    let src = bracket_project();
    let r = run(&src, "rig.assembly", &ResolveRequest::default());
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    assert_eq!(r.nodes.len(), 1 + 3 + 2);
    let br = r.find_display("bracket").unwrap();
    // The bracket's base (0.04, 0.025, 0) sits on the plate's top (0.1, 0.05, 0.01).
    let base = r.node(br).world.apply([0.04, 0.025, 0.0]);
    assert!(close(base, [0.1, 0.05, 0.01]), "{:?}", base);
    let b2 = r.find_display("bolts/b2").unwrap();
    assert!(close(r.node(b2).world.translation, [0.08, 0.02, 0.01]));
    assert_eq!(r.node(br).configuration.label(), "light");
    // Prototypes: two bolts share one.
    let bolt_protos: Vec<_> = r
        .prototypes
        .iter()
        .filter(|p| p.definition.file.0 == "bolt.part")
        .collect();
    assert_eq!(bolt_protos.len(), 1);
    assert_eq!(bolt_protos[0].instances, 2);
    assert!(
        codes(&r).contains(&DiagCode::AsmUnlocked),
        "unlocked items are reported once per file"
    );
}

#[test]
fn a02_identity_is_stable_through_renames_and_reorders() {
    let mut src = bracket_project();
    let mut lock = LockSection::default();
    let mut m = SeededMinter { seed: "t".into() };
    for d in src.assemblies.values() {
        lock_assembly(d, &mut lock, &mut m);
    }
    let r1 = run_with(
        &src,
        "rig.assembly",
        &ResolveRequest::default(),
        Some(&lock),
        None,
        None,
        None,
    );
    assert!(!codes(&r1).contains(&DiagCode::AsmUnlocked));
    let u_bolt = r1.node(r1.find_display("bolts/b2").unwrap()).uuid;
    let u_br = r1.node(r1.find_display("bracket").unwrap()).uuid;
    // Rename `bracket` → `support` (the lock keeps the id) and reorder the children.
    let mut top = (*src.assemblies[&(FilePath::new("rig.assembly"), None)]).clone();
    top.occurrences
        .iter_mut()
        .find(|o| o.name == "bracket")
        .unwrap()
        .name = "support".into();
    top.occurrences.reverse();
    src.add_assembly(top);
    lock.rename(
        &item_path(&FilePath::new("rig.assembly"), "occurrence", "bracket"),
        &item_path(&FilePath::new("rig.assembly"), "occurrence", "support"),
    )
    .unwrap();
    let r2 = run_with(
        &src,
        "rig.assembly",
        &ResolveRequest::default(),
        Some(&lock),
        None,
        None,
        None,
    );
    assert_eq!(
        r2.node(r2.find_display("support").unwrap()).uuid,
        u_br,
        "renamed occurrences keep their id"
    );
    assert_eq!(
        r2.node(r2.find_display("bolts/b2").unwrap()).uuid,
        u_bolt,
        "reorders do not change ids"
    );
    let l = lock
        .lookup(&item_path(
            &FilePath::new("rig.assembly"),
            "occurrence",
            "bracket",
        ))
        .unwrap();
    assert!(l.via_alias, "the old path still resolves through the alias");
}

#[test]
fn a03_typed_configurations_are_checked() {
    let mut src = bracket_project();
    let mut top = (*src.assemblies[&(FilePath::new("rig.assembly"), None)]).clone();
    let br = top
        .occurrences
        .iter_mut()
        .find(|o| o.name == "bracket")
        .unwrap();
    br.configuration = ConfigurationSelection::Inputs {
        base: None,
        values: [(InputName::new("thickness"), mm(3.0))]
            .into_iter()
            .collect(),
    };
    src.add_assembly(top.clone());
    let r = run(&src, "rig.assembly", &ResolveRequest::default());
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    let n = r.node(r.find_display("bracket").unwrap());
    assert!((n.configuration.inputs[&InputName::new("thickness")] - 0.003).abs() < 1e-12);
    // Wrong dimension.
    let br = top
        .occurrences
        .iter_mut()
        .find(|o| o.name == "bracket")
        .unwrap();
    br.configuration = ConfigurationSelection::Inputs {
        base: None,
        values: [(InputName::new("thickness"), ValueExpr::deg(3.0))]
            .into_iter()
            .collect(),
    };
    src.add_assembly(top.clone());
    let r = run(&src, "rig.assembly", &ResolveRequest::default());
    assert!(codes(&r).contains(&DiagCode::AsmInputType));
    // Unknown configuration name.
    let br = top
        .occurrences
        .iter_mut()
        .find(|o| o.name == "bracket")
        .unwrap();
    br.configuration = ConfigurationSelection::Named {
        name: "heavy".into(),
    };
    src.add_assembly(top);
    let r = run(&src, "rig.assembly", &ResolveRequest::default());
    assert!(codes(&r).contains(&DiagCode::AsmUnknownConfiguration));
}

// ------------------------------------------------------------------------------ variants

fn variant_project() -> MemorySource {
    let mut src = MemorySource::default();
    for f in ["engine-a.part", "engine-b.part", "door.part", "pylon.part"] {
        add_part(
            &mut src,
            f,
            &f.to_uppercase(),
            unit_box(),
            vec![("mount", frame_at(0.0, 0.0, 0.0))],
        );
    }
    let mut top = AssemblyDocument::new("ac.assembly", "ac");
    top.options = Some(OptionsBlock {
        axes: vec![
            AxisDecl {
                name: "engine".into(),
                domain: AxisDomain::Enum {
                    values: vec!["a".into(), "b".into(), "c".into()],
                },
                default: Some(AxisValue::Enum("a".into())),
            },
            AxisDecl {
                name: "msn".into(),
                domain: AxisDomain::Unit { min: 1, max: 500 },
                default: Some(AxisValue::Int(1)),
            },
            AxisDecl {
                name: "delivery".into(),
                domain: AxisDomain::Date {
                    min: CivilDate::new(2027, 1, 1),
                    max: CivilDate::new(2035, 12, 31),
                },
                default: Some(AxisValue::Date(CivilDate::new(2027, 6, 1))),
            },
        ],
        rules: vec![OptionRule {
            name: "no_c_before_100".into(),
            condition: Condition::requires(
                Condition::is_enum("engine", "c"),
                Condition::range("msn", Some(100), None),
            ),
        }],
        groups: vec![],
        configurations: vec![],
    });
    let mut a = Occurrence::new("engine", DefinitionRef::part("engine-a.part", None));
    a.when = Some(Condition::is_enum("engine", "a"));
    a.required = true;
    let mut b = Occurrence::new("engine", DefinitionRef::part("engine-b.part", None));
    b.when = Some(Condition::In {
        axis: "engine".into(),
        values: vec![AxisValue::Enum("b".into()), AxisValue::Enum("c".into())],
    });
    b.required = true;
    top.occurrences = vec![a, b];
    let mut door = Occurrence::new("door", DefinitionRef::part("door.part", None));
    door.when = Some(Condition::Range {
        axis: "delivery".into(),
        min: Some(AxisValue::Date(CivilDate::new(2029, 1, 1))),
        max: None,
    });
    top.occurrences.push(door);
    top.occurrences.push(Occurrence::new(
        "pylon",
        DefinitionRef::part("pylon.part", None),
    ));
    src.add_assembly(top);
    src
}

#[test]
fn a04_alternatives_resolve_per_configuration_and_the_proof_passes() {
    let src = variant_project();
    let v = validate_product(&FilePath::new("ac.assembly"), None, &src);
    assert!(v.ok(), "{:?}", v.diagnostics.diagnostics);
    assert!(v.stats.pair_checks >= 1);
    let r = run(
        &src,
        "ac.assembly",
        &ResolveRequest {
            variant: values(&[
                ("engine", AxisValue::Enum("c".into())),
                ("msn", AxisValue::Int(150)),
            ]),
            ..Default::default()
        },
    );
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    assert_eq!(
        r.node(r.find_display("engine").unwrap()).definition.file.0,
        "engine-b.part"
    );
    assert!(
        r.find_display("door").is_none(),
        "date effectivity: no door before 2029"
    );
    let r = run(
        &src,
        "ac.assembly",
        &ResolveRequest {
            variant: values(&[("delivery", AxisValue::Date(CivilDate::new(2030, 3, 1)))]),
            ..Default::default()
        },
    );
    assert!(r.find_display("door").is_some());
    // An illegal point (engine c before MSN 100) is diagnosed with the broken rule.
    let r = run(
        &src,
        "ac.assembly",
        &ResolveRequest {
            variant: values(&[
                ("engine", AxisValue::Enum("c".into())),
                ("msn", AxisValue::Int(10)),
            ]),
            ..Default::default()
        },
    );
    let d = r.diagnostics.with_code(DiagCode::AsmIllegalConfiguration);
    assert_eq!(d.len(), 1);
    assert!(d[0].message.contains("no_c_before_100"));
}

#[test]
fn a05_the_validator_finds_overlaps_gaps_and_dangling_mates_with_witnesses() {
    let mut src = variant_project();
    let mut top = (*src.assemblies[&(FilePath::new("ac.assembly"), None)]).clone();
    // b now also claims `a` for MSN ≥ 300: overlap with alternative 0.
    top.occurrences[1].when = Some(Condition::any(vec![
        Condition::In {
            axis: "engine".into(),
            values: vec![AxisValue::Enum("b".into())],
        },
        Condition::all(vec![
            Condition::is_enum("engine", "a"),
            Condition::range("msn", Some(300), None),
        ]),
    ]));
    // Engine c is no longer covered (gap) — the required name has no alternative there.
    // A mate on the door is active everywhere, but the door exists only from 2029.
    top.connectors.push(MateConnector {
        name: "door_c".into(),
        occurrence: RelPath::new(&["door"]),
        anchor: AnchorRef::Origin,
        offset: None,
        flip: false,
    });
    top.connectors.push(MateConnector {
        name: "pylon_c".into(),
        occurrence: RelPath::new(&["pylon"]),
        anchor: AnchorRef::Origin,
        offset: None,
        flip: false,
    });
    top.mates.push(MateDef {
        name: "door_to_pylon".into(),
        kind: MateKind::Fasten,
        a: "door_c".into(),
        b: "pylon_c".into(),
        when: None,
        suppressed: false,
    });
    src.add_assembly(top.clone());
    let v = validate_product(&FilePath::new("ac.assembly"), None, &src);
    let amb = v.diagnostics.with_code(DiagCode::AsmAmbiguousVariant);
    assert_eq!(amb.len(), 1, "{:?}", v.diagnostics.diagnostics);
    assert!(
        amb[0].message.contains("engine = a") && amb[0].message.contains("msn = 300"),
        "{}",
        amb[0].message
    );
    let gap = v.diagnostics.with_code(DiagCode::AsmIncompleteVariant);
    assert_eq!(gap.len(), 1);
    assert!(gap[0].message.contains("engine = c"), "{}", gap[0].message);
    let dangling = v.diagnostics.with_code(DiagCode::AsmAbsentInConfiguration);
    assert_eq!(dangling.len(), 1, "{:?}", v.diagnostics.diagnostics);
    // The fix: restrict the mate to where the door exists.
    top.mates[0].when = top
        .occurrences
        .iter()
        .find(|o| o.name == "door")
        .unwrap()
        .when
        .clone();
    top.occurrences[1].when = Some(Condition::In {
        axis: "engine".into(),
        values: vec![AxisValue::Enum("b".into()), AxisValue::Enum("c".into())],
    });
    src.add_assembly(top);
    let v = validate_product(&FilePath::new("ac.assembly"), None, &src);
    assert!(v.ok(), "{:?}", v.diagnostics.diagnostics);
    // At resolve, both alternatives active at once is diagnosed too (no silent pick).
    let mut top = (*src.assemblies[&(FilePath::new("ac.assembly"), None)]).clone();
    top.occurrences[1].when = None;
    src.add_assembly(top);
    let r = run(&src, "ac.assembly", &ResolveRequest::default());
    assert!(codes(&r).contains(&DiagCode::AsmDuplicateName));
}

#[test]
fn a06_change_groups_and_classification() {
    let block = OptionsBlock {
        axes: vec![AxisDecl {
            name: "msn".into(),
            domain: AxisDomain::Unit { min: 1, max: 100 },
            default: None,
        }],
        rules: vec![],
        groups: vec![ChangeGroup {
            name: "ECO_7".into(),
            effectivity: Condition::range("msn", Some(41), None),
            description: None,
        }],
        configurations: vec![],
    };
    let s = OptionSpace::new(&block);
    let in_40s = s
        .compile(&Condition::range("msn", Some(1), Some(40)))
        .unwrap();
    assert_eq!(
        s.classify(
            &Condition::Group {
                name: "ECO_7".into()
            },
            &in_40s
        )
        .unwrap(),
        options::Coverage::Never
    );
    let wide = s
        .compile(&Condition::range("msn", Some(30), Some(60)))
        .unwrap();
    match s
        .classify(
            &Condition::Group {
                name: "ECO_7".into(),
            },
            &wide,
        )
        .unwrap()
    {
        options::Coverage::Varies { active, inactive } => {
            assert_eq!(active, vec![41]);
            assert_eq!(inactive, vec![30]);
        }
        other => panic!("{:?}", other),
    }
}

// ------------------------------------------------------------------------------ overrides

fn deep_project() -> MemorySource {
    let mut src = bracket_project();
    let mut mid = AssemblyDocument::new("mid.assembly", "mid");
    let mut r = Occurrence::new("rig", DefinitionRef::assembly("rig.assembly"));
    r.overrides.push(Override {
        target: RelPath::new(&["bolts", "b1"]),
        when: None,
        set: OverrideKind::Material {
            material: "material.steel".into(),
        },
    });
    mid.occurrences.push(r);
    src.add_assembly(mid);
    let mut top = AssemblyDocument::new("top.assembly", "top");
    let mut m = Occurrence::new("mid", DefinitionRef::assembly("mid.assembly"));
    m.overrides.push(Override {
        target: RelPath::new(&["rig", "bolts", "b1"]),
        when: None,
        set: OverrideKind::Material {
            material: "material.titanium".into(),
        },
    });
    m.overrides.push(Override {
        target: RelPath::new(&["rig", "bracket"]),
        when: None,
        set: OverrideKind::Configuration {
            select: ConfigurationSelection::Default,
        },
    });
    m.overrides.push(Override {
        target: RelPath::new(&["rig", "bolts", "b2"]),
        when: None,
        set: OverrideKind::Suppress { suppressed: true },
    });
    m.overrides.push(Override {
        target: RelPath::new(&["rig", "bolt", "b9"]),
        when: None,
        set: OverrideKind::Suppress { suppressed: true },
    });
    top.occurrences.push(m);
    src.add_assembly(top);
    src
}

#[test]
fn a07_typed_overrides_travel_down_outer_wins_and_missing_targets_are_diagnosed() {
    let src = deep_project();
    let r = run(&src, "top.assembly", &ResolveRequest::default());
    let b1 = r.node(r.find_display("mid/rig/bolts/b1").unwrap());
    assert_eq!(
        b1.material.as_ref().unwrap().0,
        "material.titanium",
        "the outer override wins"
    );
    assert!(codes(&r).contains(&DiagCode::AsmOverrideShadowed));
    assert!(
        r.find_display("mid/rig/bolts/b2").is_none(),
        "suppressed by override"
    );
    assert!(r.suppressed.iter().any(|p| p == "mid/rig/bolts/b2"));
    assert_eq!(
        r.node(r.find_display("mid/rig/bracket").unwrap())
            .configuration
            .label(),
        "default"
    );
    let missing = r.diagnostics.with_code(DiagCode::AsmOverrideTargetMissing);
    assert_eq!(missing.len(), 1);
    assert!(
        missing[0].message.contains("bolt/b9"),
        "{}",
        missing[0].message
    );
    assert!(matches!(
        missing[0].repairs[0],
        runmat_cad::diagnostics::Repair::RetargetOverride { .. }
    ));
}

// ------------------------------------------------------------------------------ mates

fn hinge_project(limit: Option<(f64, f64)>, drive: Option<f64>) -> MemorySource {
    let mut src = MemorySource::default();
    add_part(
        &mut src,
        "base.part",
        "B",
        unit_box(),
        vec![("pin", frame_at(0.0, 0.0, 0.05))],
    );
    add_part(
        &mut src,
        "arm.part",
        "A",
        unit_box(),
        vec![
            ("eye", frame_at(0.0, 0.0, 0.0)),
            ("tip", frame_at(0.2, 0.0, 0.0)),
        ],
    );
    let mut top = AssemblyDocument::new("hinge.assembly", "hinge");
    top.occurrences.push(Occurrence::new(
        "base",
        DefinitionRef::part("base.part", None),
    ));
    let mut arm = Occurrence::new("arm", DefinitionRef::part("arm.part", None));
    arm.fixed = false;
    arm.placement = Placement::Offset {
        offset: OffsetExpr {
            translation: [mm(5.0), mm(3.0), mm(40.0)],
            rotation: RotationExpr::AxisAngle {
                axis: [0.0, 0.0, 1.0],
                angle: ValueExpr::deg(30.0),
            },
        },
    };
    top.occurrences.push(arm);
    for (n, o, a) in [("pin", "base", "pin"), ("eye", "arm", "eye")] {
        top.connectors.push(MateConnector {
            name: n.into(),
            occurrence: RelPath::new(&[o]),
            anchor: AnchorRef::Published { name: a.into() },
            offset: None,
            flip: false,
        });
    }
    let angle = match (limit, drive) {
        (Some((lo, hi)), _) => Some(TargetExpr::Range {
            min: ValueExpr::deg(lo),
            max: ValueExpr::deg(hi),
        }),
        (None, Some(v)) => Some(TargetExpr::Exact {
            value: ValueExpr::deg(v),
        }),
        _ => None,
    };
    top.mates.push(MateDef {
        name: "hinge".into(),
        kind: MateKind::Revolute { angle },
        a: "pin".into(),
        b: "eye".into(),
        when: None,
        suppressed: false,
    });
    src.add_assembly(top);
    src
}

#[test]
fn a08_revolute_mates_solve_report_dof_and_limits() {
    let src = hinge_project(None, None);
    let r = run(&src, "hinge.assembly", &ResolveRequest::default());
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    let arm = r.node(r.find_display("arm").unwrap());
    assert!(
        close(arm.world.translation, [0.0, 0.0, 0.05]),
        "{:?}",
        arm.world.translation
    );
    assert_eq!(arm.dof, Some(1), "a revolute leaves one rotation");
    assert_eq!(arm.pose, PoseStatus::Floating);
    assert!(codes(&r).contains(&DiagCode::AsmUnderDefined));
    // Driven: fully defined at 90°.
    let r = run(
        &hinge_project(None, Some(90.0)),
        "hinge.assembly",
        &ResolveRequest::default(),
    );
    let arm = r.node(r.find_display("arm").unwrap());
    assert_eq!(arm.dof, Some(0));
    assert_eq!(arm.pose, PoseStatus::Solved);
    let tip = arm.world.apply([0.2, 0.0, 0.0]);
    assert!(close(tip, [0.0, 0.2, 0.05]), "{:?}", tip);
    // Limited to 0..10°: the 30° start clamps at 10°.
    let r = run(
        &hinge_project(Some((0.0, 10.0)), None),
        "hinge.assembly",
        &ResolveRequest::default(),
    );
    let m = &r.mates[0];
    assert_eq!(m.active_limits.len(), 1, "{:?}", r.mates);
    assert!((m.active_limits[0] - 10f64.to_radians()).abs() < 1e-9);
    assert!(codes(&r).contains(&DiagCode::AsmLimitActive));
}

/// A mechanism: `base` fixed, the other parts floating at the origin, each part publishing
/// the named frames, connectors named `<occurrence>.<frame>`.
fn mechanism(parts: &[(&str, Vec<(&str, Frame)>)], mates: Vec<(&str, MateKind, &str, &str)>) -> MemorySource {
    let mut src = MemorySource::default();
    let mut top = AssemblyDocument::new("m.assembly", "m");
    for (name, frames) in parts {
        let published = frames
            .iter()
            .map(|(n, f)| (*n, PublishedGeometry::Frame { frame: *f }))
            .collect();
        add_part(&mut src, &format!("{name}.part"), name, unit_box(), published);
        let mut o = Occurrence::new(name, DefinitionRef::part(&format!("{name}.part"), None));
        o.fixed = *name == "base";
        top.occurrences.push(o);
        for (f, _) in frames {
            top.connectors.push(MateConnector {
                name: format!("{name}.{f}"),
                occurrence: RelPath::new(&[name]),
                anchor: AnchorRef::Published { name: (*f).into() },
                offset: None,
                flip: false,
            });
        }
    }
    for (name, kind, a, b) in mates {
        top.mates.push(MateDef { name: name.into(), kind, a: a.into(), b: b.into(), when: None, suppressed: false });
    }
    src.add_assembly(top);
    src
}

fn axes(origin: [f64; 3], x: [f64; 3], z: [f64; 3]) -> Frame {
    let y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
    Frame { origin, x, y, z }
}

fn driven(deg: f64) -> MateKind {
    MateKind::Revolute { angle: Some(TargetExpr::Exact { value: ValueExpr::deg(deg) }) }
}

#[test]
fn a08b_gear_rack_screw_and_pin_slot_mates_couple_their_motions() {
    let up = |x: f64, y: f64| axes([x, y, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    // Gears: A driven 30° on its pin, B (twice A's teeth) turns -15° on its own.
    let src = mechanism(
        &[("base", vec![("pa", up(0.0, 0.0)), ("pb", up(0.1, 0.0))]), ("ga", vec![("eye", up(0.0, 0.0))]), ("gb", vec![("eye", up(0.0, 0.0))])],
        vec![
            ("ra", driven(30.0), "base.pa", "ga.eye"),
            ("rb", MateKind::Revolute { angle: None }, "base.pb", "gb.eye"),
            ("mesh", MateKind::Gear { ratio: ValueExpr::number(2.0) }, "ga.eye", "gb.eye"),
        ],
    );
    let r = run(&src, "m.assembly", &ResolveRequest::default());
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    let gb = r.node(r.find_display("gb").unwrap());
    let x = gb.world.apply_vector([1.0, 0.0, 0.0]);
    let t = (-15f64).to_radians();
    assert!(close(x, [t.cos(), t.sin(), 0.0]), "{x:?}");
    assert!(close(gb.world.translation, [0.1, 0.0, 0.0]), "{:?}", gb.world.translation);
    assert_eq!(gb.dof, Some(0), "the gear takes B's one rotation");

    // Rack and pinion: the pinion turns 90°; the rack, on a slider along x, moves a quarter of
    // its 100 mm travel per turn.
    let along_x = axes([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]);
    let src = mechanism(
        &[("base", vec![("pin", up(0.0, 0.0)), ("guide", along_x)]), ("pinion", vec![("eye", up(0.0, 0.0))]), ("rack", vec![("rail", along_x), ("teeth", up(0.0, 0.0))])],
        vec![
            ("turn", driven(90.0), "base.pin", "pinion.eye"),
            ("slide", MateKind::Slider { distance: None }, "base.guide", "rack.rail"),
            ("drive", MateKind::RackPinion { travel: mm(100.0) }, "pinion.eye", "rack.teeth"),
        ],
    );
    let r = run(&src, "m.assembly", &ResolveRequest::default());
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    let rack = r.node(r.find_display("rack").unwrap());
    assert!(close(rack.world.translation, [0.025, 0.0, 0.0]), "{:?}", rack.world.translation);
    assert_eq!(rack.dof, Some(0));

    // Screw: a nut turned 90° on a 2 mm lead advances 0.5 mm.
    let src = mechanism(
        &[("base", vec![("axis", up(0.0, 0.0))]), ("nut", vec![("eye", up(0.0, 0.0))])],
        vec![
            ("thread", MateKind::Screw { lead: mm(2.0) }, "base.axis", "nut.eye"),
            ("turn", MateKind::Angle { measure: AngleMeasure::AboutZ, target: TargetExpr::Exact { value: ValueExpr::deg(90.0) } }, "base.axis", "nut.eye"),
        ],
    );
    let r = run(&src, "m.assembly", &ResolveRequest::default());
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    let nut = r.node(r.find_display("nut").unwrap());
    assert!(close(nut.world.translation, [0.0, 0.0, 0.0005]), "{:?}", nut.world.translation);
    assert_eq!(nut.dof, Some(0));

    // Pin in a slot: the pin slides along the slot and turns in it.
    let src = mechanism(
        &[("base", vec![("slot", up(0.0, 0.0))]), ("link", vec![("pin", up(0.0, 0.0))])],
        vec![("guide", MateKind::PinSlot, "link.pin", "base.slot")],
    );
    let r = run(&src, "m.assembly", &ResolveRequest::default());
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    assert_eq!(r.node(r.find_display("link").unwrap()).dof, Some(2));
}

#[test]
fn a09_conflicting_and_redundant_mates_are_named() {
    let mut src = hinge_project(None, Some(90.0));
    let mut top = (*src.assemblies[&(FilePath::new("hinge.assembly"), None)]).clone();
    top.mates.push(MateDef {
        name: "hinge_again".into(),
        kind: MateKind::Revolute {
            angle: Some(TargetExpr::Exact {
                value: ValueExpr::deg(45.0),
            }),
        },
        a: "pin".into(),
        b: "eye".into(),
        when: None,
        suppressed: false,
    });
    src.add_assembly(top.clone());
    let r = run(&src, "hinge.assembly", &ResolveRequest::default());
    let c = r.diagnostics.with_code(DiagCode::AsmMateConflict);
    assert!(!c.is_empty(), "{:?}", r.diagnostics.diagnostics);
    assert!(c[0].message.contains("hinge"), "{}", c[0].message);
    // Same angle twice: redundant, not conflicting.
    top.mates[1].kind = MateKind::Revolute {
        angle: Some(TargetExpr::Exact {
            value: ValueExpr::deg(90.0),
        }),
    };
    src.add_assembly(top);
    let r = run(&src, "hinge.assembly", &ResolveRequest::default());
    assert!(
        r.diagnostics
            .with_code(DiagCode::AsmMateConflict)
            .is_empty(),
        "{:?}",
        r.diagnostics.diagnostics
    );
    assert!(!r
        .diagnostics
        .with_code(DiagCode::AsmMateRedundant)
        .is_empty());
}

#[test]
fn a10_flexible_sub_assemblies_join_the_parent_solve() {
    let mut src = MemorySource::default();
    add_part(
        &mut src,
        "base.part",
        "B",
        unit_box(),
        vec![
            ("pin", frame_at(0.0, 0.0, 0.05)),
            ("stop", frame_at(0.3, 0.3, 0.05)),
        ],
    );
    add_part(
        &mut src,
        "link.part",
        "L",
        unit_box(),
        vec![
            ("eye", frame_at(0.0, 0.0, 0.0)),
            ("tip", frame_at(0.2, 0.0, 0.0)),
        ],
    );
    let mut arm = AssemblyDocument::new("arm.assembly", "arm");
    arm.occurrences.push(Occurrence::new(
        "link",
        DefinitionRef::part("link.part", None),
    ));
    arm.connectors.push(MateConnector {
        name: "eye".into(),
        occurrence: RelPath::new(&["link"]),
        anchor: AnchorRef::Published { name: "eye".into() },
        offset: None,
        flip: false,
    });
    arm.publishes.push(AssemblyPublish {
        name: "eye".into(),
        target: AssemblyPublishTarget::Connector {
            connector: "eye".into(),
        },
    });
    src.add_assembly(arm);
    let mut top = AssemblyDocument::new("top.assembly", "top");
    top.occurrences.push(Occurrence::new(
        "base",
        DefinitionRef::part("base.part", None),
    ));
    let mut a = Occurrence::new("arm", DefinitionRef::assembly("arm.assembly"));
    a.fixed = false;
    a.flexible = true;
    a.placement = at(1.0, 2.0, 3.0);
    top.occurrences.push(a);
    top.connectors.push(MateConnector {
        name: "pin".into(),
        occurrence: RelPath::new(&["base"]),
        anchor: AnchorRef::Published { name: "pin".into() },
        offset: None,
        flip: false,
    });
    // A connector through the flexible sub-assembly's published connector.
    top.connectors.push(MateConnector {
        name: "arm_eye".into(),
        occurrence: RelPath::new(&["arm"]),
        anchor: AnchorRef::Published { name: "eye".into() },
        offset: None,
        flip: false,
    });
    top.mates.push(MateDef {
        name: "hinge".into(),
        kind: MateKind::Revolute {
            angle: Some(TargetExpr::Exact {
                value: ValueExpr::deg(0.0),
            }),
        },
        a: "pin".into(),
        b: "arm_eye".into(),
        when: None,
        suppressed: false,
    });
    src.add_assembly(top);
    let r = run(&src, "top.assembly", &ResolveRequest::default());
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    let link = r.node(r.find_display("arm/link").unwrap());
    assert!(
        close(link.world.translation, [0.0, 0.0, 0.05]),
        "{:?}",
        link.world.translation
    );
    assert_eq!(r.stats.mate_scopes, 1);
}

// ------------------------------------------------------------------------------ patterns

#[test]
fn a11_patterns_and_mirrors_keep_semantic_keys() {
    let mut src = MemorySource::default();
    let keys: Vec<(InstanceKey, Frame)> = ["h1", "h2", "h3"]
        .iter()
        .enumerate()
        .map(|(i, k)| {
            (
                InstanceKey::Row { key: k.to_string() },
                Frame {
                    origin: [0.1 * i as f64, 0.0, 0.01],
                    ..Frame::WORLD
                },
            )
        })
        .collect();
    add_part(
        &mut src,
        "plate.part",
        "P",
        unit_box(),
        vec![("holes", PublishedGeometry::Pattern { frames: keys })],
    );
    add_part(&mut src, "bolt.part", "BT", unit_box(), vec![]);
    add_part(&mut src, "rib.part", "RB", unit_box(), vec![]);
    let mut top = AssemblyDocument::new("p.assembly", "p");
    let mut plate = Occurrence::new("plate", DefinitionRef::part("plate.part", None));
    plate.placement = at(1000.0, 0.0, 0.0);
    top.occurrences.push(plate);
    let mut bolt = Occurrence::new("bolt", DefinitionRef::part("bolt.part", None));
    bolt.placement = at(1000.0, 0.0, 10.0);
    top.occurrences.push(bolt);
    let mut rib = Occurrence::new("rib", DefinitionRef::part("rib.part", None));
    rib.placement = at(0.0, 500.0, 0.0);
    top.occurrences.push(rib);
    top.patterns.push(OccurrencePattern {
        name: "bolts".into(),
        seeds: vec!["bolt".into()],
        layout: OccurrenceLayout::FollowPattern {
            occurrence: RelPath::new(&["plate"]),
            publish: "holes".into(),
        },
        skip: vec![],
        when: None,
    });
    top.patterns.push(OccurrencePattern {
        name: "ribs".into(),
        seeds: vec!["rib".into()],
        layout: OccurrenceLayout::Linear {
            direction: [1.0, 0.0, 0.0],
            spacing: mm(100.0),
            count: ValueExpr::number(5.0),
        },
        skip: vec![InstanceKey::Linear { step: 2 }],
        when: None,
    });
    top.mirrors.push(OccurrenceMirror {
        name: "lr".into(),
        seeds: vec!["rib".into()],
        plane: PlaneSpec::Explicit {
            point: [mm(0.0), mm(0.0), mm(0.0)],
            normal: [0.0, 1.0, 0.0],
        },
        hand: MirrorHand::Symmetric {
            point: [mm(0.0), mm(0.0), mm(0.0)],
            normal: [0.0, 1.0, 0.0],
        },
        when: None,
    });
    src.add_assembly(top.clone());
    let r = run(&src, "p.assembly", &ResolveRequest::default());
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    let h3 = r
        .find_display("bolt@bolts[\"h3\"]")
        .expect("keyed by the hole key");
    assert!(
        close(r.node(h3).world.translation, [1.2, 0.0, 0.01]),
        "{:?}",
        r.node(h3).world.translation
    );
    assert!(r.find_display("rib@ribs[2]").is_none(), "skipped by key");
    let r4 = r.find_display("rib@ribs[4]").unwrap();
    assert!(close(r.node(r4).world.translation, [0.4, 0.5, 0.0]));
    let m = r.node(r.find_display("rib@lr[mirror]").unwrap());
    assert!(
        close(m.world.translation, [0.0, -0.5, 0.0]),
        "{:?}",
        m.world.translation
    );
    assert!(
        (m.world.determinant() - 1.0).abs() < 1e-12,
        "mirrored placements are proper"
    );
    // A pattern count change keeps existing keys and ids.
    let id4 = r.node(r4).uuid;
    if let OccurrenceLayout::Linear { count, .. } = &mut top.patterns[1].layout {
        *count = ValueExpr::number(8.0);
    }
    src.add_assembly(top);
    let r2 = run(&src, "p.assembly", &ResolveRequest::default());
    assert_eq!(r2.node(r2.find_display("rib@ribs[4]").unwrap()).uuid, id4);
    assert!(r2.find_display("rib@ribs[7]").is_some());
}

#[test]
fn a11b_a_mirror_of_a_pattern_reflects_its_seed_and_every_instance() {
    let mut src = MemorySource::default();
    add_part(&mut src, "rivet.part", "RV", unit_box(), vec![]);
    let mut top = AssemblyDocument::new("w.assembly", "w");
    let mut rivet = Occurrence::new("rivet", DefinitionRef::part("rivet.part", None));
    rivet.placement = at(100.0, 400.0, 0.0);
    top.occurrences.push(rivet);
    top.patterns.push(OccurrencePattern {
        name: "row".into(),
        seeds: vec!["rivet".into()],
        layout: OccurrenceLayout::Linear {
            direction: [0.6, 0.8, 0.0],
            spacing: mm(50.0),
            count: ValueExpr::number(4.0),
        },
        skip: vec![InstanceKey::Linear { step: 2 }],
        when: None,
    });
    let mirror = |seeds: Vec<String>| OccurrenceMirror {
        name: "port".into(),
        seeds,
        plane: PlaneSpec::Explicit {
            point: [mm(0.0), mm(0.0), mm(0.0)],
            normal: [0.0, 1.0, 0.0],
        },
        hand: MirrorHand::Symmetric {
            point: [mm(0.0), mm(0.0), mm(0.0)],
            normal: [0.0, 1.0, 0.0],
        },
        when: None,
    };
    top.mirrors.push(mirror(vec!["row".into()]));
    src.add_assembly(top.clone());
    let r = run(&src, "w.assembly", &ResolveRequest::default());
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    let seed = r.node(r.find_display("rivet@port[mirror]").expect("the seed's image"));
    assert!(close(seed.world.translation, [0.1, -0.4, 0.0]), "{:?}", seed.world.translation);
    // Step 3 sits 150 mm along (0.6, 0.8); its image runs along (0.6, -0.8).
    let s3 = r.node(r.find_display("rivet@port[3]").expect("an instance's image"));
    assert!(
        close(s3.world.translation, [0.19, -0.52, 0.0]),
        "{:?}",
        s3.world.translation
    );
    assert!((s3.world.determinant() - 1.0).abs() < 1e-12, "images are proper");
    assert!(r.find_display("rivet@port[1]").is_some());
    assert!(r.find_display("rivet@port[2]").is_none(), "the pattern's skip carries over");
    // Each image is the reflection of its source instance.
    let src3 = r.node(r.find_display("rivet@row[3]").unwrap());
    let t = src3.world.translation;
    assert!(close(s3.world.translation, [t[0], -t[1], t[2]]));

    // Naming the seed both directly and through its pattern is an error naming both.
    top.mirrors[0] = mirror(vec!["rivet".into(), "row".into()]);
    src.add_assembly(top);
    let r = run(&src, "w.assembly", &ResolveRequest::default());
    let e = errors(&r);
    assert!(
        e.iter()
            .any(|d| d.contains("reflects `rivet[mirror]` twice") && d.contains("pattern `row`")),
        "{:?}",
        e
    );
}

// ------------------------------------------------------------------------------ fasteners

fn fastener_project(
    pitch: f64,
    exceptions: Vec<FastenerException>,
) -> (MemorySource, TableCatalog) {
    let mut src = MemorySource::default();
    let chain = PathChain {
        key: "flange".into(),
        points: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        normals: vec![[0.0, 0.0, 1.0]; 2],
    };
    add_part(
        &mut src,
        "spar.part",
        "SP",
        Aabb::new([0.0, -0.05, -0.1], [1.0, 0.05, 0.0]),
        vec![
            (
                "upper_flange",
                PublishedGeometry::Path {
                    chains: vec![chain],
                },
            ),
            (
                "thickness",
                PublishedGeometry::Value {
                    value: Quantity::new(0.003, Dimension::LENGTH),
                },
            ),
        ],
    );
    add_part(
        &mut src,
        "skin.part",
        "SK",
        Aabb::new([0.0, -0.5, 0.0], [1.0, 0.5, 0.002]),
        vec![(
            "thickness",
            PublishedGeometry::Value {
                value: Quantity::new(0.002, Dimension::LENGTH),
            },
        )],
    );
    let fam = DefinitionRef::part("hilok.part", None);
    let mut s = DefinitionSummary::new(fam.clone());
    s.part_number = Some("HL".into());
    src.add_part(s, ConfiguredSummary::default());
    let mut cat = TableCatalog::default();
    for (d, tag) in [(0.0048, "6"), (0.0052, "6X")] {
        for g in 1..=10 {
            cat.rows.push(CatalogRow {
                family: fam.clone(),
                diameter: d,
                grip_min: (g - 1) as f64 * 1e-3,
                grip_max: g as f64 * 1e-3,
                configuration: ConfigurationName::new(format!("{}-{}", tag, g)),
                part_number: format!("HL{}-{}", tag, g),
            });
        }
    }
    let mut top = AssemblyDocument::new("joint.assembly", "joint");
    top.occurrences.push(Occurrence::new(
        "spar",
        DefinitionRef::part("spar.part", None),
    ));
    top.occurrences.push(Occurrence::new(
        "skin",
        DefinitionRef::part("skin.part", None),
    ));
    top.fasteners.push(FastenerSet {
        name: "row_a".into(),
        when: None,
        layout: FastenerLayout::Along(AlongRule {
            occurrence: RelPath::new(&["spar"]),
            path: "upper_flange".into(),
            chains: vec![],
            spacing: Spacing::Pitch { pitch: mm(pitch) },
            edge_distance: mm(10.0),
            rows: vec![
                RowSpec {
                    key: "a".into(),
                    offset: mm(-10.0),
                },
                RowSpec {
                    key: "b".into(),
                    offset: mm(10.0),
                },
            ],
            stations: StationBasis::ArcLength,
            key_step: Some(mm(0.01)),
        }),
        fastener: FastenerSpec {
            family: fam,
            diameter: mm(4.8),
            grip: GripSpec::Auto,
        },
        stack: vec![RelPath::new(&["skin"]), RelPath::new(&["spar"])],
        drill: DrillSpec::default(),
        exceptions,
    });
    src.add_assembly(top);
    (src, cat)
}

fn at_station(mm_: f64, row: Option<&str>) -> ExceptionSelector {
    ExceptionSelector::Station {
        station: mm(mm_),
        chain: None,
        row: row.map(|r| r.into()),
        tolerance: None,
    }
}

#[test]
fn a12_fastener_rules_expand_with_stable_station_keys() {
    let ex = vec![
        FastenerException {
            at: at_station(100.0, Some("a")),
            action: ExceptionAction::Omit,
        },
        FastenerException {
            at: at_station(190.0, None),
            action: ExceptionAction::Oversize { delta: mm(0.4) },
        },
    ];
    let (src, cat) = fastener_project(30.0, ex.clone());
    let gauge = NominalThicknessGauge::default();
    let r = run_with(
        &src,
        "joint.assembly",
        &ResolveRequest::default(),
        None,
        Some(&cat),
        None,
        Some(&gauge),
    );
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    let e = &r.fasteners[0].expansion;
    // 10 mm … 970 mm at 30 mm: 33 stations × 2 rows, one omitted.
    assert_eq!(e.len(), 33 * 2 - 1);
    assert_eq!(e.omitted.len(), 1);
    let k190 = FastenerKey::Station {
        chain: "flange".into(),
        row: "b".into(),
        station: 19000,
    };
    let i = e.index_of(&k190).unwrap();
    assert!(
        (e.diameters[i] - 0.0052).abs() < 1e-12,
        "oversized on both rows"
    );
    assert_eq!(
        e.parts[e.selection[i] as usize].part_number, "HL6X-5",
        "grip 5 mm from the stack"
    );
    assert!(
        close(e.positions[i], [0.19, 0.01, 0.0]),
        "{:?}",
        e.positions[i]
    );
    assert_eq!(e.drills.len(), 2);
    assert_eq!(e.drills[0].instances.len(), e.len());
    // A pitch change keeps the exception where its station survives (190 = 10 + 6 × 30) …
    let (src, cat) = fastener_project(20.0, ex.clone());
    let r = run_with(
        &src,
        "joint.assembly",
        &ResolveRequest::default(),
        None,
        Some(&cat),
        None,
        Some(&gauge),
    );
    let e = &r.fasteners[0].expansion;
    let i = e.index_of(&k190).unwrap();
    assert!((e.diameters[i] - 0.0052).abs() < 1e-12);
    // … and diagnoses it where it does not (pitch 25: stations 10, 35, …, 185, 210).
    let (src, cat) = fastener_project(25.0, ex);
    let r = run_with(
        &src,
        "joint.assembly",
        &ResolveRequest::default(),
        None,
        Some(&cat),
        None,
        Some(&gauge),
    );
    let un = r
        .diagnostics
        .with_code(DiagCode::FastenerExceptionUnmatched);
    assert_eq!(un.len(), 2, "{:?}", r.diagnostics.diagnostics);
    assert!(matches!(
        un[0].repairs[0],
        runmat_cad::diagnostics::Repair::RemoveException { .. }
    ));
    // Expansion is deterministic to the bit.
    let r2 = run_with(
        &src,
        "joint.assembly",
        &ResolveRequest::default(),
        None,
        Some(&cat),
        None,
        Some(&gauge),
    );
    assert_eq!(
        r.fasteners[0].expansion.digest(),
        r2.fasteners[0].expansion.digest()
    );
}

#[test]
fn a13_instance_tables_round_trip_and_are_integrity_checked() {
    let n = 500;
    let t = ColumnarTable {
        columns: vec![
            Column {
                name: "key".into(),
                data: ColumnData::Text {
                    values: (0..n).map(|i| format!("R{:04}", i)).collect(),
                },
            },
            Column {
                name: "x".into(),
                data: ColumnData::F64 {
                    unit: "mm".into(),
                    values: (0..n).map(|i| i as f64 * 1.5).collect(),
                },
            },
            Column {
                name: "y".into(),
                data: ColumnData::F64 {
                    unit: "mm".into(),
                    values: vec![0.1; n],
                },
            },
            Column {
                name: "z".into(),
                data: ColumnData::F64 {
                    unit: "mm".into(),
                    values: vec![-0.0; n],
                },
            },
            Column {
                name: "lot".into(),
                data: ColumnData::I64 {
                    values: (0..n as i64).collect(),
                },
            },
        ],
    };
    let text = t.encode_text();
    assert_eq!(
        ColumnarTable::decode_text(&text).unwrap(),
        t,
        "text encoding is lossless"
    );
    assert_eq!(ColumnarTable::decode_binary(&t.encode_binary()).unwrap(), t);
    let mut tables = MemoryTables::default();
    let digest = tables.put(&t);
    let (mut src, cat) = fastener_project(30.0, vec![]);
    let mut top = (*src.assemblies[&(FilePath::new("joint.assembly"), None)]).clone();
    top.fasteners[0].layout = FastenerLayout::Table(TableRule {
        table: digest.clone(),
        columns: TableColumns {
            key: "key".into(),
            x: "x".into(),
            y: "y".into(),
            z: "z".into(),
            axis: None,
            diameter: None,
        },
        frame: RelPath::new(&["spar"]),
    });
    top.fasteners[0].exceptions = vec![FastenerException {
        at: ExceptionSelector::Key {
            key: "R0007".into(),
        },
        action: ExceptionAction::Omit,
    }];
    src.add_assembly(top);
    let gauge = NominalThicknessGauge::default();
    let r = run_with(
        &src,
        "joint.assembly",
        &ResolveRequest::default(),
        None,
        Some(&cat),
        Some(&tables),
        Some(&gauge),
    );
    assert!(errors(&r).is_empty(), "{:?}", errors(&r));
    assert_eq!(r.fasteners[0].expansion.len(), n - 1);
    // Tampered content fails closed.
    tables.blobs.get_mut(&digest).unwrap()[40] ^= 1;
    let r = run_with(
        &src,
        "joint.assembly",
        &ResolveRequest::default(),
        None,
        Some(&cat),
        Some(&tables),
        Some(&gauge),
    );
    assert!(codes(&r).contains(&DiagCode::FastenerTableIntegrity));
    assert!(r.fasteners.is_empty());
}

// ------------------------------------------------------------------------------ BOM, where-used

#[test]
fn a14_bom_is_configuration_aware_and_counts_fasteners() {
    let (src, cat) = fastener_project(30.0, vec![]);
    let gauge = NominalThicknessGauge::default();
    let r = run_with(
        &src,
        "joint.assembly",
        &ResolveRequest::default(),
        None,
        Some(&cat),
        None,
        Some(&gauge),
    );
    let bom = flat_bom(&r, Some(&src), &BomOptions::default());
    assert_eq!(bom.total_parts, 2);
    assert_eq!(bom.total_fasteners, 66);
    let hl = bom
        .rows
        .iter()
        .find(|r| r.item.part_number.as_deref() == Some("HL6-5"))
        .unwrap();
    assert_eq!(hl.quantity, 66);
    // Dash numbers from configured summaries.
    let src = bracket_project();
    let r = run(&src, "rig.assembly", &ResolveRequest::default());
    let bom = flat_bom(&r, Some(&src), &BomOptions::default());
    assert!(
        bom.rows
            .iter()
            .any(|r| r.item.part_number.as_deref() == Some("BR-1-L")),
        "{:?}",
        bom.rows
    );
    let ind = indented_bom(&r, Some(&src));
    let bolts = ind
        .iter()
        .find(|r| r.item.part_number.as_deref() == Some("BT-6"))
        .unwrap();
    assert_eq!((bolts.level, bolts.quantity), (2, 2));
    // Where-used across the 150% structure, transitively.
    let src = deep_project();
    let docs: Vec<_> = src.assemblies.values().cloned().collect();
    let idx = WhereUsedIndex::build(&docs);
    let users = idx.transitive(&FilePath::new("bolt.part"));
    assert_eq!(
        users.into_iter().map(|f| f.0).collect::<Vec<_>>(),
        vec![
            "bolts.assembly",
            "mid.assembly",
            "rig.assembly",
            "top.assembly"
        ]
    );
}

// ------------------------------------------------------------------------------ interference

#[test]
fn a15_interference_and_keep_out_envelopes() {
    let mut src = MemorySource::default();
    add_part(
        &mut src,
        "block.part",
        "BK",
        Aabb::new([0.0; 3], [0.1, 0.1, 0.1]),
        vec![],
    );
    let mut top = AssemblyDocument::new("i.assembly", "i");
    let mut a = Occurrence::new("a", DefinitionRef::part("block.part", None));
    a.placement = at(0.0, 0.0, 0.0);
    let mut b = Occurrence::new("b", DefinitionRef::part("block.part", None));
    b.placement = at(50.0, 50.0, 50.0);
    let mut c = Occurrence::new("c", DefinitionRef::part("block.part", None));
    c.placement = at(105.0, 0.0, -60.0);
    let mut d = Occurrence::new("d", DefinitionRef::part("block.part", None));
    d.placement = at(1000.0, 1000.0, 1000.0);
    top.occurrences = vec![a, b, c, d];
    top.envelopes.push(EnvelopeDecl {
        name: "systems_zone".into(),
        owner: Some("d".into()),
        min: [mm(950.0), mm(950.0), mm(950.0)],
        max: [mm(1200.0), mm(1200.0), mm(1200.0)],
        keep_out: true,
    });
    let mut e = Occurrence::new("e", DefinitionRef::part("block.part", None));
    e.placement = at(1150.0, 1150.0, 1150.0);
    top.occurrences.push(e);
    src.add_assembly(top);
    let r = run(&src, "i.assembly", &ResolveRequest::default());
    let rep = check(&r, &src, &BoxKernel, &InterferenceRequest::default());
    let pairs: Vec<(String, String)> = rep
        .pairs
        .iter()
        .map(|p| (p.a.clone(), p.b.clone()))
        .collect();
    assert_eq!(
        pairs,
        vec![
            ("a".to_string(), "b".to_string()),
            ("e".to_string(), "envelope.systems_zone".to_string())
        ],
        "the owner `d` may sit in its envelope; `e` intrudes"
    );
    // Clearance: a and c are 5 mm apart.
    let rep = check(
        &r,
        &src,
        &BoxKernel,
        &InterferenceRequest {
            mode: CheckMode::Clearance { min: 0.01 },
            include_envelopes: false,
            ..Default::default()
        },
    );
    assert!(rep.pairs.iter().any(|p| p.a == "a" && p.b == "c"));
}

// ------------------------------------------------------------------------------ working sets

#[test]
fn a16_working_sets_plan_loads_and_dedupe_artifacts() {
    let p = generate(&SynthSpec::small());
    let e = Env::new();
    let ctx = ResolveContext {
        source: &p.source,
        units: &e.units,
        dynamic: &e.dynamic,
        lock: Some(&p.lock),
        catalog: Some(&p.catalog),
        tables: None,
        gauge: Some(&p.gauge),
    };
    let r = resolve(&p.top, &ResolveRequest::default(), &ctx);
    let ws = WorkingSet {
        name: "section1 frames".into(),
        rules: vec![
            LoadRule {
                query: Query::Glob {
                    pattern: "section1/**/web".into(),
                },
                level: LoadLevel::R2,
                representation: None,
            },
            LoadRule {
                query: Query::Under {
                    path: RelPath::new(&["section2"]),
                },
                level: LoadLevel::R1,
                representation: None,
            },
            LoadRule {
                query: Query::Glob {
                    pattern: "wing_left".into(),
                },
                level: LoadLevel::R3,
                representation: None,
            },
        ],
        default: LoadLevel::R0,
    };
    let plan = plan(&ws, &r, &p.source);
    let spec = SynthSpec::small();
    assert_eq!(plan.counts[&LoadLevel::R2], spec.frames_per_section as u64);
    // All section-1 webs share one prototype: one exact artifact.
    let exact: Vec<_> = plan
        .artifacts
        .iter()
        .filter(|a| {
            a.level == LoadLevel::R2
                && r.prototypes[a.prototype as usize].definition.file.0 == "library/frame-std.part"
        })
        .collect();
    assert_eq!(exact.len(), 1, "{:?}", exact);
    // The editable wing needs its exact B-rep too (R3 implies R2).
    assert!(plan.artifacts.iter().any(|a| a.level == LoadLevel::R2
        && r.prototypes[a.prototype as usize].definition.file.0 == "wing/wing.part"));
    assert!(plan.text.contains(&FilePath::new("wing/wing.part")));
    assert!(plan.text.contains(&FilePath::new("aircraft.assembly")));
    assert!(!plan.allows(0, Operation::Edit));
}

#[test]
fn a17_collapsed_sub_assemblies_are_not_loaded() {
    let p = generate(&SynthSpec::small());
    let e = Env::new();
    let ctx = ResolveContext {
        source: &p.source,
        units: &e.units,
        dynamic: &e.dynamic,
        lock: Some(&p.lock),
        catalog: Some(&p.catalog),
        tables: None,
        gauge: Some(&p.gauge),
    };
    let before = p.source.loads.assemblies.get();
    let r = resolve(
        &p.top,
        &ResolveRequest {
            max_depth: Some(1),
            ..Default::default()
        },
        &ctx,
    );
    assert_eq!(
        p.source.loads.assemblies.get() - before,
        1,
        "only the top is opened at depth 1"
    );
    assert_eq!(r.stats.collapsed, SynthSpec::small().sections as u64);
    let bom = flat_bom(&r, None, &BomOptions::default());
    assert!(
        bom.rows
            .iter()
            .any(|r| r.item.kind == BomKind::Assembly && r.quantity == 2),
        "collapsed sections stand for themselves"
    );
}

// ------------------------------------------------------------------------------ partitions

#[test]
fn a18_live_uses_across_partitions_must_be_pinned() {
    let mut src = bracket_project();
    src.set_partition("rig.assembly", "structures");
    src.set_partition("plate.part", "systems");
    let mut s = DefinitionSummary::new(DefinitionRef::part("plate.part", None));
    s.part_number = Some("PL-1".into());
    src.add_part_revision(
        s,
        ConfiguredSummary {
            bbox: Some(unit_box()),
            publishes: [(PublishName::new("top"), frame_at(0.1, 0.05, 0.01))]
                .into_iter()
                .collect(),
            ..Default::default()
        },
        "B",
    );
    let r = run(&src, "rig.assembly", &ResolveRequest::default());
    let d = r.diagnostics.with_code(DiagCode::AsmLiveAcrossPartition);
    assert_eq!(d.len(), 1);
    assert!(
        matches!(&d[0].repairs[0], runmat_cad::diagnostics::Repair::PinOccurrence { revision, .. } if revision.0 == "B")
    );
    let mut top = (*src.assemblies[&(FilePath::new("rig.assembly"), None)]).clone();
    top.occurrences[0].source.context = XrefContext::Pinned {
        revision: "B".into(),
    };
    src.add_assembly(top.clone());
    let r = run(&src, "rig.assembly", &ResolveRequest::default());
    assert!(r
        .diagnostics
        .with_code(DiagCode::AsmLiveAcrossPartition)
        .is_empty());
    assert_eq!(
        r.node(r.find_display("plate").unwrap())
            .revision
            .as_ref()
            .unwrap()
            .0,
        "B"
    );
    top.occurrences[0].source.context = XrefContext::Pinned {
        revision: "Z".into(),
    };
    src.add_assembly(top);
    let r = run(&src, "rig.assembly", &ResolveRequest::default());
    assert!(r
        .diagnostics
        .with_code(DiagCode::AsmUnknownDefinition)
        .iter()
        .any(|d| d.message.contains("revision `Z`")));
}

// ------------------------------------------------------------------------------ index, cards

#[test]
fn a19_index_and_summary_cards_are_deterministic_and_compact() {
    let p = generate(&SynthSpec::small());
    let e = Env::new();
    let mk = || {
        let ctx = ResolveContext {
            source: &p.source,
            units: &e.units,
            dynamic: &e.dynamic,
            lock: Some(&p.lock),
            catalog: Some(&p.catalog),
            tables: None,
            gauge: Some(&p.gauge),
        };
        resolve(
            &p.top,
            &ResolveRequest {
                variant: VariantSelection::Named {
                    name: "freighter".into(),
                },
                ..Default::default()
            },
            &ctx,
        )
    };
    let (a, b) = (mk(), mk());
    let ja = assembly_index_json(&build_assembly_index(&a, Some(&p.source)));
    let jb = assembly_index_json(&build_assembly_index(&b, Some(&p.source)));
    assert_eq!(ja, jb);
    let ca = summary_card(&a, Some(&p.source), &CardOptions::default());
    let cb = summary_card(&b, Some(&p.source), &CardOptions::default());
    assert_eq!(ca.digest, cb.digest);
    assert!(
        ca.token_estimate < 2000,
        "{} tokens:\n{}",
        ca.token_estimate,
        ca.text
    );
    assert!(ca.text.contains("freighter=true"), "{}", ca.text);
    assert!(a.find_display("cargo_door").is_some());
    assert_eq!(
        a.node(a.find_display("engine_left").unwrap())
            .definition
            .file
            .0,
        "propulsion/engine-b.part"
    );
    // The heavy frames apply from MSN 41 (the freighter is MSN 120).
    let web = a.node(a.find_display("section1/frame/web").unwrap());
    assert_eq!(web.configuration.label(), "heavy");
}

#[test]
fn a20_synthetic_products_validate_and_resolve_every_named_configuration() {
    let p = generate(&SynthSpec::small());
    let v = validate_product(&p.top, None, &p.source);
    assert!(v.ok(), "{:?}", v.diagnostics.diagnostics);
    let e = Env::new();
    for name in ["launch", "freighter"] {
        let ctx = ResolveContext {
            source: &p.source,
            units: &e.units,
            dynamic: &e.dynamic,
            lock: Some(&p.lock),
            catalog: Some(&p.catalog),
            tables: None,
            gauge: Some(&p.gauge),
        };
        let r = resolve(
            &p.top,
            &ResolveRequest {
                variant: VariantSelection::Named { name: name.into() },
                ..Default::default()
            },
            &ctx,
        );
        assert!(errors(&r).is_empty(), "{}: {:?}", name, errors(&r));
        assert!(
            r.diagnostics
                .diagnostics
                .iter()
                .all(|d| d.severity != Severity::Warning),
            "{:?}",
            r.diagnostics.diagnostics
        );
        // The mirrored wing sits on the other side of the pinned interface frame.
        let wl = r
            .node(r.find_display("wing_left").unwrap())
            .world
            .translation;
        let wr = r
            .node(r.find_display("wing_left@wing_mirror[mirror]").unwrap())
            .world
            .translation;
        assert!((wl[1] + wr[1]).abs() < 1e-9 && wl[1] > 0.0);
    }
}

/// The scale gate: about 10k occurrences and over 100k fasteners. Timings are printed so the
/// CI log carries them; the budgets are generous for unoptimised test builds.
#[test]
fn a21_scale_gate() {
    let spec = SynthSpec::scale();
    let t0 = Instant::now();
    let p = generate(&spec);
    let t_gen = t0.elapsed();
    let e = Env::new();
    let ctx = ResolveContext {
        source: &p.source,
        units: &e.units,
        dynamic: &e.dynamic,
        lock: Some(&p.lock),
        catalog: Some(&p.catalog),
        tables: None,
        gauge: Some(&p.gauge),
    };
    let t1 = Instant::now();
    let v = validate_product(&p.top, None, &p.source);
    let t_val = t1.elapsed();
    assert!(v.ok(), "{:?}", v.diagnostics.diagnostics);
    let t2 = Instant::now();
    let r = resolve(&p.top, &ResolveRequest::default(), &ctx);
    let t_res = t2.elapsed();
    assert!(
        errors(&r).is_empty(),
        "{:?}",
        errors(&r).iter().take(5).collect::<Vec<_>>()
    );
    let t3 = Instant::now();
    let idx = build_assembly_index(&r, Some(&p.source));
    let json = assembly_index_json(&idx);
    let t_idx = t3.elapsed();
    let t4 = Instant::now();
    let card = summary_card(&r, Some(&p.source), &CardOptions::default());
    let t_card = t4.elapsed();
    let expected = spec.expected_fasteners();
    eprintln!(
        "scale gate: {} occurrences, {} prototypes, {} fasteners (expected {}), index {} KB, card {} tokens\n  generate {:?}, validate {:?}, resolve+expand {:?}, index {:?}, card {:?}",
        r.nodes.len(),
        r.prototypes.len(),
        r.fastener_count(),
        expected,
        json.len() / 1024,
        card.token_estimate,
        t_gen,
        t_val,
        t_res,
        t_idx,
        t_card
    );
    assert!(r.nodes.len() > 9_000, "{}", r.nodes.len());
    assert_eq!(r.fastener_count(), expected);
    assert!(r.fastener_count() > 100_000);
    assert!(card.token_estimate < 2000);
    assert!(t_res.as_secs() < 120, "resolve took {:?}", t_res);
}

#[test]
fn a22_values_flow_from_assembly_inputs_and_unknown_axes_are_reported() {
    let mut src = bracket_project();
    let mut top = (*src.assemblies[&(FilePath::new("rig.assembly"), None)]).clone();
    top.inputs
        .push(runmat_cad::value::InputDecl::new("gap", "length", mm(25.0)));
    top.configurations.push(AssemblyConfiguration {
        name: "wide".into(),
        overrides: [(InputName::new("gap"), mm(80.0))].into_iter().collect(),
        suppress: ["plate".to_string()].into_iter().collect(),
    });
    let bo = top
        .occurrences
        .iter_mut()
        .find(|o| o.name == "bolts")
        .unwrap();
    bo.placement = Placement::Offset {
        offset: OffsetExpr::translation([ValueExpr::input("gap"), mm(0.0), mm(0.0)]),
    };
    let br = top
        .occurrences
        .iter_mut()
        .find(|o| o.name == "bracket")
        .unwrap();
    br.placement = Placement::Origin;
    let mut ghost = Occurrence::new("ghost", DefinitionRef::part("bolt.part", None));
    ghost.when = Some(Condition::is_enum("nonexistent_axis", "x"));
    top.occurrences.push(ghost);
    src.add_assembly(top);
    let r = run(&src, "rig.assembly", &ResolveRequest::default());
    assert!(close(
        r.node(r.find_display("bolts").unwrap()).world.translation,
        [0.025, 0.0, 0.0]
    ));
    assert!(codes(&r).contains(&DiagCode::AsmUnknownAxis));
    let r = run(
        &src,
        "rig.assembly",
        &ResolveRequest {
            configuration: Some("wide".into()),
            ..Default::default()
        },
    );
    assert!(close(
        r.node(r.find_display("bolts").unwrap()).world.translation,
        [0.08, 0.0, 0.0]
    ));
    assert!(r.find_display("plate").is_none());
    let _ = Transform::IDENTITY;
    let _: BTreeMap<String, String> = r.variant.clone();
}

#[test]
fn a23_merges_report_paths_that_now_name_another_occurrence() {
    let base_src = bracket_project();
    let mut lock = LockSection::default();
    let mut m = SeededMinter { seed: "t".into() };
    for d in base_src.assemblies.values() {
        lock_assembly(d, &mut lock, &mut m);
    }
    let rig = FilePath::new("rig.assembly");
    let base = run_with(
        &base_src,
        "rig.assembly",
        &ResolveRequest::default(),
        Some(&lock),
        None,
        None,
        None,
    );
    // Ours renames `bracket` → `support`.
    let mut ours_src = bracket_project();
    let mut top = (*ours_src.assemblies[&(rig.clone(), None)]).clone();
    top.occurrences
        .iter_mut()
        .find(|o| o.name == "bracket")
        .unwrap()
        .name = "support".into();
    ours_src.add_assembly(top.clone());
    let mut ours_lock = lock.clone();
    ours_lock
        .rename(
            &item_path(&rig, "occurrence", "bracket"),
            &item_path(&rig, "occurrence", "support"),
        )
        .unwrap();
    let ours = run_with(
        &ours_src,
        "rig.assembly",
        &ResolveRequest::default(),
        Some(&ours_lock),
        None,
        None,
        None,
    );
    // The text merge keeps ours' rename and theirs' untouched `bracket` lines, so the merged
    // file has both a `support` and a new `bracket`.
    let mut merged_src = bracket_project();
    let mut merged = top.clone();
    let mut extra = Occurrence::new("bracket", DefinitionRef::part("plate.part", None));
    extra.placement = at(0.0, 0.0, 100.0);
    merged.occurrences.push(extra);
    merged_src.add_assembly(merged);
    let mr = run_with(
        &merged_src,
        "rig.assembly",
        &ResolveRequest::default(),
        Some(&ours_lock),
        None,
        None,
        None,
    );
    assert_ne!(
        mr.node(mr.find_display("bracket").unwrap()).uuid,
        mr.node(mr.find_display("support").unwrap()).uuid,
        "a new item under an old name is a new item"
    );
    let (redirects, diags) =
        runmat_cad::refs::merge::recheck_assembly_merge(Some(&base), &ours, &base, &mr);
    assert_eq!(redirects.len(), 1, "{:?}", redirects);
    assert_eq!(redirects[0].path, "bracket");
    assert_eq!(diags.diagnostics[0].code, DiagCode::MergeRedirected);
}
