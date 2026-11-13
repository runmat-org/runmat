//! Synthetic aircraft-shaped products for the scale gate.
//!
//! [`generate`] writes a project into a [`MemorySource`]: a top assembly with option and
//! effectivity axes (MSN, engine, freighter, a rule), fuselage sections built from a few shared
//! definitions (frames patterned along the section, clips patterned on each frame, stringers
//! patterned about the axis, a configuration switch by MSN), a mirrored wing attached to a
//! pinned interface frame, alternative engines, an optional cargo door, and lap-joint fastener
//! rules along each skin's published edge with several rows and exceptions. Sizes are
//! parameters, so the same generator serves a quick test and the 100k-fastener gate.

use crate::assembly::fasteners::*;
use crate::assembly::model::*;
use crate::assembly::options::*;
use crate::assembly::resolve::lock_assembly;
use crate::assembly::source::*;
use crate::ids::{ConfigurationName, FilePath, PublishName};
use crate::model::datum::Frame;
use crate::refs::lock::{LockSection, SeededMinter};
use crate::refs::XrefContext;
use crate::units::Dimension;
use crate::value::{BinaryOp, InputDecl, Quantity, ValueExpr};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct SynthSpec {
    pub sections: u32,
    pub frames_per_section: u32,
    pub clips_per_frame: u32,
    pub stringers_per_section: u32,
    /// Lap joints per section (one fastener set each).
    pub joints_per_section: u32,
    pub rows_per_joint: u32,
    pub skin_length_m: f64,
    pub pitch_mm: f64,
    pub msn_max: i64,
}

impl SynthSpec {
    /// A few hundred occurrences and a few thousand fasteners.
    pub fn small() -> Self {
        Self {
            sections: 2,
            frames_per_section: 6,
            clips_per_frame: 4,
            stringers_per_section: 8,
            joints_per_section: 1,
            rows_per_joint: 2,
            skin_length_m: 3.0,
            pitch_mm: 21.6,
            msn_max: 200,
        }
    }

    /// About 10k occurrences and over 100k fasteners.
    pub fn scale() -> Self {
        Self {
            sections: 10,
            frames_per_section: 40,
            clips_per_frame: 20,
            stringers_per_section: 60,
            joints_per_section: 2,
            rows_per_joint: 5,
            skin_length_m: 25.0,
            pitch_mm: 21.6,
            msn_max: 1000,
        }
    }

    /// The fasteners the rules should expand to (before exceptions).
    pub fn expected_fasteners(&self) -> u64 {
        let usable = self.skin_length_m - 2.0 * 0.0096;
        let per_row = (usable / (self.pitch_mm * 1e-3) + 1e-9).floor() as u64 + 1;
        per_row * self.rows_per_joint as u64 * self.joints_per_section as u64 * self.sections as u64
    }
}

pub struct SynthProject {
    pub source: MemorySource,
    pub catalog: TableCatalog,
    pub gauge: NominalThicknessGauge,
    pub top: FilePath,
    pub lock: LockSection,
    pub assemblies: Vec<FilePath>,
}

fn mm(v: f64) -> ValueExpr {
    ValueExpr::mm(v)
}

fn len_value(si: f64) -> PublishedGeometry {
    PublishedGeometry::Value {
        value: Quantity::new(si, Dimension::LENGTH),
    }
}

fn part(
    source: &mut MemorySource,
    file: &str,
    pn: &str,
    bbox: Aabb,
    publishes: Vec<(&str, PublishedGeometry)>,
    configurations: &[&str],
) {
    let def = DefinitionRef::part(file, None);
    let mut s = DefinitionSummary::new(def.clone());
    s.part_number = Some(pn.into());
    s.configurations = configurations
        .iter()
        .map(|c| ConfigurationName::new(*c))
        .collect();
    s.digest = format!("synth:{}", file);
    let c = ConfiguredSummary {
        bbox: Some(bbox),
        mass: Some(bbox.volume() * 2700.0),
        center_of_mass: None,
        part_number: None,
        publishes: publishes
            .into_iter()
            .map(|(k, v)| (PublishName::new(k), v))
            .collect(),
        artifact: format!("synth:{}#default", file),
    };
    source.add_part(s, c.clone());
    for cfg in configurations {
        let mut cc = c.clone();
        cc.part_number = Some(format!("{}-{}", pn, cfg));
        cc.artifact = format!("synth:{}#{}", file, cfg);
        source.add_configuration(&def, &ConfigKey::named(cfg), cc);
    }
}

fn occ(name: &str, def: DefinitionRef) -> Occurrence {
    Occurrence::new(name, def)
}

fn at(x: f64, y: f64, z: f64) -> Placement {
    Placement::Offset {
        offset: OffsetExpr::translation([mm(x), mm(y), mm(z)]),
    }
}

pub fn generate(spec: &SynthSpec) -> SynthProject {
    let mut src = MemorySource::default();
    let l = spec.skin_length_m * 1000.0;

    // Parts.
    part(
        &mut src,
        "library/frame-std.part",
        "FR-100",
        Aabb::new([0.0, -2.0, -2.0], [0.02, 2.0, 2.0]),
        vec![],
        &["std", "heavy"],
    );
    part(
        &mut src,
        "library/clip.part",
        "CL-200",
        Aabb::new([0.0, 0.0, 0.0], [0.03, 0.02, 0.04]),
        vec![],
        &[],
    );
    part(
        &mut src,
        "library/stringer.part",
        "ST-300",
        Aabb::new([0.0, -0.01, 1.9], [spec.skin_length_m, 0.01, 1.93]),
        vec![("thickness", len_value(1.6e-3))],
        &[],
    );
    let chains: Vec<PathChain> = (0..spec.joints_per_section)
        .map(|j| {
            let y = 0.2 + j as f64 * 0.3;
            PathChain {
                key: format!("lap{}", j),
                points: vec![
                    [0.0, y, 2.0],
                    [spec.skin_length_m * 0.5, y, 2.0],
                    [spec.skin_length_m, y, 2.0],
                ],
                normals: vec![[0.0, 0.0, 1.0]; 3],
            }
        })
        .collect();
    part(
        &mut src,
        "fuselage/skin.part",
        "SK-400",
        Aabb::new([0.0, -2.0, 1.95], [spec.skin_length_m, 2.0, 2.0]),
        vec![
            ("laps", PublishedGeometry::Path { chains }),
            ("thickness", len_value(2.0e-3)),
        ],
        &[],
    );
    part(
        &mut src,
        "wing/wing.part",
        "WG-500",
        Aabb::new([0.0, 0.0, -0.5], [6.0, 18.0, 0.5]),
        vec![(
            "root",
            PublishedGeometry::Frame {
                frame: Frame::WORLD,
            },
        )],
        &[],
    );
    part(
        &mut src,
        "propulsion/engine-a.part",
        "EN-A",
        Aabb::new([0.0, -1.0, -1.0], [4.0, 1.0, 1.0]),
        vec![],
        &[],
    );
    part(
        &mut src,
        "propulsion/engine-b.part",
        "EN-B",
        Aabb::new([0.0, -1.2, -1.2], [4.5, 1.2, 1.2]),
        vec![],
        &[],
    );
    part(
        &mut src,
        "fuselage/cargo-door.part",
        "CD-600",
        Aabb::new([0.0, 1.9, -1.0], [3.0, 2.0, 1.0]),
        vec![],
        &[],
    );
    // A pinned interface skeleton.
    let iface = DefinitionRef::part("interfaces/fuselage-wing.part", None);
    let mut s = DefinitionSummary::new(iface.clone());
    s.part_number = Some("IF-700".into());
    let mut pubs = BTreeMap::new();
    pubs.insert(
        PublishName::new("wing_ref"),
        PublishedGeometry::Frame {
            frame: Frame {
                origin: [l * 0.4e-3, 2.0, -0.85],
                ..Frame::WORLD
            },
        },
    );
    src.add_part_revision(
        s,
        ConfiguredSummary {
            publishes: pubs,
            ..Default::default()
        },
        "A",
    );
    // The fastener family.
    let hilok = DefinitionRef::part("library/fasteners/hi-lok.part", None);
    let mut s = DefinitionSummary::new(hilok.clone());
    s.part_number = Some("HL".into());
    let mut catalog = TableCatalog::default();
    for d in [4.8e-3f64, 5.2e-3] {
        for g in 1..=8 {
            let cfg = format!("d{}-g{}", (d * 1e4).round() as i64, g);
            s.configurations.push(ConfigurationName::new(&cfg));
            catalog.rows.push(CatalogRow {
                family: hilok.clone(),
                diameter: d,
                grip_min: (g - 1) as f64 * 1e-3,
                grip_max: g as f64 * 1e-3,
                configuration: ConfigurationName::new(&cfg),
                part_number: format!("HL-{}-{}", (d * 1e4).round() as i64, g),
            });
        }
    }
    src.add_part(s, ConfiguredSummary::default());

    // Frame assembly: a frame plus a linear pattern of clips.
    let mut frame = AssemblyDocument::new("fuselage/frame.assembly", "frame");
    frame.part_number = Some("FA-110".into());
    let mut f = occ("web", DefinitionRef::part("library/frame-std.part", None));
    f.configuration = ConfigurationSelection::Switch {
        cases: vec![ConfigurationCase {
            when: Condition::range("msn", Some(41), None),
            select: ConfigurationSelection::Named {
                name: "heavy".into(),
            },
        }],
        otherwise: Box::new(ConfigurationSelection::Named { name: "std".into() }),
    };
    frame.occurrences.push(f);
    let mut c = occ("clip", DefinitionRef::part("library/clip.part", None));
    c.placement = at(20.0, -1500.0, 1900.0);
    frame.occurrences.push(c);
    frame.patterns.push(OccurrencePattern {
        name: "clips".into(),
        seeds: vec!["clip".into()],
        layout: OccurrenceLayout::Linear {
            direction: [0.0, 1.0, 0.0],
            spacing: mm(3000.0 / spec.clips_per_frame.max(1) as f64),
            count: ValueExpr::number(spec.clips_per_frame as f64),
        },
        skip: vec![],
        when: None,
    });

    // Section assembly.
    let mut sec = AssemblyDocument::new("fuselage/section.assembly", "section");
    sec.part_number = Some("SE-120".into());
    sec.inputs.push(InputDecl::new(
        "frame_pitch",
        "length",
        mm(l / spec.frames_per_section.max(1) as f64),
    ));
    let mut fr = occ("frame", DefinitionRef::assembly("fuselage/frame.assembly"));
    fr.placement = at(10.0, 0.0, 0.0);
    sec.occurrences.push(fr);
    sec.patterns.push(OccurrencePattern {
        name: "frames".into(),
        seeds: vec!["frame".into()],
        layout: OccurrenceLayout::Linear {
            direction: [1.0, 0.0, 0.0],
            spacing: ValueExpr::input("frame_pitch"),
            count: ValueExpr::number(spec.frames_per_section as f64),
        },
        skip: vec![],
        when: None,
    });
    sec.occurrences
        .push(occ("skin", DefinitionRef::part("fuselage/skin.part", None)));
    sec.occurrences.push(occ(
        "stringer",
        DefinitionRef::part("library/stringer.part", None),
    ));
    sec.patterns.push(OccurrencePattern {
        name: "stringers".into(),
        seeds: vec!["stringer".into()],
        layout: OccurrenceLayout::Circular {
            axis_point: [mm(0.0), mm(0.0), mm(0.0)],
            axis: [1.0, 0.0, 0.0],
            count: ValueExpr::number(spec.stringers_per_section as f64),
            angle: ValueExpr::deg(360.0),
        },
        skip: vec![],
        when: None,
    });
    let d = mm(4.8);
    for j in 0..spec.joints_per_section {
        let rows = (0..spec.rows_per_joint)
            .map(|r| RowSpec {
                key: format!("r{}", r),
                offset: mm(r as f64 * 25.0),
            })
            .collect();
        sec.fasteners.push(FastenerSet {
            name: format!("lap{}", j),
            when: None,
            layout: FastenerLayout::Along(AlongRule {
                occurrence: RelPath::new(&["skin"]),
                path: PublishName::new("laps"),
                chains: vec![format!("lap{}", j)],
                spacing: Spacing::Pitch {
                    pitch: mm(spec.pitch_mm),
                },
                edge_distance: ValueExpr::binary(BinaryOp::Mul, ValueExpr::number(2.0), d.clone()),
                rows,
                stations: StationBasis::ArcLength,
                key_step: None,
            }),
            fastener: FastenerSpec {
                family: hilok.clone(),
                diameter: d.clone(),
                grip: GripSpec::Auto,
            },
            stack: vec![RelPath::new(&["skin"]), RelPath::new(&["stringer"])],
            drill: DrillSpec::default(),
            exceptions: vec![FastenerException {
                at: ExceptionSelector::Station {
                    station: mm(9.6 + spec.pitch_mm * 3.0),
                    chain: Some(format!("lap{}", j)),
                    row: Some("r0".into()),
                    tolerance: None,
                },
                action: ExceptionAction::Oversize { delta: mm(0.4) },
            }],
        });
    }

    // Top assembly.
    let top = FilePath::new("aircraft.assembly");
    let mut ac = AssemblyDocument::new(top.as_str(), "aircraft");
    ac.part_number = Some("AX-000000".into());
    ac.options = Some(OptionsBlock {
        axes: vec![
            AxisDecl {
                name: "msn".into(),
                domain: AxisDomain::Unit {
                    min: 1,
                    max: spec.msn_max,
                },
                default: Some(AxisValue::Int(1)),
            },
            AxisDecl {
                name: "engine".into(),
                domain: AxisDomain::Enum {
                    values: vec!["a".into(), "b".into()],
                },
                default: Some(AxisValue::Enum("a".into())),
            },
            AxisDecl {
                name: "freighter".into(),
                domain: AxisDomain::Bool,
                default: Some(AxisValue::Bool(false)),
            },
        ],
        rules: vec![OptionRule {
            name: "freighter_needs_engine_b".into(),
            condition: Condition::requires(
                Condition::is("freighter", AxisValue::Bool(true)),
                Condition::is_enum("engine", "b"),
            ),
        }],
        groups: vec![ChangeGroup {
            name: "ECO_0041".into(),
            effectivity: Condition::range("msn", Some(41), None),
            description: Some("heavy frames from MSN 41".into()),
        }],
        configurations: vec![
            VariantConfiguration {
                name: "launch".into(),
                values: [("msn".to_string(), AxisValue::Int(7))]
                    .into_iter()
                    .collect(),
            },
            VariantConfiguration {
                name: "freighter".into(),
                values: [
                    ("msn".to_string(), AxisValue::Int(120)),
                    ("engine".to_string(), AxisValue::Enum("b".into())),
                    ("freighter".to_string(), AxisValue::Bool(true)),
                ]
                .into_iter()
                .collect(),
            },
        ],
    });
    for i in 0..spec.sections {
        let mut s = occ(
            &format!("section{}", i + 1),
            DefinitionRef::assembly("fuselage/section.assembly"),
        );
        s.placement = at(i as f64 * l, 0.0, 0.0);
        s.required = true;
        ac.occurrences.push(s);
    }
    let mut w = occ("wing_left", DefinitionRef::part("wing/wing.part", None));
    w.placement = Placement::Attach {
        own: AnchorRef::Published {
            name: "root".into(),
        },
        to: FrameRef::Interface {
            file: FilePath::new("interfaces/fuselage-wing.part"),
            name: "wing_ref".into(),
            context: XrefContext::Pinned {
                revision: "A".into(),
            },
        },
        offset: None,
        flip: false,
    };
    ac.occurrences.push(w);
    ac.mirrors.push(OccurrenceMirror {
        name: "wing_mirror".into(),
        seeds: vec!["wing_left".into()],
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
    for (e, file) in [
        ("a", "propulsion/engine-a.part"),
        ("b", "propulsion/engine-b.part"),
    ] {
        let mut o = occ("engine_left", DefinitionRef::part(file, None));
        o.when = Some(Condition::is_enum("engine", e));
        o.placement = at(l * 0.35, 6000.0, -1500.0);
        o.required = true;
        ac.occurrences.push(o);
    }
    let mut door = occ(
        "cargo_door",
        DefinitionRef::part("fuselage/cargo-door.part", None),
    );
    door.when = Some(Condition::is("freighter", AxisValue::Bool(true)));
    door.placement = at(l * 0.2, 0.0, 0.0);
    ac.occurrences.push(door);

    let mut lock = LockSection::default();
    let mut minter = SeededMinter {
        seed: "synth".into(),
    };
    let docs = vec![frame, sec, ac];
    let assemblies = docs.iter().map(|d| d.path.clone()).collect();
    for d in &docs {
        lock_assembly(d, &mut lock, &mut minter);
    }
    for d in docs {
        src.add_assembly(d);
    }
    SynthProject {
        source: src,
        catalog,
        gauge: NominalThicknessGauge::default(),
        top,
        lock,
        assemblies,
    }
}
