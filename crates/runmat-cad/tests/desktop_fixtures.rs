//! Desktop fixtures: real JSON projections of a part, an assembly and a drawing, generated from
//! this crate's test models, for the Desktop's Outline providers and their tests.
//!
//! The Desktop checks them in under `desktop/src/app/components/cad/fixtures/`. Set
//! `RUNMAT_DESKTOP_FIXTURES` to that folder to write them; with it unset, the test compares
//! against the sibling Desktop checkout when there is one.

mod common;
use common::*;
use runmat_cad::assembly::bom::{Bom, BomItem, BomKind, BomRow};
use runmat_cad::assembly::index::{assembly_index_json, build_assembly_index};
use runmat_cad::assembly::model::{DefinitionRef, RelPath};
use runmat_cad::assembly::resolve::{resolve, ResolveContext, ResolveRequest};
use runmat_cad::assembly::synth::{generate, SynthSpec};
use runmat_cad::drawing::index::{build_drawing_index, to_json as drawing_json};
use runmat_cad::drawing::*;
use runmat_cad::ids::{FilePath, PropertyName};
use runmat_cad::index::{build_index, to_json};
use runmat_cad::model::*;
use runmat_cad::refs::*;
use runmat_cad::units::StubUnits;
use runmat_cad::value::NoDynamicValues;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

fn out_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("RUNMAT_DESKTOP_FIXTURES") {
        return Some(PathBuf::from(dir));
    }
    let sibling = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../runmat-private-cad/desktop/src/app/components/cad/fixtures");
    sibling.exists().then_some(sibling)
}

/// Writes the fixture when asked to, else checks the checked-in copy is current.
fn emit(name: &str, json: &str) {
    let Some(dir) = out_dir() else { return };
    let path = dir.join(name);
    if std::env::var("RUNMAT_DESKTOP_FIXTURES").is_ok() {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, json).unwrap();
        return;
    }
    let checked_in = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(
        checked_in == json,
        "{} is stale; regenerate with RUNMAT_DESKTOP_FIXTURES={} cargo test -p runmat-cad --test desktop_fixtures",
        path.display(),
        dir.display()
    );
}

/// A bracket: block, cut, hole pattern, fillet, a fillet that fails on a missing edge, a
/// suppressed chamfer, regions, a publish and a second configuration.
fn bracket() -> PartDocument {
    let mut d = block();
    d.path = "bracket.part".into();
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
    d.features.push(fillet(
        "round",
        vec![Reference::edges_between(
            side("body", "s1", "l1"),
            Reference::end_cap("body"),
        )],
        1.5,
    ));
    d.features.push(fillet(
        "round_missing",
        vec![Reference::edges_between(
            side("body", "s1", "l9"),
            Reference::end_cap("body"),
        )],
        1.0,
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
    d.configurations.push(Configuration {
        name: "long".into(),
        ..Default::default()
    });
    d
}

#[test]
fn part_fixture() {
    let env = Env::new();
    let d = bracket();
    let r = env.regen(&d);
    let index = build_index(&d, &r, None);
    // The fixture shows every node state the tree draws.
    assert!(
        !index.diagnostics.is_empty(),
        "the bracket carries a failing fillet"
    );
    let mut json = String::from("{\n\"document\": ");
    json.push_str(&serde_json::to_string_pretty(&d).unwrap());
    json.push_str(",\n\"index\": ");
    json.push_str(to_json(&index).trim_end());
    json.push_str("\n}\n");
    serde_json::from_str::<serde_json::Value>(&json).unwrap();
    emit("bracket.part.json", &json);
}

#[test]
fn assembly_fixture() {
    let mut p = generate(&SynthSpec::small());
    add_frame_mates(&mut p.source);
    let units = StubUnits;
    let dynamic = NoDynamicValues;
    let ctx = ResolveContext {
        source: &p.source,
        units: &units,
        dynamic: &dynamic,
        lock: Some(&p.lock),
        catalog: Some(&p.catalog),
        tables: None,
        gauge: Some(&p.gauge),
    };
    let r = resolve(&p.top, &ResolveRequest::default(), &ctx);
    let index = build_assembly_index(&r, Some(&p.source));
    assert!(!index.occurrences.is_empty());
    let statuses: BTreeSet<String> = index
        .mates
        .iter()
        .map(|m| {
            format!("{:?}", m.status)
                .split([' ', '{'])
                .next()
                .unwrap()
                .to_owned()
        })
        .collect();
    assert!(
        statuses.contains("Conflicting") && statuses.contains("Failed"),
        "the Mates folder shows an over-constrained and a failed mate: {statuses:?}"
    );
    emit("rig.assembly.json", &assembly_index_json(&index));
}

/// The frame's web and first clip as mate connectors: the clip fastened to the web although
/// its placement already fixes it (over-constrained), and a hinge naming a connector that was
/// renamed away (failed, with its repair).
fn add_frame_mates(source: &mut runmat_cad::assembly::source::MemorySource) {
    use runmat_cad::assembly::model::*;
    let key = source
        .assemblies
        .keys()
        .find(|(path, _)| path.as_str() == "fuselage/frame.assembly")
        .cloned()
        .expect("synth frame assembly");
    let mut frame = (*source.assemblies[&key]).clone();
    let connector = |name: &str, occurrence: &str| MateConnector {
        name: name.into(),
        occurrence: RelPath::new(&[occurrence]),
        anchor: AnchorRef::Origin,
        offset: None,
        flip: false,
    };
    frame.connectors.push(connector("web_root", "web"));
    frame.connectors.push(connector("clip_seat", "clip"));
    let mate = |name: &str, kind: MateKind, a: &str, b: &str| MateDef {
        name: name.into(),
        kind,
        a: a.into(),
        b: b.into(),
        when: None,
        suppressed: false,
    };
    frame.mates.push(mate(
        "clip_to_web",
        MateKind::Fasten,
        "clip_seat",
        "web_root",
    ));
    frame.mates.push(mate(
        "clip_hinge",
        MateKind::Revolute { angle: None },
        "clip_seat",
        "web_flange",
    ));
    source.assemblies.insert(key, std::sync::Arc::new(frame));
}

#[test]
fn drawing_fixture() {
    let index = build_drawing_index(&drawing(), &Model);
    assert!(!index.document.views.is_empty());
    emit("bracket.drawing.json", &drawing_json(&index));
}

struct Model;

impl ModelReferences for Model {
    fn resolve(&self, _: &ModelRef, r: &Reference) -> Result<Vec<Vec<EntityName>>, String> {
        match r {
            Reference::Name { name, .. } if name.maker().0 == "body" => {
                Ok(vec![vec![name.clone()]])
            }
            Reference::EdgesBetween { .. } => Ok(vec![vec![]]),
            _ => Err("is consumed by `cut2`".into()),
        }
    }
    fn property(&self, _: &ModelRef, name: &PropertyName) -> Option<String> {
        match name.as_str() {
            "part_number" => Some("BR-1".into()),
            "revision" => Some("C".into()),
            _ => None,
        }
    }
    fn occurrence(&self, _: &ModelRef, path: &RelPath) -> bool {
        path.to_string() == "bolts/b1"
    }
    fn pmi(&self, _: &ModelRef) -> (BTreeSet<String>, BTreeSet<String>, BTreeSet<String>) {
        (
            ["top_parallel".to_string()].into(),
            ["thickness".to_string()].into(),
            ["A".to_string()].into(),
        )
    }
    fn bom(&self, _: &ModelRef) -> Option<Bom> {
        let item = |pn: &str| BomItem {
            kind: BomKind::Part,
            part_number: Some(pn.into()),
            definition: DefinitionRef::part("x.part", None),
            revision: None,
            configuration: "default".into(),
            material: None,
        };
        Some(Bom {
            variant: BTreeMap::new(),
            rows: vec![
                BomRow {
                    item: item("BR-1"),
                    quantity: 1,
                    paths: vec!["bracket".into()],
                },
                BomRow {
                    item: item("BT-6"),
                    quantity: 2,
                    paths: vec!["bolts/b1".into(), "bolts/b2".into()],
                },
            ],
            total_parts: 3,
            total_fasteners: 0,
        })
    }
}

fn drawing() -> DrawingDocument {
    let model = ModelRef {
        file: FilePath::new("rig.assembly"),
        configuration: None,
        variant: BTreeMap::new(),
    };
    let view = |name: &str, kind: ViewKind| View {
        name: name.into(),
        sheet: "1".into(),
        model: model.clone(),
        kind,
        at: [100.0, 100.0],
        scale: None,
        display: Display::HiddenLines,
        exclude: vec![],
        exploded: None,
    };
    DrawingDocument {
        path: FilePath::new("rig.drawing"),
        header: Default::default(),
        formats: vec![SheetFormat {
            name: "a3".into(),
            size: SheetSize::Iso { name: "A3".into() },
            title_block: vec![
                TitleField {
                    label: "Part number".into(),
                    source: FieldSource::Property {
                        name: "part_number".into(),
                    },
                },
                TitleField {
                    label: "Rev".into(),
                    source: FieldSource::Revision,
                },
                TitleField {
                    label: "Sheet".into(),
                    source: FieldSource::SheetNumber,
                },
                TitleField {
                    label: "Drawn".into(),
                    source: FieldSource::Property {
                        name: "drawn_by".into(),
                    },
                },
            ],
        }],
        sheets: vec![Sheet {
            name: "1".into(),
            format: "a3".into(),
            scale: (1, 2),
            projection: Projection::ThirdAngle,
        }],
        views: vec![
            view(
                "front",
                ViewKind::Standard {
                    orientation: Orientation::Front,
                    frame: None,
                },
            ),
            view(
                "top",
                ViewKind::Projected {
                    from: "front".into(),
                    direction: ProjectDirection::Up,
                },
            ),
            view(
                "a_a",
                ViewKind::Section {
                    from: "front".into(),
                    plane: Reference::start_cap("body"),
                    label: "A".into(),
                    aligned: false,
                },
            ),
            view(
                "b",
                ViewKind::Detail {
                    from: "a_a".into(),
                    center: Reference::end_cap("body"),
                    radius: mm(20.0),
                    label: "B".into(),
                },
            ),
        ],
        annotations: vec![
            Annotation {
                id: "d1".into(),
                view: "front".into(),
                kind: AnnotationKind::Dimension {
                    style: DimensionStyle::Linear,
                    references: vec![Reference::start_cap("body"), Reference::end_cap("body")],
                    pmi: Some("thickness".into()),
                    group: None,
                    reference: false,
                },
                at: [0.0, 0.0],
            },
            Annotation {
                id: "g1".into(),
                view: "front".into(),
                kind: AnnotationKind::Gdt {
                    tolerance: "top_parallel".into(),
                },
                at: [0.0, 0.0],
            },
            Annotation {
                id: "b1".into(),
                view: "front".into(),
                kind: AnnotationKind::Balloon {
                    occurrence: RelPath::new(&["bolts", "b1"]),
                    table: "bom".into(),
                },
                at: [0.0, 0.0],
            },
            Annotation {
                id: "n1".into(),
                view: "top".into(),
                kind: AnnotationKind::Note {
                    text: "PART {part_number} REV {revision}".into(),
                    leader: None,
                },
                at: [0.0, 0.0],
            },
        ],
        tables: vec![DrawingTable {
            name: "bom".into(),
            sheet: "1".into(),
            kind: TableKind::Bom {
                view: "front".into(),
                include_fasteners: true,
                spec: Default::default(),
            },
            at: [0.0, 0.0],
        }],
    }
}
