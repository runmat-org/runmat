//! Semantic PMI on parts, and the drawing model's checks against the model it shows.

mod common;
use common::*;
use runmat_cad::assembly::bom::{Bom, BomItem, BomKind, BomRow};
use runmat_cad::assembly::model::{DefinitionRef, RelPath};
use runmat_cad::diagnostics::DiagCode;
use runmat_cad::drawing::*;
use runmat_cad::ids::{FilePath, PropertyName};
use runmat_cad::pmi::*;
use runmat_cad::refs::*;
use runmat_cad::units::StubUnits;
use runmat_cad::value::{NoDynamicValues, ValueExpr};
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn m01_pmi_resolves_by_reference_and_checks_semantics() {
    let env = Env::new();
    let doc = block();
    let r = env.regen(&doc);
    let pmi = PmiSet {
        datums: vec![
            DatumFeature {
                label: "A".into(),
                on: Reference::start_cap("body"),
            },
            DatumFeature {
                label: "B".into(),
                on: Reference::side("body", "s1", "l1"),
            },
        ],
        tolerances: vec![
            GeometricTolerance {
                name: "top_flat".into(),
                characteristic: Characteristic::Flatness,
                on: vec![Reference::end_cap("body")],
                tolerance: mm(0.05),
                diametral: false,
                modifier: MaterialCondition::Rfs,
                projected: None,
                datums: vec![],
            },
            GeometricTolerance {
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
            },
        ],
        sizes: vec![SizeTolerance {
            name: "thickness".into(),
            on: vec![Reference::start_cap("body"), Reference::end_cap("body")],
            size: ValueExpr::Toleranced {
                nominal: Box::new(mm(10.0)),
                minus: Box::new(mm(0.1)),
                plus: Box::new(mm(0.1)),
            },
            fit: None,
        }],
        finishes: vec![SurfaceFinish {
            on: vec![Reference::end_cap("body")],
            ra: ValueExpr::qty(0.0016, "mm"),
            process: Some("milled".into()),
            lay: None,
        }],
        ..Default::default()
    };
    let out = resolve_pmi(&doc, &pmi, &r, &StubUnits, &NoDynamicValues);
    assert!(
        out.diagnostics.errors().next().is_none(),
        "{:?}",
        out.diagnostics
    );
    assert_eq!(
        out.tolerances["top_parallel"],
        vec![vec![EntityName::generated(
            "body",
            Role::EndCap,
            SourceTag::None
        )]]
    );
    assert_eq!(out.sizes["thickness"].len(), 2);
    // Broken semantics and a broken reference.
    let mut bad = pmi.clone();
    bad.tolerances[0].datums.push(DatumReference {
        label: "A".into(),
        modifier: MaterialCondition::Rfs,
    });
    bad.tolerances[1].datums.clear();
    bad.tolerances[1].modifier = MaterialCondition::Mmc;
    bad.datums.push(DatumFeature {
        label: "A".into(),
        on: Reference::end_cap("missing"),
    });
    let out = resolve_pmi(&doc, &bad, &r, &StubUnits, &NoDynamicValues);
    assert_eq!(
        out.diagnostics.with_code(DiagCode::PmiInvalid).len(),
        3,
        "{:?}",
        out.diagnostics
    );
    assert_eq!(out.diagnostics.with_code(DiagCode::PmiReference).len(), 1);
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

#[test]
fn d01_drawings_check_views_annotations_tables_and_title_blocks() {
    let d = drawing();
    let rep = check_drawing(&d, &Model);
    assert!(
        rep.diagnostics.errors().next().is_none(),
        "{:?}",
        rep.diagnostics
    );
    assert_eq!(
        rep.balloons["b1"], "2",
        "balloon numbers follow the BOM table"
    );
    assert_eq!(rep.resolved["d1"].len(), 2);
    let tb = &rep.title_blocks["1"];
    assert_eq!(tb[0], ("Part number".to_string(), "BR-1".to_string()));
    assert_eq!(tb[1].1, "C");
    let warnings: Vec<&str> = rep
        .diagnostics
        .with_code(DiagCode::DrawingReference)
        .iter()
        .map(|d| d.message.as_str())
        .collect();
    assert_eq!(
        warnings,
        vec![
            "table `bom` shows `description`, which 2 item(s) do not have",
            "title block `Drawn` reads `drawn_by`, which the drawn model does not have",
        ],
        "the BOM table's description column and `drawn_by` are missing (warnings)"
    );
}

#[test]
fn d02_broken_references_are_reported_with_view_and_annotation() {
    let mut d = drawing();
    d.annotations.push(Annotation {
        id: "d2".into(),
        view: "b".into(),
        kind: AnnotationKind::Dimension {
            style: DimensionStyle::Radius,
            references: vec![Reference::end_cap("cut1")],
            pmi: None,
            group: None,
            reference: true,
        },
        at: [0.0, 0.0],
    });
    d.annotations.push(Annotation {
        id: "g2".into(),
        view: "front".into(),
        kind: AnnotationKind::Gdt {
            tolerance: "nope".into(),
        },
        at: [0.0, 0.0],
    });
    d.views.push(View {
        name: "loop".into(),
        sheet: "1".into(),
        model: d.views[0].model.clone(),
        kind: ViewKind::Projected {
            from: "loop".into(),
            direction: ProjectDirection::Left,
        },
        at: [0.0, 0.0],
        scale: None,
        display: Display::VisibleLines,
        exclude: vec![],
        exploded: None,
    });
    let rep = check_drawing(&d, &Model);
    let refs: Vec<&str> = rep
        .diagnostics
        .with_code(DiagCode::DrawingReference)
        .iter()
        .map(|x| x.message.as_str())
        .collect();
    assert!(
        refs.iter()
            .any(|m| m.contains("`d2`") && m.contains("consumed by")),
        "{:?}",
        refs
    );
    assert!(refs.iter().any(|m| m.contains("`nope`")));
    assert!(
        !rep.resolved.contains_key("d2"),
        "dangling annotations are kept but unresolved"
    );
    assert_eq!(rep.diagnostics.with_code(DiagCode::DrawingView).len(), 1);
}
