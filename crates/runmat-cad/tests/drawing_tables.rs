//! Drawing BOM tables, hole tables through the geometry seam, exploded views on drawing
//! views, and three-way drawing merges by stable id.

use runmat_cad::assembly::bom::{Bom, BomItem, BomKind, BomRow, IndentedRow};
use runmat_cad::assembly::model::{DefinitionRef, RelPath};
use runmat_cad::diagnostics::{DiagCode, Severity};
use runmat_cad::drawing::bom_table::*;
use runmat_cad::drawing::hole_table::*;
use runmat_cad::drawing::merge::*;
use runmat_cad::drawing::*;
use runmat_cad::ids::{FeatureId, FilePath, MaterialRef, PropertyName};
use runmat_cad::model::datum::Frame;
use runmat_cad::refs::*;
use runmat_cad::value::ValueExpr;
use std::collections::{BTreeMap, BTreeSet};

fn item(kind: BomKind, file: &str, pn: &str, material: Option<&str>) -> BomItem {
    BomItem {
        kind,
        part_number: Some(pn.into()),
        definition: DefinitionRef::part(file, None),
        revision: None,
        configuration: "default".into(),
        material: material.map(MaterialRef::new),
    }
}

fn flat() -> Bom {
    let row = |i: BomItem, q: u64, paths: &[&str]| BomRow {
        item: i,
        quantity: q,
        paths: paths.iter().map(|p| p.to_string()).collect(),
    };
    Bom {
        variant: BTreeMap::new(),
        rows: vec![
            row(
                item(BomKind::Part, "plate.part", "PL-200", Some("al_6061")),
                1,
                &["plate"],
            ),
            row(
                item(BomKind::Part, "bracket.part", "BR-100", Some("steel_1018")),
                2,
                &["bracket.left", "bracket.right"],
            ),
            row(
                item(BomKind::Part, "pin.part", "PN-010", Some("steel_1018")),
                4,
                &["pins/p1", "pins/p2", "pins/p3", "pins/p4"],
            ),
            row(
                item(BomKind::Fastener, "bolts.part", "M6x20", None),
                8,
                &["fasteners.mount"],
            ),
        ],
        total_parts: 7,
        total_fasteners: 8,
    }
}

struct Props;

impl BomProperties for Props {
    fn property(&self, item: &BomItem, name: &PropertyName) -> Option<String> {
        match (item.part_number.as_deref(), name.as_str()) {
            (Some("PL-200"), "description") => Some("Base plate".into()),
            (Some("BR-100"), "description") => Some("Bracket".into()),
            (Some("M6x20"), "description") => Some("Hex bolt".into()),
            _ => None,
        }
    }
}

fn cells(d: &BomTableData) -> Vec<Vec<String>> {
    d.rows().map(|r| r.cells.clone()).collect()
}

fn strs(v: &[&[&str]]) -> Vec<Vec<String>> {
    v.iter()
        .map(|r| r.iter().map(|c| c.to_string()).collect())
        .collect()
}

#[test]
fn t01_flat_bom_tables_sort_number_and_keep_numbers() {
    let mut spec = BomTableSpec {
        sort: vec![BomSort {
            column: BomColumn::PartNumber,
            descending: false,
        }],
        exclude_kinds: vec![BomKind::Fastener],
        ..Default::default()
    };
    let d = render_bom_table("bom", &spec, BomSource::Flat(&flat()), &Props);
    let titles: Vec<&str> = d.columns.iter().map(|c| c.title.as_str()).collect();
    assert_eq!(
        titles,
        ["ITEM", "PART NUMBER", "DESCRIPTION", "MATERIAL", "QTY"]
    );
    assert_eq!(d.columns[4].align, Align::Right);
    assert_eq!(
        cells(&d),
        strs(&[
            &["1", "BR-100", "Bracket", "Steel 1018", "2"],
            &["2", "PL-200", "Base plate", "Al 6061", "1"],
            &["3", "PN-010", "", "Steel 1018", "4"],
        ])
    );
    assert_eq!(d.total_quantity, 7);
    // The pin has no description: one warning, naming the column and the count.
    let w = d.diagnostics.with_code(DiagCode::DrawingReference);
    assert_eq!(w.len(), 1);
    assert_eq!(w[0].severity, Severity::Warning);
    assert_eq!(
        w[0].message,
        "table `bom` shows `description`, which 1 item(s) do not have"
    );
    assert_eq!(d.number_of_path("pins/p3"), Some("3"));

    // Kept numbers survive a re-sort; new items take the free numbers in display order.
    let pin_key = bom_item_key(&flat().rows[2].item);
    spec.numbering.keep.insert(pin_key.clone(), 1);
    spec.sort[0].descending = true;
    let d = render_bom_table("bom", &spec, BomSource::Flat(&flat()), &Props);
    let numbers: Vec<(String, String)> = d
        .rows()
        .map(|r| (r.item.part_number.clone().unwrap(), r.number.clone()))
        .collect();
    assert_eq!(
        numbers,
        vec![
            ("PN-010".into(), "1".into()),
            ("PL-200".into(), "2".into()),
            ("BR-100".into(), "3".into()),
        ]
    );
    assert_eq!(d.numbers[&pin_key], "1");
    // Two items kept on one number is an error.
    spec.numbering
        .keep
        .insert(bom_item_key(&flat().rows[0].item), 1);
    let d = render_bom_table("bom", &spec, BomSource::Flat(&flat()), &Props);
    assert_eq!(d.diagnostics.with_code(DiagCode::DrawingInvalid).len(), 1);
}

#[test]
fn t02_grouped_and_property_columns() {
    let spec = BomTableSpec {
        columns: vec![
            BomColumnSpec::new(BomColumn::Item),
            BomColumnSpec {
                column: BomColumn::Property {
                    name: PropertyName::new("description"),
                },
                title: Some("Name".into()),
            },
            BomColumnSpec::new(BomColumn::Kind),
            BomColumnSpec::new(BomColumn::Quantity),
        ],
        group_by: Some(BomColumn::Material),
        ..Default::default()
    };
    let d = render_bom_table("bom", &spec, BomSource::Flat(&flat()), &Props);
    let shape: Vec<String> = d
        .lines
        .iter()
        .map(|l| match l {
            BomTableLine::Group { title, rows } => format!("[{}:{}]", title, rows),
            BomTableLine::Row(r) => r.cells.join("|"),
        })
        .collect();
    assert_eq!(
        shape,
        vec![
            "[:1]",
            "1|Hex bolt|fastener|8",
            "[Al 6061:1]",
            "2|Base plate|part|1",
            "[Steel 1018:2]",
            "3|Bracket|part|2",
            "4||part|4",
        ]
    );
    assert_eq!(d.columns[1].title, "Name");
    assert_eq!(d.columns[1].key, "property.description");
}

fn indented() -> Vec<IndentedRow> {
    let r = |level: u32, parent: &str, i: BomItem, q: u64| IndentedRow {
        level,
        parent: parent.into(),
        item: i,
        quantity: q,
    };
    let sub = BomItem {
        kind: BomKind::Assembly,
        part_number: Some("ARM-1".into()),
        definition: DefinitionRef::assembly("arm.assembly"),
        revision: None,
        configuration: "default".into(),
        material: None,
    };
    vec![
        r(1, "", item(BomKind::Part, "plate.part", "PL-200", None), 1),
        r(1, "", sub, 2),
        r(
            2,
            "arm.left",
            item(BomKind::Part, "link.part", "LK-5", None),
            2,
        ),
        r(
            2,
            "arm.left",
            item(BomKind::Fastener, "bolts.part", "M6x20", None),
            4,
        ),
        r(
            2,
            "arm.left",
            item(BomKind::Part, "bush.part", "BU-1", None),
            1,
        ),
    ]
}

#[test]
fn t03_indented_and_top_level_tables() {
    let spec = BomTableSpec {
        structure: BomStructure::Indented,
        columns: vec![
            BomColumnSpec::new(BomColumn::Item),
            BomColumnSpec::new(BomColumn::PartNumber),
            BomColumnSpec::new(BomColumn::Level),
            BomColumnSpec::new(BomColumn::Quantity),
        ],
        sort: vec![BomSort {
            column: BomColumn::PartNumber,
            descending: false,
        }],
        exclude_kinds: vec![BomKind::Fastener],
        ..Default::default()
    };
    let rows = indented();
    let d = render_bom_table("bom", &spec, BomSource::Indented(&rows), &NoBomProperties);
    assert_eq!(
        cells(&d),
        strs(&[
            &["1", "ARM-1", "1", "2"],
            &["1.1", "BU-1", "2", "1"],
            &["1.2", "LK-5", "2", "2"],
            &["2", "PL-200", "1", "1"],
        ])
    );
    let top = BomTableSpec {
        structure: BomStructure::TopLevel,
        ..spec.clone()
    };
    let d = render_bom_table("bom", &top, BomSource::Indented(&rows), &NoBomProperties);
    assert_eq!(d.rows().count(), 2);
    // Parts-only lists need the flat BOM; indented ones need the indented BOM.
    let parts = BomTableSpec::default();
    let d = render_bom_table("bom", &parts, BomSource::Indented(&rows), &NoBomProperties);
    assert_eq!(
        d.diagnostics.diagnostics[0].message,
        "table `bom` lists parts only, which needs the model's flat BOM (indented quantities are per parent)"
    );
    let d = render_bom_table("bom", &spec, BomSource::Flat(&flat()), &NoBomProperties);
    assert!(d.diagnostics.diagnostics[0]
        .message
        .contains("needs the model's indented BOM"));
}

// ------------------------------------------------------------------------------ holes

struct FakeHoles {
    holes: Vec<HoleInstance>,
}

fn hole(feature: &str, key: &str, x: f64, y: f64, d: f64) -> HoleInstance {
    HoleInstance {
        feature: FeatureId::new(feature),
        key: key.into(),
        position: [x, y, 0.01],
        axis: [0.0, 0.0, -1.0],
        diameter: d,
        depth: None,
        thread: None,
        counterbore: None,
        countersink: None,
    }
}

impl FakeHoles {
    fn plate() -> Self {
        let mut holes = vec![
            hole("mount", "nw", 0.010, 0.090, 0.0066),
            hole("mount", "ne", 0.110, 0.090, 0.0066),
            hole("mount", "sw", 0.010, 0.010, 0.0066),
            hole("mount", "se", 0.110, 0.010, 0.0066),
        ];
        for (k, x) in [("t1", 0.040), ("t2", 0.080)] {
            let mut h = hole("tapped", k, x, 0.050, 0.0068);
            h.depth = Some(0.015);
            h.thread = Some(ThreadCallout {
                designation: "M8x1.25".into(),
                class: Some("6H".into()),
                depth: Some(0.012),
            });
            holes.push(h);
        }
        let mut cb = hole("cbore", "c1", 0.060, 0.020, 0.0066);
        cb.counterbore = Some((0.011, 0.0065));
        holes.push(cb);
        let mut cs = hole("csink", "k1", 0.060, 0.080, 0.0045);
        cs.countersink = Some((0.009, 90f64.to_radians()));
        holes.push(cs);
        // A cross hole along X cannot be located in the top view.
        let mut side = hole("side", "s1", 0.0, 0.05, 0.005);
        side.axis = [1.0, 0.0, 0.0];
        holes.push(side);
        FakeHoles { holes }
    }
}

impl HoleGeometry for FakeHoles {
    fn holes(&self, _model: &ModelRef) -> Result<Vec<HoleInstance>, String> {
        Ok(self.holes.clone())
    }
    fn view_frame(&self, _view: &View) -> Result<Frame, String> {
        Ok(Frame::WORLD)
    }
    fn point(&self, _model: &ModelRef, r: &Reference) -> Result<[f64; 3], String> {
        match r {
            Reference::Name { name, .. } if name.maker().0 == "body" => Ok([0.010, 0.010, 0.01]),
            _ => Err("is not a point of the model".into()),
        }
    }
}

fn model() -> ModelRef {
    ModelRef {
        file: FilePath::new("plate.part"),
        configuration: None,
        variant: BTreeMap::new(),
    }
}

fn top_view() -> View {
    View {
        name: "top".into(),
        sheet: "1".into(),
        model: model(),
        kind: ViewKind::Standard {
            orientation: Orientation::Top,
            frame: None,
        },
        at: [0.0, 0.0],
        scale: None,
        display: Display::VisibleLines,
        exclude: vec![],
        exploded: None,
    }
}

#[test]
fn t04_hole_tables_tag_locate_and_call_out() {
    let g = FakeHoles::plate();
    let origin = Reference::start_cap("body");
    let d = hole_table("holes", &HoleTableSpec::default(), &top_view(), &origin, &g);
    let rows: Vec<(String, String, String, String)> = d
        .rows
        .iter()
        .map(|r| {
            (
                r.tag.clone(),
                r.x_text.clone(),
                r.y_text.clone(),
                r.size.clone(),
            )
        })
        .collect();
    let row = |t: &str, x: &str, y: &str, s: &str| (t.into(), x.into(), y.into(), s.into());
    assert_eq!(
        rows,
        vec![
            row("B1", "100.00", "80.00", "Ø6.60 THRU"),
            row("B2", "0.00", "80.00", "Ø6.60 THRU"),
            row("B3", "100.00", "0.00", "Ø6.60 THRU"),
            row("B4", "0.00", "0.00", "Ø6.60 THRU"),
            row("D1", "30.00", "40.00", "Ø6.80 ↧ 15.00, M8x1.25-6H ↧ 12.00"),
            row("D2", "70.00", "40.00", "Ø6.80 ↧ 15.00, M8x1.25-6H ↧ 12.00"),
            row("C1", "50.00", "10.00", "Ø6.60 THRU, ⌴ Ø11.00 ↧ 6.50"),
            row("A1", "50.00", "70.00", "Ø4.50 THRU, ⌵ Ø9.00 × 90°"),
        ]
    );
    let sizes: Vec<(String, u32)> = d
        .sizes
        .iter()
        .map(|s| (s.letter.clone(), s.quantity))
        .collect();
    assert_eq!(
        sizes,
        vec![
            ("A".into(), 1),
            ("B".into(), 4),
            ("C".into(), 1),
            ("D".into(), 2)
        ]
    );
    let w = d.diagnostics.with_code(DiagCode::DrawingHoleTable);
    assert_eq!(w.len(), 1);
    assert_eq!(w[0].severity, Severity::Warning);
    assert_eq!(
        w[0].message,
        "hole table `holes` leaves out 1 hole(s) not along the view direction: side:s1"
    );

    // Sequential tags sorted by X then Y, in inches, one feature only, a kept tag.
    let mut spec = HoleTableSpec {
        features: vec![FeatureId::new("mount")],
        tagging: HoleTagging::Sequential,
        sort: HoleSort::X,
        units: LengthUnit::Inch,
        decimals: 3,
        keep: BTreeMap::new(),
    };
    spec.keep.insert("mount:se".into(), "H1".into());
    let d = hole_table("holes", &spec, &top_view(), &origin, &g);
    let rows: Vec<(String, String, String, String)> = d
        .rows
        .iter()
        .map(|r| {
            (
                r.tag.clone(),
                r.key.clone(),
                r.x_text.clone(),
                r.y_text.clone(),
            )
        })
        .collect();
    assert_eq!(
        rows,
        vec![
            ("H2".into(), "sw".into(), "0.000".into(), "0.000".into()),
            ("H3".into(), "nw".into(), "0.000".into(), "3.150".into()),
            ("H1".into(), "se".into(), "3.937".into(), "0.000".into()),
            ("H4".into(), "ne".into(), "3.937".into(), "3.150".into()),
        ]
    );
    assert_eq!(d.rows[0].size, "Ø0.260 THRU");
    assert!(d.sizes.is_empty());
    // Unknown features and origins are specific.
    spec.features.push(FeatureId::new("nope"));
    let d = hole_table("holes", &spec, &top_view(), &origin, &g);
    assert_eq!(
        d.diagnostics.with_code(DiagCode::DrawingHoleTable)[0].message,
        "hole table `holes` lists `nope`, which makes no holes in `plate.part`"
    );
    let d = hole_table(
        "holes",
        &HoleTableSpec::default(),
        &top_view(),
        &Reference::start_cap("cut3"),
        &g,
    );
    assert_eq!(
        d.diagnostics.diagnostics[0].message,
        format!(
            "hole table `holes` origin `{}` is not a point of the model",
            Reference::start_cap("cut3")
        )
    );
}

// ------------------------------------------------------------------------------ the drawing

struct Model {
    exploded: BTreeSet<String>,
}

impl ModelReferences for Model {
    fn resolve(&self, _: &ModelRef, r: &Reference) -> Result<Vec<Vec<EntityName>>, String> {
        match r {
            Reference::Name { name, .. } if name.maker().0 == "body" => {
                Ok(vec![vec![name.clone()]])
            }
            _ => Err("is consumed by `cut2`".into()),
        }
    }
    fn property(&self, _: &ModelRef, _: &PropertyName) -> Option<String> {
        None
    }
    fn occurrence(&self, _: &ModelRef, path: &RelPath) -> bool {
        path.to_string().starts_with("pins/") || path.to_string() == "plate"
    }
    fn pmi(&self, _: &ModelRef) -> (BTreeSet<String>, BTreeSet<String>, BTreeSet<String>) {
        Default::default()
    }
    fn bom(&self, _: &ModelRef) -> Option<Bom> {
        Some(flat())
    }
    fn exploded_views(&self, _: &ModelRef) -> BTreeSet<String> {
        self.exploded.clone()
    }
    fn item_property(&self, _: &ModelRef, item: &BomItem, name: &PropertyName) -> Option<String> {
        Props.property(item, name)
    }
}

fn annotation(id: &str, view: &str, kind: AnnotationKind) -> Annotation {
    Annotation {
        id: id.into(),
        view: view.into(),
        kind,
        at: [0.0, 0.0],
    }
}

fn dim(refs: Vec<Reference>) -> AnnotationKind {
    AnnotationKind::Dimension {
        style: DimensionStyle::Linear,
        references: refs,
        pmi: None,
        group: None,
        reference: false,
        condition: Default::default(),
    }
}

fn drawing() -> DrawingDocument {
    let mut iso = top_view();
    iso.name = "iso".into();
    iso.model.file = FilePath::new("rig.assembly");
    iso.kind = ViewKind::Standard {
        orientation: Orientation::Isometric,
        frame: None,
    };
    iso.exploded = Some("service".into());
    let mut detail = top_view();
    detail.name = "b".into();
    detail.kind = ViewKind::Detail {
        from: "top".into(),
        center: Reference::end_cap("body"),
        radius: ValueExpr::mm(20.0),
        label: "B".into(),
    };
    DrawingDocument {
        path: FilePath::new("rig.drawing"),
        header: Default::default(),
        formats: vec![SheetFormat {
            standard: Default::default(),
            name: "a3".into(),
            size: SheetSize::Iso { name: "A3".into() },
            title_block: vec![],
        }],
        sheets: vec![Sheet {
            name: "1".into(),
            format: "a3".into(),
            scale: (1, 2),
            projection: Projection::ThirdAngle,
        }],
        views: vec![top_view(), iso, detail],
        annotations: vec![
            annotation(
                "d1",
                "top",
                dim(vec![
                    Reference::start_cap("body"),
                    Reference::end_cap("body"),
                ]),
            ),
            annotation(
                "g1",
                "top",
                AnnotationKind::DatumSymbol { label: "A".into() },
            ),
            annotation(
                "p1",
                "iso",
                AnnotationKind::Balloon {
                    occurrence: RelPath::new(&["pins", "p3"]),
                    table: "bom".into(),
                },
            ),
        ],
        tables: vec![
            DrawingTable {
                name: "bom".into(),
                sheet: "1".into(),
                kind: TableKind::Bom {
                    view: "iso".into(),
                    include_fasteners: false,
                    spec: BomTableSpec {
                        sort: vec![BomSort {
                            column: BomColumn::PartNumber,
                            descending: false,
                        }],
                        ..Default::default()
                    },
                },
                at: [0.0, 0.0],
            },
            DrawingTable {
                name: "holes".into(),
                sheet: "1".into(),
                kind: TableKind::Hole {
                    view: "top".into(),
                    origin: Reference::start_cap("body"),
                    spec: HoleTableSpec::default(),
                },
                at: [0.0, 0.0],
            },
        ],
    }
}

#[test]
fn t05_drawings_render_tables_number_balloons_and_check_exploded_views() {
    let d = drawing();
    let m = Model {
        exploded: ["service".to_string()].into(),
    };
    let rep = check_drawing_with(&d, &m, Some(&FakeHoles::plate()));
    let errors: Vec<String> = rep
        .diagnostics
        .errors()
        .map(|e| e.message.clone())
        .collect();
    // The datum symbol names a datum the (empty) PMI does not declare: the only error.
    assert_eq!(
        errors,
        vec!["datum `A` is not declared in the model's PMI".to_string()]
    );
    assert_eq!(rep.balloons["p1"], "3", "the table's sorted numbering");
    assert_eq!(
        rep.bom_tables["bom"].rows().count(),
        3,
        "fasteners left out"
    );
    assert_eq!(rep.hole_tables["holes"].rows.len(), 8);
    // Without hole geometry there are no hole rows, and nothing else changes.
    let plain = check_drawing(&d, &m);
    assert!(plain.hole_tables.is_empty());
    assert_eq!(plain.balloons, rep.balloons);
    // A view showing an exploded state the model does not declare.
    let none = Model {
        exploded: BTreeSet::new(),
    };
    let rep = check_drawing(&d, &none);
    assert!(rep.diagnostics.errors().any(|e| e.message
        == "view `iso` shows exploded view `service`, which `rig.assembly` does not declare"));
    // Serde: table specs default when absent, and round-trip.
    let json = serde_json::to_string(&d).unwrap();
    let back: DrawingDocument = serde_json::from_str(&json).unwrap();
    assert_eq!(back, d);
    let old = r#"{"table":"bom","view":"iso","include_fasteners":true}"#;
    let k: TableKind = serde_json::from_str(old).unwrap();
    assert!(matches!(k, TableKind::Bom { spec, .. } if spec == BomTableSpec::default()));
}

// ------------------------------------------------------------------------------ merge

fn moved(d: &mut DrawingDocument, id: &str, at: [f64; 2]) {
    d.annotations.iter_mut().find(|a| a.id == id).unwrap().at = at;
}

#[test]
fn t06_drawing_merges_field_by_field_by_stable_id() {
    let base = drawing();
    let mut ours = base.clone();
    let mut theirs = base.clone();
    // Ours moves d1; theirs re-picks its references: both apply.
    moved(&mut ours, "d1", [10.0, 20.0]);
    theirs.annotations[0].kind = dim(vec![
        Reference::start_cap("body"),
        Reference::side("body", "s1", "l2"),
    ]);
    // Theirs adds a note after g1; ours adds a table.
    theirs.annotations.insert(
        2,
        annotation(
            "n3",
            "top",
            AnnotationKind::Note {
                text: "BREAK EDGES".into(),
                leader: None,
            },
        ),
    );
    ours.tables.push(DrawingTable {
        name: "revisions".into(),
        sheet: "1".into(),
        kind: TableKind::Revision { rows: Vec::new() },
        at: [5.0, 5.0],
    });
    let m = merge_drawings(&base, &ours, &theirs, None);
    assert!(m.conflicts.is_empty(), "{:#?}", m.conflicts);
    assert!(m.diagnostics.diagnostics.is_empty(), "{:#?}", m.diagnostics);
    let d1 = &m.merged.annotations[0];
    assert_eq!(d1.at, [10.0, 20.0]);
    assert_eq!(d1.kind, theirs.annotations[0].kind);
    let ids: Vec<&str> = m.merged.annotations.iter().map(|a| a.id.as_str()).collect();
    assert_eq!(ids, ["d1", "g1", "n3", "p1"]);
    assert_eq!(m.merged.tables.len(), 3);
    // Merging is symmetric in content when nothing conflicts.
    let r = merge_drawings(&base, &theirs, &ours, None);
    assert_eq!(r.merged.annotations[0], m.merged.annotations[0]);
}

#[test]
fn t07_drawing_merge_conflicts_and_re_resolution() {
    let base = drawing();
    let mut ours = base.clone();
    let mut theirs = base.clone();
    // Both move d1, differently.
    moved(&mut ours, "d1", [1.0, 1.0]);
    moved(&mut theirs, "d1", [2.0, 2.0]);
    // Ours rescales the detail view; theirs deletes it.
    ours.views[2].scale = Some((2, 1));
    theirs.views.retain(|v| v.name != "b");
    // Theirs deletes the BOM table the balloon numbers by.
    theirs.tables.retain(|t| t.name != "bom");
    // Both add `n9`, differently.
    for (d, text) in [(&mut ours, "OURS"), (&mut theirs, "THEIRS")] {
        d.annotations.push(annotation(
            "n9",
            "top",
            AnnotationKind::Note {
                text: text.into(),
                leader: None,
            },
        ));
    }
    // Theirs re-points g1's view to the deleted detail view.
    theirs.annotations[1].view = "b".into();
    let m = merge_drawings(&base, &ours, &theirs, None);
    let kinds: Vec<(DrawingCollection, String, DrawingConflictKind)> = m
        .conflicts
        .iter()
        .map(|c| (c.collection, c.id.clone(), c.kind.clone()))
        .collect();
    assert_eq!(
        kinds,
        vec![
            (
                DrawingCollection::View,
                "b".into(),
                DrawingConflictKind::DeleteModify {
                    deleted_by: MergeSide::Theirs
                }
            ),
            (
                DrawingCollection::Annotation,
                "d1".into(),
                DrawingConflictKind::BothChanged {
                    fields: vec!["at".into()]
                }
            ),
            (
                DrawingCollection::Annotation,
                "n9".into(),
                DrawingConflictKind::BothAdded {
                    fields: vec!["kind".into()]
                }
            ),
        ]
    );
    // Ours is kept on conflicting fields; the modified view survives its deletion.
    assert_eq!(m.merged.annotations[0].at, [1.0, 1.0]);
    assert!(m.merged.view("b").is_some());
    assert_eq!(m.merged.annotations[1].view, "b");
    let msgs: Vec<(DiagCode, String)> = m
        .diagnostics
        .diagnostics
        .iter()
        .map(|d| (d.code, d.message.clone()))
        .collect();
    let want = [
        (
            DiagCode::DrawingMergeConflict,
            "they deleted view `b`, which we changed; the changed view is kept",
        ),
        (
            DiagCode::DrawingMergeConflict,
            "both sides changed at of annotation `d1`; ours is kept",
        ),
        (
            DiagCode::DrawingMergeConflict,
            "both sides added annotation `n9` with different kind; ours is kept",
        ),
        (
            DiagCode::MergeBroken,
            "after the merge, balloon `p1` numbers by BOM table `bom`, which was deleted",
        ),
    ];
    for (c, w) in want {
        assert!(
            msgs.iter().any(|(cc, mm)| *cc == c && mm == w),
            "missing {:?} {:?} in {:#?}",
            c,
            w,
            msgs
        );
    }
    assert_eq!(msgs.len(), want.len(), "{:#?}", msgs);

    // With the model: theirs re-picks d1 onto geometry that no longer resolves.
    let base = drawing();
    let ours = base.clone();
    let mut theirs = base.clone();
    theirs.annotations[0].kind = dim(vec![Reference::end_cap("cut1")]);
    let model = Model {
        exploded: ["service".to_string()].into(),
    };
    let m = merge_drawings(&base, &ours, &theirs, Some(&model));
    let broken = m.diagnostics.with_code(DiagCode::MergeBroken);
    assert_eq!(broken.len(), 1, "{:#?}", m.diagnostics);
    assert_eq!(
        broken[0].message,
        "annotation `d1` resolved on our side but no longer resolves after the merge"
    );
    assert_eq!(broken[0].site.key(), "annotation.d1#references");
}
