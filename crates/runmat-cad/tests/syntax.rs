//! The `.part`, `.assembly` and `.drawing` front ends: model → print → parse gives the same
//! model, every site has its span, and structured edits change only what they must.

mod common;
use common::*;
use runmat_cad::assembly::model::RelPath;
use runmat_cad::assembly::synth::{generate, SynthSpec};
use runmat_cad::drawing::*;
use runmat_cad::ids::FilePath;
use runmat_cad::model::*;
use runmat_cad::refs::lock::{ItemPath, LockSection, SeededMinter};
use runmat_cad::refs::*;
use runmat_cad::scope::UseDecl;
use runmat_cad::syntax::edit::{self, ConfigurationScope, ValueEdit};
use runmat_cad::syntax::projection::model_projection;
use runmat_cad::syntax::{
    parse_part, print_assembly, print_drawing, print_part,
};
use runmat_cad::value::{value_expr_examples, InputDecl, LetDecl, ValueExpr};
use runmat_design_syntax::Dialect;
use std::collections::BTreeMap;

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
            profile: ProfileRef::sketch("s_cut").into(),
            extent: Extent::ThroughAll,
            reverse: false,
            mode: BodyMode::cut(),
            from: Default::default(),
            second: None,
            draft: None,
            thin: None,
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
            diameter: Some(mm(4.0)),
            end: HoleEnd::ThroughAll,
            form: HoleForm::Simple,
            spec: None,
            drill_point: None,
            series: None,
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

fn rich() -> PartDocument {
    let mut d = bracket();
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
    let mut thickness = InputDecl::new("plate", "length", mm(3.0));
    thickness.minimum = Some(mm(1.0));
    thickness.maximum = Some(input("thickness"));
    d.inputs.push(thickness);
    for (index, (_, value)) in value_expr_examples().into_iter().enumerate() {
        d.lets.push(LetDecl {
            name: runmat_cad::ids::LetName::new(format!("v{index}")),
            kind: None,
            value,
        });
    }
    d.lets.push(LetDecl {
        name: "half".into(),
        kind: Some(runmat_cad::units::QuantityKindId::new("length")),
        value: ValueExpr::binary(
            runmat_cad::value::BinaryOp::Mul,
            mm(-2.5),
            ValueExpr::Neg {
                operand: Box::new(ValueExpr::number(3.0)),
            },
        ),
    });
    d.materials.push(MaterialAssignment {
        body: Reference::feature_body("body"),
        material: "material.steel".into(),
    });
    d.links.push(Link {
        kind: LinkKind::Verifies,
        from: LocalItem::Feature {
            feature: "body".into(),
        },
        to: "checks/mass.fea#check.mass_budget".into(),
    });
    d.configurations[0]
        .overrides
        .insert("width".into(), mm(120.0));
    d.configurations[0].suppress.insert(fid("round_missing"));
    d.features[1].suppressed = true;
    d.closed = true;
    d
}

fn uses() -> Vec<UseDecl> {
    vec![
        UseDecl {
            symbol: Some(("material".into(), "aluminum".into())),
            from: "./materials".into(),
            alias: None,
            revision: None,
        },
        UseDecl {
            symbol: Some(("part".into(), "skeleton".into())),
            from: "//wing/skeleton".into(),
            alias: Some("sk".into()),
            revision: Some("R3".into()),
        },
        UseDecl {
            symbol: None,
            from: "./fasteners".into(),
            alias: Some("lib".into()),
            revision: None,
        },
    ]
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
            standard: Default::default(),
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
            View {
                scale: Some((2, 1)),
                ..view(
                    "b",
                    ViewKind::Detail {
                        from: "a_a".into(),
                        center: Reference::end_cap("body"),
                        radius: mm(20.0),
                        label: "B".into(),
                    },
                )
            },
            view(
                "flat",
                ViewKind::FlatPattern {
                    feature: runmat_cad::ids::FeatureId::new("flat"),
                    bend_notes: true,
                    bend_table: false,
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
                    condition: Default::default(),
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

fn round_trip_part(document: &PartDocument, uses: &[UseDecl]) -> String {
    let text = print_part(document, uses).unwrap();
    let parsed = parse_part(&document.path.0, &text);
    assert!(
        parsed.diagnostics.is_empty(),
        "{text}\n{:#?}",
        parsed.diagnostics
    );
    assert_eq!(parsed.document.as_ref(), Some(document), "{text}");
    assert_eq!(parsed.uses, uses);
    // Printing is stable: the printed text prints back to itself.
    assert_eq!(
        print_part(parsed.document.as_ref().unwrap(), &parsed.uses).unwrap(),
        text
    );
    text
}

#[test]
fn parts_round_trip_through_text() {
    round_trip_part(&block(), &[]);
    round_trip_part(&bracket(), &[]);
    let text = round_trip_part(&rich(), &uses());
    assert!(
        text.starts_with("runmat 0.7\nstd 1\n\nuse material aluminum from \"./materials\"\n"),
        "{text}"
    );
    assert!(text.contains("\npart bracket {\n"), "{text}");
    assert!(
        text.contains("    input width : length = 80 mm\n"),
        "{text}"
    );
    assert!(text.contains("    input plate : length = { default = 3 mm, minimum = 1 mm, maximum = input.thickness }\n"), "{text}");
    assert!(text.contains("    let v6 = 10 kohm ± 1 %\n"), "{text}");
    assert!(
        text.contains("    let v7 = airDensity(altitude = 10 km)\n"),
        "{text}"
    );
    assert!(text.contains("    let v8 = rho(2) as kg/m^3\n"), "{text}");
    assert!(text.contains("    let v10 = part.bracket.mass\n"), "{text}");
    assert!(
        text.contains("    let half : length = (-2.5 mm) * -(3)\n"),
        "{text}"
    );
    assert!(
        text.contains("    feature body = extrude(profile = sketch.s1, distance = input.thickness, suppressed)\n"),
        "{text}"
    );
    assert!(text.contains("    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])\n"), "{text}");
    assert!(
        text.contains(
            "    feature cut = extrude(profile = sketch.s_cut, through_all, mode = cut)\n"
        ),
        "{text}"
    );
    assert!(
        text.contains("        at = { a = point(60 mm, 10 mm, 10 mm) },\n"),
        "{text}"
    );
    assert!(
        text.contains(
            "edges = [edges_between(feature.body.face.side(s1.l1), feature.body.face.end)],"
        ),
        "{text}"
    );
    assert!(text.contains("        layout = grid(\n"), "{text}");
    assert!(
        text.contains("    link verifies feature.body to \"checks/mass.fea#check.mass_budget\"\n"),
        "{text}"
    );
    assert!(text.contains("    closed = true\n"), "{text}");
    assert!(
        text.contains("    publish top = region(region = top)\n"),
        "{text}"
    );
}

#[test]
fn every_value_form_reads_back_from_its_text() {
    for (_, value) in value_expr_examples() {
        let mut d = block();
        d.lets.push(LetDecl {
            name: "x".into(),
            kind: None,
            value: value.clone(),
        });
        let pinned = ValueExpr::published(
            "bracket.part",
            "mass",
            XrefContext::Pinned {
                revision: "R3".into(),
            },
        );
        d.lets.push(LetDecl {
            name: "y".into(),
            kind: None,
            value: ValueExpr::binary(runmat_cad::value::BinaryOp::Add, pinned, value),
        });
        round_trip_part(&d, &[]);
    }
}

#[test]
fn hand_written_values_take_their_source_forms() {
    let text = "part p {\n    input w : length = 80 mm\n    let half = input.w / 2 - 1 mm\n    let twice = half * 2\n    let tol = 10 mm ± [0.02 mm, 0 mm]\n    let pin = part.skeleton.publish.pitch @ \"R3\"\n    let live = volatile sensor(channel = 2)\n}\n";
    let parsed = parse_part("p.part", text);
    assert!(parsed.is_valid(), "{:#?}", parsed.diagnostics);
    let d = parsed.document.unwrap();
    use runmat_cad::value::BinaryOp::*;
    assert_eq!(
        d.lets[0].value,
        ValueExpr::binary(
            Sub,
            ValueExpr::binary(Div, input("w"), ValueExpr::number(2.0)),
            mm(1.0)
        )
    );
    assert_eq!(
        d.lets[1].value,
        ValueExpr::binary(Mul, ValueExpr::let_("half"), ValueExpr::number(2.0))
    );
    assert_eq!(
        d.lets[2].value,
        ValueExpr::Toleranced {
            nominal: Box::new(mm(10.0)),
            plus: Box::new(mm(0.02)),
            minus: Box::new(mm(0.0)),
        }
    );
    assert_eq!(
        d.lets[3].value,
        ValueExpr::published(
            "skeleton.part",
            "pitch",
            XrefContext::Pinned {
                revision: "R3".into()
            }
        )
    );
    let ValueExpr::Dynamic { call } = &d.lets[4].value else {
        panic!()
    };
    assert!(call.volatile);
}

#[test]
fn every_site_points_at_its_text() {
    let text = print_part(&rich(), &[]).unwrap();
    let parsed = parse_part("bracket.part", &text);
    let at = |key: &str| {
        let span = parsed
            .spans
            .spans
            .get(key)
            .unwrap_or_else(|| panic!("no site `{key}`"));
        &text[span.start as usize..span.end as usize]
    };
    assert!(at("feature.round").starts_with("feature round = fillet("));
    assert!(at("feature.h#diameter").starts_with("4 mm"));
    assert_eq!(at("input.width#default"), "80 mm");
    assert_eq!(at("input.plate#maximum"), "input.thickness");
    assert!(at("region.top").starts_with("region top : face = {"));
    assert_eq!(at("publish.top#region"), "top");
    assert!(at("material[0]").starts_with("material = {"));
    assert!(at("link[0]").starts_with("link verifies"));
    assert_eq!(at("closed"), "closed = true");
    assert!(at("document").starts_with("part bracket {"));
    // The keys are the diagnostics' site keys.
    let site = runmat_cad::diagnostics::Site::feature(&fid("round"), Some("edges[0]"));
    assert!(
        parsed.spans.spans.contains_key(&site.key()),
        "{:?}",
        parsed.spans.spans.keys().collect::<Vec<_>>()
    );
}

#[test]
fn model_errors_point_at_the_value() {
    let text = "part p {\n    input w : length = \"wide\"\n}\n";
    let parsed = parse_part("p.part", text);
    assert!(parsed.document.is_none());
    let diagnostic = &parsed.diagnostics[0];
    assert_eq!(
        &text[diagnostic.span.start..diagnostic.span.end],
        "\"wide\""
    );
}

#[test]
fn assemblies_round_trip_through_text() {
    let project = generate(&SynthSpec::small());
    let mut count = 0;
    for document in project.source.assemblies.values() {
        let text = print_assembly(document, &[]).unwrap();
        let parsed = reparse_assembly(&document.path.0, &text, document.as_ref());
        assert!(
            parsed.diagnostics.is_empty(),
            "{text}\n{:#?}",
            parsed.diagnostics
        );
        assert_eq!(parsed.document.as_ref(), Some(document.as_ref()), "{text}");
        count += 1;
    }
    assert!(count > 2);
}

#[test]
fn drawings_round_trip_through_text() {
    let document = drawing();
    let text = print_drawing(&document, &[]).unwrap();
    let parsed = reparse_drawing("rig.drawing", &text, &document);
    assert!(
        parsed.diagnostics.is_empty(),
        "{text}\n{:#?}",
        parsed.diagnostics
    );
    assert_eq!(parsed.document, Some(document), "{text}");
    assert!(text.contains("\ndrawing rig {\n"), "{text}");
    assert!(
        text.contains("    sheet 1") || text.contains("    sheet "),
        "{text}"
    );
}

#[test]
fn the_model_projection_is_versioned() {
    let text = print_part(&block(), &[]).unwrap();
    let parsed = parse_part("bracket.part", &text);
    let projection = model_projection(Dialect::Part, "bracket.part", &parsed);
    assert_eq!(projection["format"], "runmat.design-model");
    assert_eq!(projection["version"], 1);
    assert_eq!(projection["dialect"], "part");
    assert_eq!(projection["document"]["inputs"][0]["name"], "width");
    assert!(projection["spans"]["feature.body"].is_array());
}

// ------------------------------------------------------------------------------- edits

fn plate() -> PartDocument {
    let mut d = PartDocument::new("plate.part");
    d.inputs.push(InputDecl::new("width", "length", mm(80.0)));
    d.inputs.push(InputDecl::new("depth", "length", mm(50.0)));
    d.features.push(sketch(
        "s1",
        rect(
            PlaneRef::origin_xy(),
            mm(0.0),
            mm(0.0),
            input("width"),
            input("depth"),
        ),
    ));
    d.features
        .push(extrude("body", "s1", mm(10.0), BodyMode::New));
    d.features.push(fillet("round", vec![], 2.0));
    d.configurations.push(Configuration {
        name: "long".into(),
        ..Default::default()
    });
    d
}

/// The plate as a person might keep it: printed, then commented.
fn source() -> String {
    let text = print_part(&plate(), &[]).unwrap();
    let text = text
        .replacen(
            "part plate {\n",
            "// A plate with a hole.\npart plate {\n    // Overall size.\n",
            1,
        )
        .replacen(
            "input width : length = 80 mm\n",
            "input width : length = 80 mm   // from the enclosure\n",
            1,
        )
        .replacen(
            "    feature body =",
            "    // The body.\n    feature body =",
            1,
        );
    assert!(
        text.contains("// The body.") && text.contains("// from the enclosure"),
        "{text}"
    );
    text
}

fn check(change: &edit::SourceChange) -> PartDocument {
    assert!(!change.reprinted, "{}", change.source);
    let parsed = parse_part("plate.part", &change.source);
    assert!(
        parsed.diagnostics.is_empty(),
        "{}\n{:#?}",
        change.source,
        parsed.diagnostics
    );
    parsed.document.unwrap()
}

#[test]
fn the_edit_fixture_parses() {
    let parsed = parse_part("plate.part", &source());
    assert!(parsed.is_valid(), "{:#?}", parsed.diagnostics);
}

#[test]
fn setting_a_parameter_replaces_only_its_value() {
    let source = source();
    let change = edit::set_feature_param(
        &source,
        "plate.part",
        "round",
        "radius",
        serde_json::to_value(mm(3.0)).unwrap(),
    )
    .unwrap();
    assert_eq!(change.edits.len(), 1, "{:#?}", change.edits);
    assert_eq!(
        change.source,
        source.replace("radius = 2 mm", "radius = 3 mm")
    );
    check(&change);
}

#[test]
fn a_value_edit_sets_the_default_or_a_configuration_override() {
    let source = source();
    let all = ValueEdit {
        value: mm(90.0),
        text: "90 mm".into(),
        scope: ConfigurationScope::All,
    };
    let change = edit::set_value(&source, "plate.part", "width", &all).unwrap();
    assert_eq!(change.source, source.replace("80 mm   //", "90 mm   //"));
    let this = ValueEdit {
        scope: ConfigurationScope::This {
            configuration: "long".into(),
        },
        ..all
    };
    let change = edit::set_value(&source, "plate.part", "width", &this).unwrap();
    assert!(
        change
            .source
            .contains("configuration long = { width = 90 mm }"),
        "{}",
        change.source
    );
    assert!(change.source.contains("80 mm   // from the enclosure"));
    let document = check(&change);
    assert_eq!(
        document.configurations[0].overrides[&"width".into()],
        mm(90.0)
    );
    let back = edit::set_value(
        &change.source,
        "plate.part",
        "width",
        &ValueEdit {
            scope: ConfigurationScope::All,
            ..this.clone()
        },
    )
    .unwrap();
    assert!(
        back.source.contains("configuration long = {}"),
        "{}",
        back.source
    );
}

#[test]
fn adding_removing_and_moving_features_keep_comments() {
    let source = source();
    let feature = fillet("edge_round", vec![], 1.0);
    let change = edit::add_feature(&source, "plate.part", feature, Some("body")).unwrap();
    let document = check(&change);
    assert_eq!(document.features[2].id, fid("edge_round"));
    let (added, round) = (
        change.source.find("feature edge_round =").unwrap(),
        change.source.find("feature round =").unwrap(),
    );
    assert!(
        change.source.find("feature body =").unwrap() < added && added < round,
        "{}",
        change.source
    );
    assert_eq!(change.edits.len(), 1);

    let change = edit::remove_feature(&source, "plate.part", "body").unwrap();
    assert!(!change.source.contains("// The body."), "{}", change.source);
    assert!(change.source.contains("// Overall size."));
    check(&change);

    let change = edit::reorder_feature(&source, "plate.part", "round", Some("s1")).unwrap();
    let document = check(&change);
    let order: Vec<&str> = document.features.iter().map(|f| f.id.0.as_str()).collect();
    assert_eq!(order, ["s1", "round", "body"]);
    assert!(
        change.source.contains("// The body.\n    feature body"),
        "{}",
        change.source
    );
    assert!(
        change.source.find("feature round =").unwrap()
            < change.source.find("// The body.").unwrap(),
        "{}",
        change.source
    );

    let change = edit::suppress_feature(&source, "plate.part", "round", true).unwrap();
    assert!(
        change.source.contains("radius = 2 mm, suppressed)"),
        "{}",
        change.source
    );
    assert!(check(&change).features[2].suppressed);
}

#[test]
fn renaming_a_feature_records_the_alias_in_the_lock() {
    let source = source();
    let mut lock = LockSection::default();
    let mut minter = SeededMinter {
        seed: "test".into(),
    };
    let change = edit::rename_feature(
        &source,
        "plate.part",
        "round",
        "edge_round",
        &mut lock,
        &mut minter,
    )
    .unwrap();
    assert_eq!(
        change.source,
        source.replace("feature round =", "feature edge_round =")
    );
    check(&change);
    let file = FilePath::new("plate.part");
    let found = lock
        .lookup(&ItemPath::feature(&file, &fid("round")))
        .unwrap();
    assert!(found.via_alias);
    assert_eq!(found.current, ItemPath::feature(&file, &fid("edge_round")));
}

#[test]
fn adding_a_publish_and_removing_it_restores_the_source() {
    let source = source();
    let entry = PublishEntry {
        name: "face".into(),
        target: PublishTarget::Region {
            region: "top".into(),
        },
        auto: false,
    };
    let change = edit::add_publish(&source, "plate.part", entry).unwrap();
    assert!(
        change
            .source
            .contains("    publish face = region(region = top)\n"),
        "{}",
        change.source
    );
    assert!(change.source.contains("// A plate with a hole."));
    check(&change);
    let change = edit::remove_publish(&change.source, "plate.part", "face").unwrap();
    assert_eq!(change.source, source);
}

#[allow(dead_code)]
fn unused(_: BTreeMap<(), ()>, _: RelPath) {}
