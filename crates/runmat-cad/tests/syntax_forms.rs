//! The final surface forms: every feature kind, sketch entity and constraint, and assembly
//! occurrence round-trips through text, the document corpus reads as its models and prints
//! back unchanged, and record forms of entries that have a call form or a `use` form are
//! refused.

mod common;
use common::*;
use runmat_cad::assembly::AssemblyDocument;
use runmat_cad::drawing::DrawingDocument;
use runmat_cad::model::*;
use runmat_cad::syntax::{self, parse_part, print_part};
use std::collections::BTreeSet;
use std::path::PathBuf;

/// The rig's folder: the designs the occurrence tests name, by symbol.
const RIG_FOLDER: &[(&str, &str)] = &[
    ("plate.part", "part plate {\n}\n"),
    ("gearbox.assembly", "assembly gearbox {\n}\n"),
    ("wing.assembly", "assembly wing {\n}\n"),
    ("parts/bracket.part", "part bracket {\n    output plate = { body = feature.b.body }\n}\n"),
    ("parts/clip.part", "part clip {\n}\n"),
    ("parts/spar_b.part", "part spar_b {\n    output spar = { body = feature.b.body }\n}\n"),
];

fn rig_scope(text: &str) -> runmat_cad::modules::ModuleScope {
    runmat_cad::modules::ModuleScope::from_files(
        std::iter::once(("rig.assembly", text)).chain(RIG_FOLDER.iter().copied()),
    )
}

fn parse_rig(text: &str) -> syntax::Parsed<AssemblyDocument> {
    syntax::parse_assembly_in("rig.assembly", text, &rig_scope(text))
}

fn corpus() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/corpus")
}

fn round_trip(document: &PartDocument) -> String {
    let text = print_part(document, &[]).unwrap();
    let parsed = parse_part(&document.path.0, &text);
    assert!(
        parsed.diagnostics.is_empty(),
        "{text}\n{:#?}",
        parsed.diagnostics
    );
    assert_eq!(parsed.document.as_ref(), Some(document), "{text}");
    assert_eq!(
        print_part(parsed.document.as_ref().unwrap(), &[]).unwrap(),
        text
    );
    text
}

/// Features harvested from the crate's test models: one or more of every kind and every
/// variant of the values inside them.
fn corpus_features() -> Vec<Feature> {
    let text = std::fs::read_to_string(corpus().join("features.part")).unwrap();
    let parsed = parse_part("features.part", &text);
    assert!(parsed.is_valid(), "{:#?}", parsed.diagnostics);
    let document = parsed.document.unwrap();
    // The corpus file is in the printed form.
    assert_eq!(print_part(&document, &[]).unwrap(), text);
    let mut features = document.features;
    // Kinds the harvest did not reach.
    features.push(Feature::new(
        "p_at",
        FeatureKind::Point(PointDef::At {
            at: PointRef::Coords {
                x: mm(1.0),
                y: mm(2.0),
                z: mm(3.0),
            },
        }),
    ));
    features.push(Feature::new(
        "frame_at",
        FeatureKind::Frame(FrameDef::At {
            origin: PointRef::Datum {
                datum: DatumRef::OriginPoint,
            },
            z: AxisRef::origin(OriginAxis::Z),
            x: Some(AxisRef::origin(OriginAxis::X)),
        }),
    ));
    features.push(Feature::new(
        "rib1",
        FeatureKind::Rib(runmat_cad::model::detail::Rib {
            profile: ProfileRef::sketch("s1"),
            thickness: mm(2.0),
            depth: mm(5.0),
            reverse: false,
        }),
    ));
    features
}

#[test]
fn every_feature_kind_round_trips_in_its_call_form() {
    let features = corpus_features();
    let mut kinds = BTreeSet::new();
    for feature in &features {
        kinds.insert(feature.kind.name());
        let mut document = PartDocument::new("kinds.part");
        document.features.push(feature.clone());
        let text = round_trip(&document);
        // The canonical form is the call form (or a sketch block), never the record form.
        let kind = serde_json::to_value(&feature.kind).unwrap()["feature"]
            .as_str()
            .unwrap()
            .to_string();
        if kind == "sketch" {
            assert!(
                text.contains(&format!("    sketch {} on ", feature.id)),
                "{text}"
            );
        } else {
            assert!(
                text.contains(&format!("    feature {} = {kind}", feature.id))
                    && !text.contains(&format!("= {kind} {{")),
                "{text}"
            );
        }
    }
    // Every kind the model has, apart from those the families report by their own name.
    for kind in [
        "sketch",
        "plane",
        "axis",
        "point",
        "frame",
        "extrude",
        "revolve",
        "sweep",
        "loft",
        "hole",
        "fillet",
        "face_fillet",
        "full_round",
        "chamfer",
        "shell",
        "draft",
        "pattern",
        "mirror",
        "boolean",
        "split",
        "body_op",
        "import",
    ] {
        assert!(
            kinds.contains(kind),
            "the corpus has no `{kind}`: {kinds:?}"
        );
    }
    assert!(kinds.len() >= 33, "{kinds:?}");
    // All features in one part, as a design would hold them.
    let mut document = PartDocument::new("kinds.part");
    document.features = features;
    round_trip(&document);
}

#[test]
fn hand_written_features_read_their_readable_forms() {
    let text = r#"part p {
    input t : length = 8 mm

    sketch s1 on origin.xy {
        l1 = line(start = [0, 0] mm, end = [80 mm, 0 mm])
        l2 = line(start = l1.end, end = [80 mm, 50 mm])
        l3 = line(start = l2.end, end = [0 mm, 50 mm])
        l4 = line(start = l3.end, end = l1.start)
        c1 = circle(center = [40 mm, 25 mm], radius = 5 mm, construction)
        a1 = arc(center = [0 mm, 0 mm], start = [1 mm, 0 mm], end = [0 mm, 1 mm])
        p1 = point(at = [3 mm, 4 mm])
        constrain horizontal(l1)
        constrain coincident(l1.end, l2.start, id = k1)
        constrain distance(l1.start, l2.end) = 60 mm
        constrain length(l1, id = len) = 80 mm
        constrain radius(c1) = 5 mm
        constrain angle(l1, l2) = 90 deg
        constrain limit(length(l3, value = 80 mm), min = 60 mm, max = 90 mm)
        constrain fix(p1)
        solved {
            l1.start = [0, 0]
            l2.end = [0.08, 0.05]
            c1.radius = 0.005
            k1.hidden_0 = 0.25
        }
    }
    feature body = extrude(profile = sketch.s1, distance = input.t)
    feature both = extrude(profile = sketch.s1, distance = 4 mm, symmetric, mode = add)
    feature cut = extrude(profile = sketch.s1, through_all, mode = cut, reverse)
    feature mid = plane.offset(base = origin.xy, distance = 20 mm)
    feature edge = chamfer(edges = [edges_between(feature.body.face.side(s1.l1), feature.body.face.end)], distance = 1 mm, angle = 45 deg)
    feature h = hole(face = feature.body.face.end, at = { a = point(10 mm, 10 mm, 8 mm), "b 2" = sketch.s1.p1.at }, diameter = 4 mm, depth = 6 mm, form = counterbore(diameter = 8 mm, depth = 2 mm))
    feature d = draft(faces = [feature.body.face.side(s1.l2)], neutral = feature.body.face.start, angle = 2 deg)
    feature m = mirror(features = [h], plane = plane.mid)
}
"#;
    let parsed = parse_part("p.part", text);
    assert!(parsed.is_valid(), "{:#?}", parsed.diagnostics);
    let d = parsed.document.unwrap();
    let FeatureKind::Sketch(s) = &d.features[0].kind else {
        panic!()
    };
    assert_eq!(s.entities.len(), 7);
    assert!(s.entities[4].construction);
    assert_eq!(s.constraints.len(), 8);
    assert_eq!(s.constraints[1].id.as_ref().unwrap().0, "k1");
    assert!(matches!(
        s.constraints[6].kind,
        SketchConstraintKind::Limit { .. }
    ));
    assert_eq!(s.solved.points.len(), 2);
    assert_eq!(s.solved.scalars.len(), 2);
    let FeatureKind::Extrude(both) = &d.features[2].kind else {
        panic!()
    };
    assert!(matches!(both.extent, Extent::Symmetric { .. }));
    assert_eq!(both.mode, BodyMode::add());
    let FeatureKind::Extrude(cut) = &d.features[3].kind else {
        panic!()
    };
    assert!(cut.reverse && matches!(cut.extent, Extent::ThroughAll));
    assert!(matches!(
        d.features[4].kind,
        FeatureKind::Plane(PlaneDef::Offset { .. })
    ));
    let FeatureKind::Chamfer(c) = &d.features[5].kind else {
        panic!()
    };
    assert!(matches!(c.form, ChamferForm::DistanceAngle { .. }));
    let FeatureKind::Hole(h) = &d.features[6].kind else {
        panic!()
    };
    assert_eq!(h.positions[1].key, "b 2");
    assert!(matches!(h.end, HoleEnd::Blind { .. }));
    assert!(matches!(h.form, HoleForm::Counterbore { .. }));
    let FeatureKind::Draft(draft) = &d.features[7].kind else {
        panic!()
    };
    assert!(matches!(draft.neutral, PlaneRef::Face { .. }));
    // Spans keep the model's site keys, wherever the short form wrote the value.
    let at = |key: &str| {
        let span = parsed
            .spans
            .spans
            .get(key)
            .unwrap_or_else(|| panic!("no site `{key}`"));
        &text[span.start as usize..span.end as usize]
    };
    assert_eq!(at("feature.body#extent.distance"), "input.t");
    assert_eq!(at("feature.h#positions[0].at"), "point(10 mm, 10 mm, 8 mm)");
    assert!(at("feature.s1#constraints[2]").starts_with("constrain distance("));
    assert!(at("feature.s1").starts_with("sketch s1 on origin.xy {"));
    // The printed forms read back to the same model, and print the same way again.
    round_trip(&d);
    // The long forms read as the same model.
    let long = text
        .replace(
            "feature body = extrude(profile = sketch.s1, distance = input.t)",
            "feature body = extrude(profile = { sketch = s1 }, extent = blind(distance = input.t))",
        )
        .replace("mode = cut, reverse)", "mode = cut(), reverse = true)");
    let long = parse_part("p.part", &long);
    assert!(long.diagnostics.is_empty(), "{:#?}", long.diagnostics);
    assert_eq!(long.document.as_ref(), Some(&d));
}

#[test]
fn conflicting_short_forms_are_diagnosed() {
    for (body, message) in [
        (
            "extrude(profile = sketch.s, distance = 1 mm, through_all)",
            "do not go together",
        ),
        (
            "extrude(profile = sketch.s, distance = 1 mm, extent = through_all)",
            "both set",
        ),
        ("extrude(profile = sketch.s, through_all = false)", "flag"),
    ] {
        let text = format!("part p {{\n    feature e = {body}\n}}\n");
        let parsed = parse_part("p.part", &text);
        assert!(
            parsed
                .diagnostics
                .iter()
                .any(|d| d.message.contains(message)),
            "{body}: {:#?}",
            parsed.diagnostics
        );
    }
}

#[test]
fn occurrences_are_use_statements_with_overrides() {
    let text = r#"assembly rig {
    use part bracket from "./parts" as left {
        configuration = { select = named, name = light }
        fixed = false
        suppressed = true
    }
    use part plate from "./parts" @ "R3" as right
    use assembly gearbox as gb
}
"#;
    let parsed = parse_rig(text);
    assert!(parsed.is_valid(), "{:#?}", parsed.diagnostics);
    let d: AssemblyDocument = parsed.document.unwrap();
    assert_eq!(d.occurrences.len(), 3);
    assert!(!d.occurrences[0].fixed && d.occurrences[0].suppressed);
    assert_eq!(
        d.occurrences[1].source.definition.part.as_ref().unwrap().0,
        "plate"
    );
    assert!(d.occurrences[0].source.definition.part.is_none());
    let printed = runmat_cad::syntax::print_assembly(&d, &[]).unwrap();
    assert!(
        printed.contains("    use part bracket from \"./parts\" as left {\n"),
        "{printed}"
    );
    assert!(
        printed.contains("    use part plate from \"./parts\" @ \"R3\" as right\n"),
        "{printed}"
    );
    let back = parse_rig(&printed);
    assert_eq!(back.document.as_ref(), Some(&d), "{printed}");
    assert!(back.spans.spans.contains_key("occurrence.left"));
}

/// A fastener rule's family names its design as an occurrence does (`{ kind = part, name,
/// from }`), resolves to the declaring file, and prints back by name.
#[test]
fn a_fastener_family_names_its_design() {
    let source = std::fs::read_to_string(corpus().join("documents/synth2.assembly")).unwrap();
    assert!(
        source.contains("family = { kind = part, name = hi_lok, from = \"../library/fasteners\" }"),
        "{source}"
    );
    let model: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(corpus().join("documents/synth2.assembly.json")).unwrap(),
    )
    .unwrap();
    let parsed = reparse_assembly("fuselage/section.assembly", &source, &model);
    assert!(parsed.is_valid(), "{:#?}", parsed.diagnostics);
    let document = parsed.document.unwrap();
    assert_eq!(
        document.fasteners[0].fastener.family.file.0,
        "library/fasteners/hi-lok.part"
    );
    let printed = syntax::print_assembly(&document, &parsed.uses).unwrap();
    assert!(
        printed.contains("family = { kind = part, name = hi_lok, from = \"../library/fasteners\" }"),
        "{printed}"
    );
}

/// The document corpus: parts, assemblies and a drawing in the final forms. Each reads as
/// its model (`<file>.json`) and prints back to the same text.
#[test]
fn the_document_corpus_reads_as_its_models_and_prints_back() {
    let dir = corpus().join("documents");
    let mut count = 0;
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if name.ends_with(".json") {
            continue;
        }
        let source = std::fs::read_to_string(&path).unwrap();
        let model: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap(),
        )
        .unwrap();
        let file = model["path"].as_str().unwrap().to_string();
        let (read, printed) = if name.ends_with(".part") {
            let p = parse_part(&file, &source);
            assert!(p.is_valid(), "{name}: {:#?}", p.diagnostics);
            let d = p.document.unwrap();
            (
                serde_json::to_value(&d).unwrap(),
                print_part(&d, &p.uses).unwrap(),
            )
        } else if name.ends_with(".assembly") {
            let p = reparse_assembly(&file, &source, &model);
            assert!(p.is_valid(), "{name}: {:#?}", p.diagnostics);
            let d = p.document.unwrap();
            (
                serde_json::to_value(&d).unwrap(),
                syntax::print_assembly(&d, &p.uses).unwrap(),
            )
        } else {
            let p = reparse_drawing(&file, &source, &model);
            assert!(p.is_valid(), "{name}: {:#?}", p.diagnostics);
            let d: DrawingDocument = p.document.unwrap();
            (
                serde_json::to_value(&d).unwrap(),
                syntax::print_drawing(&d, &p.uses).unwrap(),
            )
        };
        assert_eq!(read, model, "{name}: the file reads as its model");
        if std::env::var_os("RUNMAT_CAD_CORPUS_WRITE").is_some() {
            std::fs::write(&path, &printed).unwrap();
        }
        assert_eq!(printed, source, "{name}: the file is in the printed form");
        count += 1;
    }
    assert!(count >= 35, "{count}");
}

#[test]
fn record_forms_of_call_form_entries_are_refused() {
    for (path, text) in [
        (
            "p.part",
            "part p {\n    feature body = extrude { profile = sketch.s1, extent = { extent = blind, distance = 10 mm } }\n}\n",
        ),
        (
            "p.part",
            "part p {\n    feature body = { kind = { feature = extrude } }\n}\n",
        ),
        (
            "p.part",
            "part p {\n    feature body = extrude(profile = sketch.s1, extent = { extent = blind, distance = 10 mm })\n}\n",
        ),
        (
            "p.part",
            "part p {\n    feature cut = extrude(profile = sketch.s1, through_all, mode = { mode = cut })\n}\n",
        ),
        (
            "rig.assembly",
            "assembly rig {\n    occurrence left = { source = { definition = { kind = part, file = \"b.part\" } } }\n}\n",
        ),
        (
            "rig.assembly",
            "assembly rig {\n    use part clip from \"c.part\" as clip\n    pattern clips = { seeds = [clip], layout = { layout = linear, direction = [1, 0, 0], spacing = 10 mm, count = 2 } }\n}\n",
        ),
        (
            "rig.assembly",
            "assembly rig {\n    use part b from \"b.part\" as left {\n        overrides = []\n    }\n}\n",
        ),
    ] {
        let diagnostics = if path.ends_with(".part") {
            parse_part(path, text).diagnostics
        } else {
            syntax::parse_assembly(path, text).diagnostics
        };
        assert!(!diagnostics.is_empty(), "{text}");
    }
}

const PLATE: &str = r#"runmat 0.7
std 1

// A plate.
part plate {
    input width : length = 80 mm // from the enclosure

    sketch s1 on origin.xy {
        // The outline.
        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])
        l2 = line(start = l1.end, end = [80 mm, 50 mm])
        constrain horizontal(l1)
        constrain distance(l1.start, l2.end) = 60 mm // checked
    }
    // The body.
    feature body = extrude(profile = sketch.s1, distance = 10 mm)
    feature round = fillet(edges = [], radius = 2 mm)
}
"#;

fn edited(change: &syntax::edit::SourceChange) -> PartDocument {
    assert!(!change.reprinted, "{}", change.source);
    let parsed = parse_part("plate.part", &change.source);
    assert!(
        parsed.is_valid(),
        "{}\n{:#?}",
        change.source,
        parsed.diagnostics
    );
    parsed.document.unwrap()
}

#[test]
fn edits_on_the_surface_forms_change_only_what_they_must() {
    use syntax::edit;
    assert!(parse_part("plate.part", PLATE).is_valid());
    // A short-form argument is set by its model path.
    let change = edit::set_feature_param(
        PLATE,
        "plate.part",
        "body",
        "extent.distance",
        serde_json::to_value(mm(12.0)).unwrap(),
    )
    .unwrap();
    assert_eq!(
        change.source,
        PLATE.replace("distance = 10 mm", "distance = 12 mm")
    );
    assert_eq!(change.edits.len(), 1);
    edited(&change);
    // A sketch dimension changes on its own line; comments stay.
    let change = edit::edit_part(PLATE, "plate.part", |d| {
        let FeatureKind::Sketch(s) = &mut d.features[0].kind else {
            panic!()
        };
        let SketchConstraintKind::Distance { value, .. } = &mut s.constraints[1].kind else {
            panic!()
        };
        *value = mm(65.0);
        Ok(())
    })
    .unwrap();
    assert_eq!(
        change.source,
        PLATE.replace("= 60 mm // checked", "= 65 mm // checked")
    );
    edited(&change);
    // Adding a sketch entity keeps the feature's other statements.
    let change = edit::edit_part(PLATE, "plate.part", |d| {
        let FeatureKind::Sketch(s) = &mut d.features[0].kind else {
            panic!()
        };
        s.entities.push(SketchEntity {
            id: "c1".into(),
            geometry: SketchGeometry::Circle {
                center: PointExpr::mm(10.0, 10.0),
                radius: mm(2.0),
            },
            construction: false,
        });
        Ok(())
    })
    .unwrap();
    let document = edited(&change);
    assert!(change.source.contains("// The body.") && change.source.contains("// A plate."));
    let FeatureKind::Sketch(s) = &document.features[0].kind else {
        panic!()
    };
    assert_eq!(s.entities.len(), 3);
    // Changing the mode adds one argument.
    let change = edit::edit_part(PLATE, "plate.part", |d| {
        let FeatureKind::Extrude(e) = &mut d.features[1].kind else {
            panic!()
        };
        e.mode = BodyMode::cut();
        Ok(())
    })
    .unwrap();
    assert!(change
        .source
        .contains("extrude(profile = sketch.s1, distance = 10 mm, mode = cut)"));
    edited(&change);
}

#[test]
fn the_version_header_is_rewritten_whole() {
    use syntax::edit;
    let change = edit::edit_part(PLATE, "plate.part", |d| {
        d.header.language = "0.8".into();
        d.header.std = "2".into();
        Ok(())
    })
    .unwrap();
    assert_eq!(
        change.source,
        PLATE.replacen("runmat 0.7\nstd 1\n", "runmat 0.8\nstd 2\n", 1)
    );
    edited(&change);
    // A file without a header keeps having none until the header changes.
    let bare = PLATE.replacen("runmat 0.7\nstd 1\n\n", "", 1);
    let change = edit::set_feature_param(
        &bare,
        "plate.part",
        "round",
        "radius",
        serde_json::to_value(mm(3.0)).unwrap(),
    )
    .unwrap();
    assert!(
        change.source.starts_with("// A plate."),
        "{}",
        change.source
    );
    let change = edit::edit_part(&bare, "plate.part", |d| {
        d.header.std = "2".into();
        Ok(())
    })
    .unwrap();
    assert!(
        change
            .source
            .starts_with("runmat 0.7\nstd 2\n\n// A plate."),
        "{}",
        change.source
    );
    edited(&change);
}

#[test]
fn occurrence_patterns_read_and_print_in_their_call_form() {
    let text = r#"assembly rig {
    use part clip from "./parts" as clip
    pattern clips = linear(seeds = [clip], direction = [0, 1, 0], spacing = 750 mm, count = 4)
    pattern ring = circular(seeds = [clip], axis_point = [0 mm, 0 mm, 0 mm], axis = [0, 0, 1], count = 6, angle = 360 deg, skip = [{ instance = circular, step = 2 }])
    pattern old = linear(count = 2, spacing = 10 mm, direction = [1, 0, 0], seeds = [clip])
}
"#;
    let parsed = parse_rig(text);
    assert!(parsed.is_valid(), "{:#?}", parsed.diagnostics);
    let d: AssemblyDocument = parsed.document.unwrap();
    assert_eq!(d.patterns.len(), 3);
    assert_eq!(d.patterns[1].skip.len(), 1);
    // The spans of a call argument point at the argument.
    let span = &parsed.spans.spans["pattern.clips#spacing"];
    assert_eq!(&text[span.start as usize..span.end as usize], "750 mm");
    let printed = runmat_cad::syntax::print_assembly(&d, &[]).unwrap();
    assert!(
        printed.contains(
            "    pattern clips = linear(seeds = [clip], direction = [0, 1, 0], spacing = 750 mm, count = 4)\n"
        ),
        "{printed}"
    );
    assert!(
        printed.contains(
            "    pattern old = linear(seeds = [clip], direction = [1, 0, 0], spacing = 10 mm, count = 2)\n"
        ),
        "{printed}"
    );
    assert!(!printed.contains("layout ="), "{printed}");
    let back = parse_rig(&printed);
    assert_eq!(back.document.as_ref(), Some(&d), "{printed}");
}

const OVERRIDES: &str = r#"assembly rig {
    use assembly wing @ "R3" as port {
        fixed = false
        // The heavy clip on the third rib.
        override "rib@ribs[3]/clip" {
            configuration = { select = named, name = heavy }
            input.thickness = 2 mm
        }
        override spar {
            replace = part(spar, from = "./parts", revision = "B")
        }
        override "rib@ribs[7]" {
            suppressed = true
        }
        override "box@grid[1,2]/lid" {
            representation = envelope
            flexible = true
        }
    }
}
"#;

#[test]
fn occurrence_overrides_read_and_print_as_override_blocks() {
    use runmat_cad::assembly::{OverrideKind, RelPath};
    let parsed = parse_rig(OVERRIDES);
    assert!(parsed.is_valid(), "{:#?}", parsed.diagnostics);
    let d: AssemblyDocument = parsed.document.unwrap();
    let o = &d.occurrences[0];
    assert!(!o.fixed);
    assert_eq!(o.overrides.len(), 6);
    assert_eq!(o.overrides[0].target.to_string(), "rib@ribs[3]/clip");
    assert!(matches!(
        &o.overrides[0].set,
        OverrideKind::Configuration { .. }
    ));
    let OverrideKind::Input { name, .. } = &o.overrides[1].set else {
        panic!("{:?}", o.overrides[1])
    };
    assert_eq!(name.0, "thickness");
    let OverrideKind::Replace { source } = &o.overrides[2].set else {
        panic!("{:?}", o.overrides[2])
    };
    assert_eq!(source.definition.file.0, "parts/spar_b.part");
    assert_eq!(o.overrides[2].target, RelPath::new(&["spar"]));
    assert!(matches!(
        o.overrides[3].set,
        OverrideKind::Suppress { suppressed: true }
    ));
    assert_eq!(o.overrides[4].target.to_string(), "box@grid[1,2]/lid");
    let printed = runmat_cad::syntax::print_assembly(&d, &[]).unwrap();
    for line in [
        "        override \"rib@ribs[3]/clip\" {\n",
        "            input.thickness = 2 mm\n",
        "        override spar {\n            replace = part(spar, from = \"./parts\", revision = \"B\")\n        }\n",
        "        override \"rib@ribs[7]\" {\n            suppressed = true\n        }\n",
    ] {
        assert!(printed.contains(line), "{line}\n---\n{printed}");
    }
    assert!(!printed.contains("overrides ="), "{printed}");
    let back = parse_rig(&printed);
    assert_eq!(back.document.as_ref(), Some(&d), "{printed}");
    for bad in [
        "override \"rib@ribs[x]\" {\n            suppressed = true\n        }",
        "override spar {\n            colour = red\n        }",
        "override spar {\n        }",
        "override spar {\n            replace = part(spar)\n        }",
    ] {
        let text = OVERRIDES.replace(
            "override \"rib@ribs[7]\" {\n            suppressed = true\n        }",
            bad,
        );
        assert!(
            !parse_rig(&text).diagnostics.is_empty(),
            "{bad}"
        );
    }
}

#[test]
fn an_override_edit_keeps_the_comments_of_the_occurrence() {
    use runmat_cad::assembly::OverrideKind;
    let parsed = parse_rig(OVERRIDES);
    let mut d: AssemblyDocument = parsed.document.clone().unwrap();
    d.occurrences[0].overrides[3].set = OverrideKind::Suppress { suppressed: false };
    let change = syntax::edit::diff_source_in(
        &syntax::ASSEMBLY,
        "rig.assembly",
        OVERRIDES,
        &d,
        &parsed.uses,
        &[],
        &rig_scope(""),
    )
    .unwrap();
    assert!(!change.reprinted, "{}", change.source);
    assert_eq!(
        change.source,
        OVERRIDES.replace(
            "override \"rib@ribs[7]\" {\n            suppressed = true",
            "override \"rib@ribs[7]\" {\n            suppressed = false"
        )
    );
}

#[test]
fn adding_or_removing_a_sketch_line_changes_only_that_line() {
    use syntax::edit;
    let circle = SketchEntity {
        id: "c1".into(),
        geometry: SketchGeometry::Circle {
            center: PointExpr::mm(10.0, 10.0),
            radius: mm(2.0),
        },
        construction: false,
    };
    // Added after the last entity, at its indentation; the comments stay.
    let change = edit::edit_part(PLATE, "plate.part", |d| {
        let FeatureKind::Sketch(s) = &mut d.features[0].kind else {
            panic!()
        };
        s.entities.push(circle.clone());
        Ok(())
    })
    .unwrap();
    let added = PLATE.replace(
        "        l2 = line(start = l1.end, end = [80 mm, 50 mm])\n",
        "        l2 = line(start = l1.end, end = [80 mm, 50 mm])\n        c1 = circle(center = [10 mm, 10 mm], radius = 2 mm)\n",
    );
    assert_eq!(change.source, added);
    assert_eq!(change.edits.len(), 1);
    edited(&change);
    // Added first: before the first line and the comment above it.
    let change = edit::edit_part(PLATE, "plate.part", |d| {
        let FeatureKind::Sketch(s) = &mut d.features[0].kind else {
            panic!()
        };
        s.entities.insert(0, circle.clone());
        Ok(())
    })
    .unwrap();
    assert_eq!(
        change.source,
        PLATE.replace(
            "        // The outline.\n",
            "        c1 = circle(center = [10 mm, 10 mm], radius = 2 mm)\n        // The outline.\n",
        )
    );
    edited(&change);
    // Removed: the line goes; the other lines and their comments stay.
    let change = edit::edit_part(&added, "plate.part", |d| {
        let FeatureKind::Sketch(s) = &mut d.features[0].kind else {
            panic!()
        };
        s.entities.retain(|e| e.id.0 != "c1");
        Ok(())
    })
    .unwrap();
    assert_eq!(change.source, PLATE);
    // A constraint is added and another removed in one edit.
    let change = edit::edit_part(PLATE, "plate.part", |d| {
        let FeatureKind::Sketch(s) = &mut d.features[0].kind else {
            panic!()
        };
        s.constraints.remove(0);
        s.constraints.push(SketchConstraint {
            id: None,
            kind: SketchConstraintKind::Vertical {
                line: SketchTarget::Entity {
                    entity: "l2".into(),
                },
            },
        });
        Ok(())
    })
    .unwrap();
    assert_eq!(
        change.source,
        PLATE
            .replace("        constrain horizontal(l1)\n", "")
            .replace(
                "= 60 mm // checked\n",
                "= 60 mm // checked\n        constrain vertical(l2)\n"
            )
    );
    edited(&change);
}

#[test]
fn a_version_header_below_leading_comments_is_edited_in_place() {
    use syntax::edit;
    let commented = format!("// Copyright line.\n// Second line.\n\n{PLATE}");
    assert!(parse_part("plate.part", &commented).is_valid());
    let change = edit::edit_part(&commented, "plate.part", |d| {
        d.header.std = "2".into();
        Ok(())
    })
    .unwrap();
    assert_eq!(
        change.source,
        commented.replacen("std 1\n", "std 2\n", 1),
        "{}",
        change.source
    );
    assert_eq!(change.source.matches("runmat 0.7").count(), 1);
    edited(&change);
    // An edit that leaves the header alone does not touch it.
    let change = edit::set_feature_param(
        &commented,
        "plate.part",
        "round",
        "radius",
        serde_json::to_value(mm(3.0)).unwrap(),
    )
    .unwrap();
    assert_eq!(
        change.source,
        commented.replace("radius = 2 mm", "radius = 3 mm")
    );
}

const TWO_PLATES: &str = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part plate as a\n    use part plate as b {\n        placement = { placement = offset, offset = { translation = [100 mm, 0 mm, 0 mm] } }\n    }\n}\n";

#[test]
fn removing_or_suppressing_an_occurrence_with_a_body_changes_only_that_occurrence() {
    let parsed = parse_rig(TWO_PLATES);
    let original: AssemblyDocument = parsed.document.clone().expect("reads");
    // Remove `b`: its statement and block go, nothing else changes.
    let mut d = original.clone();
    d.occurrences.retain(|o| o.name.as_str() != "b");
    let change = syntax::edit::diff_source_in(
        &syntax::ASSEMBLY,
        "rig.assembly",
        TWO_PLATES,
        &d,
        &parsed.uses,
        &[],
        &rig_scope(""),
    )
    .expect("an edit");
    assert!(!change.reprinted, "{}", change.source);
    assert_eq!(
        change.source,
        "runmat 0.7\nstd 1\n\nassembly rig {\n    use part plate as a\n}\n"
    );
    // Suppress `b`: only its block changes.
    let mut d = original.clone();
    let b = d
        .occurrences
        .iter_mut()
        .find(|o| o.name.as_str() == "b")
        .expect("b");
    b.suppressed = true;
    let change = syntax::edit::diff_source_in(
        &syntax::ASSEMBLY,
        "rig.assembly",
        TWO_PLATES,
        &d,
        &parsed.uses,
        &[],
        &rig_scope(""),
    )
    .expect("an edit");
    assert!(!change.reprinted, "{}", change.source);
    assert!(change.source.starts_with(
        "runmat 0.7\nstd 1\n\nassembly rig {\n    use part plate as a\n    use part plate as b {\n"
    ));
    assert!(
        change.source.contains("suppressed = true"),
        "{}",
        change.source
    );
}

#[test]
fn names_made_from_other_names_spell_them_out_and_read_back() {
    use runmat_cad::refs::{EntityName, Reference};
    use runmat_cad::syntax::refs::print_reference;
    let text = r#"part p {
    sketch s1 on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        l2 = line(start = l1.end, end = [80 mm, 50 mm])
        l3 = line(start = l2.end, end = [0 mm, 50 mm])
        l4 = line(start = l3.end, end = l1.start)
    }
    feature body = extrude(profile = sketch.s1, distance = 25 mm)
    feature hollow = shell(body = feature.body.body, open_faces = [feature.body.face.start], thickness = 2 mm)
    feature round = fillet(edges = [edges_between(feature.hollow.face.inner(feature.body.face.side(s1.l1)), feature.hollow.face.rim(feature.body.face.side(s1.l1)))], radius = 1 mm)
    feature ch = chamfer(edges = [edges_between(feature.round.face.blend(between = [feature.body.face.end, feature.body.face.side(s1.l2)]), feature.body.face.end)], distance = 1 mm, angle = 45 deg)
}
"#;
    let parsed = parse_part("p.part", text);
    assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
    let document = parsed.document.unwrap();
    // The model holds tokens; printing spells each token as the name it stands for.
    let printed = print_part(&document, &[]).unwrap();
    assert!(printed.contains("feature.hollow.face.inner(feature.body.face.side(s1.l1))"), "{printed}");
    assert!(printed.contains("feature.round.face.blend(between = ["), "{printed}");
    assert_eq!(parse_part("p.part", &printed).document.unwrap(), document);
    let inner = Reference::Name {
        kind: runmat_cad::refs::TopoKind::Face,
        name: EntityName::Generated(runmat_cad::refs::GenerationName {
            feature: runmat_cad::ids::FeatureId::new("hollow"),
            role: runmat_cad::refs::Role::ShellInner,
            tag: runmat_cad::refs::SourceTag::face_of(&EntityName::Generated(runmat_cad::refs::GenerationName {
                feature: runmat_cad::ids::FeatureId::new("body"),
                role: runmat_cad::refs::Role::EndCap,
                tag: runmat_cad::refs::SourceTag::None,
                ordinal: None,
                step: None,
            })),
            ordinal: None,
            step: None,
        }),
    };
    assert_eq!(print_reference(&inner).as_deref(), Some("feature.hollow.face.inner(feature.body.face.end)"));
    // A tag names faces only.
    let bad = text.replace("inner(feature.body.face.side(s1.l1))", "inner(feature.body.body)");
    assert!(!parse_part("p.part", &bad).diagnostics.is_empty());
}
