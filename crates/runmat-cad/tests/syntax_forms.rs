//! The canonical surface forms: every feature kind, sketch entity and constraint, and
//! assembly occurrence round-trips through text, and files in the old generic forms still
//! read and format to the canonical forms.

mod common;
use common::*;
use runmat_cad::assembly::AssemblyDocument;
use runmat_cad::drawing::DrawingDocument;
use runmat_cad::model::*;
use runmat_cad::syntax::{self, parse_assembly, parse_drawing, parse_part, print_part};
use std::collections::BTreeSet;
use std::path::PathBuf;

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
    let text = std::fs::read_to_string(corpus().join("features.json")).unwrap();
    let mut features: Vec<Feature> = serde_json::from_str(&text).unwrap();
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
        .replace(
            "mode = cut, reverse)",
            "mode = { mode = cut }, reverse = true)",
        );
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
    use part bracket from "parts/bracket.part" as left {
        configuration = { select = named, name = light }
        fixed = false
        suppressed = true
    }
    use part plate from "parts/bracket.part" @ "R3" as right
    use assembly gearbox from "gearbox.assembly" as gb
}
"#;
    let parsed = parse_assembly("rig.assembly", text);
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
        printed.contains("    use part bracket from \"parts/bracket.part\" as left {\n"),
        "{printed}"
    );
    assert!(
        printed.contains("    use part plate from \"parts/bracket.part\" @ \"R3\" as right\n"),
        "{printed}"
    );
    let back = parse_assembly("rig.assembly", &printed);
    assert_eq!(back.document.as_ref(), Some(&d), "{printed}");
    assert!(back.spans.spans.contains_key("occurrence.left"));
}

/// Old-form files (the generic record forms, as printed before the surface forms) read as
/// the same model and format to the canonical forms.
#[test]
fn old_form_files_read_and_format_to_the_canonical_forms() {
    let dir = corpus().join("old-form");
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
        let formatted = syntax::format(&file, &source).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            syntax::format(&file, &formatted).unwrap(),
            formatted,
            "{name}"
        );
        let check = |text: &str| -> serde_json::Value {
            if name.ends_with(".part") {
                let p = parse_part(&file, text);
                assert!(p.is_valid(), "{name}: {:#?}", p.diagnostics);
                serde_json::to_value(p.document.unwrap()).unwrap()
            } else if name.ends_with(".assembly") {
                let p = parse_assembly(&file, text);
                assert!(p.is_valid(), "{name}: {:#?}", p.diagnostics);
                serde_json::to_value(p.document.unwrap()).unwrap()
            } else {
                let p = parse_drawing(&file, text);
                assert!(p.is_valid(), "{name}: {:#?}", p.diagnostics);
                serde_json::to_value::<DrawingDocument>(p.document.unwrap()).unwrap()
            }
        };
        assert_eq!(
            check(&source),
            model,
            "{name}: the old form reads as the model"
        );
        assert_eq!(
            check(&formatted),
            model,
            "{name}: the formatted file reads as the model"
        );
        if source.contains("= extrude {") {
            assert!(formatted.contains("= extrude("), "{name}\n{formatted}");
        }
        count += 1;
    }
    assert!(count >= 35, "{count}");
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
