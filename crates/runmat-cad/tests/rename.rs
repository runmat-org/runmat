//! Rename: the declaration and every reference follow, in the part and in drawings of it.

use runmat_cad::modules::ModuleScope;
use runmat_cad::syntax::edit::EditFailure;
use runmat_cad::syntax::parse_part;
use runmat_cad::syntax::rename::{rename_in_drawing, rename_part_item, RenameKind};

const PLATE: &str = r#"runmat 0.7
std 1

part plate {
    input width : length = 40 mm
    sketch s1 on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])
        l2 = line(start = [input.width, 0 mm], end = [40 mm, 20 mm])
        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])
        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])
    }
    // The plate.
    feature body = extrude(profile = sketch.s1, distance = 10 mm)
    feature mid = plane.offset(base = origin.xy, distance = 5 mm)
    sketch s2 on plane.mid {
        c1 = circle(center = [10 mm, 10 mm], radius = 2 mm)
    }
    feature hole = extrude(profile = sketch.s2, distance = 10 mm)
    feature round = fillet(edges = [edges_between(feature.body.face.side(s1.l1), feature.body.face.end)], radius = 1 mm)
    region top : face = { select = [feature.body.face.end], expect = exactly_one }
    publish deck = region(region = top)
    configuration short = { width = 30 mm, suppress = [feature.round] }
    rollback_after = round
}
"#;

fn rename(kind: RenameKind, from: &str, to: &str) -> String {
    let change = rename_part_item(PLATE, "/p/plate.part", kind, from, to)
        .unwrap_or_else(|e| panic!("{from} → {to}: {e}"));
    let parsed = parse_part("/p/plate.part", &change.source);
    assert!(
        parsed.is_valid(),
        "{:?}\n{}",
        parsed.diagnostics,
        change.source
    );
    assert!(!change.reprinted, "{}", change.source);
    change.source
}

#[test]
fn a_feature_rename_rewrites_its_declaration_and_every_reference() {
    let text = rename(RenameKind::Feature, "body", "slab");
    assert!(text.contains("feature slab = extrude("), "{text}");
    assert!(text.contains("feature.slab.face.side(s1.l1)"), "{text}");
    assert!(text.contains("select = [feature.slab.face.end]"), "{text}");
    assert!(!text.contains("body"), "{text}");
    // Comments and untouched lines stay as written.
    assert!(text.contains("    // The plate.\n"), "{text}");
}

#[test]
fn sketch_plane_and_rolled_back_feature_renames_follow_their_forms() {
    let text = rename(RenameKind::Feature, "s1", "base");
    assert!(text.contains("sketch base on origin.xy"), "{text}");
    assert!(text.contains("profile = sketch.base"), "{text}");
    assert!(text.contains("side(base.l1)"), "{text}");
    let text = rename(RenameKind::Feature, "mid", "datum_mid");
    assert!(text.contains("sketch s2 on plane.datum_mid"), "{text}");
    let text = rename(RenameKind::Feature, "round", "edge_round");
    assert!(text.contains("suppress = [feature.edge_round]"), "{text}");
    assert!(text.contains("rollback_after = edge_round"), "{text}");
}

#[test]
fn input_and_region_renames_follow_their_references() {
    let text = rename(RenameKind::Input, "width", "span");
    assert!(text.contains("input span : length"), "{text}");
    assert!(text.contains("end = [input.span, 0 mm]"), "{text}");
    assert!(text.contains("span = 30 mm"), "{text}");
    assert!(!text.contains("width"), "{text}");
    let text = rename(RenameKind::Region, "top", "deck_face");
    assert!(text.contains("region deck_face : face"), "{text}");
    assert!(text.contains("region = deck_face"), "{text}");
}

#[test]
fn a_clash_or_a_bad_name_is_refused_with_the_reason() {
    let refuse = |kind, from, to| match rename_part_item(PLATE, "/p/plate.part", kind, from, to) {
        Err(EditFailure::Model(m)) => m,
        other => panic!("{to}: {other:?}"),
    };
    assert!(refuse(RenameKind::Feature, "body", "hole").contains("already has a feature `hole`"));
    assert!(refuse(RenameKind::Feature, "body", "2nd").contains("isn't a name"));
    assert!(refuse(RenameKind::Feature, "body", "my body").contains("isn't a name"));
    assert!(refuse(RenameKind::Feature, "body", "sketch").contains("reserved"));
    assert!(refuse(RenameKind::Feature, "nothing", "x").contains("has no feature `nothing`"));
    assert!(refuse(RenameKind::Input, "width", "").contains("type a name"));
}

#[test]
fn a_drawing_of_the_part_follows_a_feature_rename() {
    let drawing = r#"runmat 0.7
std 1

drawing plate {
    format a3 = { size = { size = iso, name = A3 }, title_block = [] }
    sheet "1" = { format = a3, scale = [1, 1], projection = third_angle }
    view front = standard(orientation = front, sheet = "1", model = { part = plate }, at = [100, 100], display = visible_lines)
    view a_a = section(from = front, plane = feature.body.face.start, label = A, sheet = "1", model = { part = plate }, at = [200, 100], display = visible_lines)
    annotation t = dimension(style = linear, references = [feature.body.face.start, feature.body.face.end], view = front)
}
"#;
    let scope = ModuleScope::from_files([("/p/plate.part", PLATE), ("/p/plate.drawing", drawing)]);
    let change = rename_in_drawing(
        drawing,
        "/p/plate.drawing",
        &scope,
        "/p/plate.part",
        "body",
        "slab",
    )
    .unwrap()
    .expect("the drawing names the feature");
    assert!(
        change.source.contains("plane = feature.slab.face.start"),
        "{}",
        change.source
    );
    assert!(
        change
            .source
            .contains("references = [feature.slab.face.start, feature.slab.face.end]"),
        "{}",
        change.source
    );
    // Another part's drawing isn't touched.
    let other = rename_in_drawing(
        drawing,
        "/p/plate.drawing",
        &scope,
        "/p/other.part",
        "body",
        "slab",
    )
    .unwrap();
    assert!(other.is_none());
}

/// Every feature of every corpus part renames, and no reference is left on the old name.
#[test]
fn every_corpus_feature_renames_cleanly() {
    fn parts(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                parts(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("part") {
                out.push(path);
            }
        }
    }
    let mut files = Vec::new();
    parts(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus"),
        &mut files,
    );
    let mut renamed = 0;
    for path in files {
        let text = std::fs::read_to_string(&path).unwrap();
        let file = path.to_str().unwrap();
        let parsed = parse_part(file, &text);
        let Some(document) = parsed.document.filter(|_| parsed.diagnostics.is_empty()) else {
            continue;
        };
        // Up to a dozen features per file, spread through the history (each rename reparses).
        let step = document.features.len().div_ceil(12).max(1);
        for feature in document.features.iter().step_by(step) {
            let from = feature.id.0.as_str();
            let to = (1..)
                .map(|n| format!("{from}_{n}"))
                .find(|n| !document.features.iter().any(|f| f.id.0 == *n))
                .unwrap();
            let change = rename_part_item(&text, file, RenameKind::Feature, from, &to)
                .unwrap_or_else(|e| panic!("{file}: {from}: {e}"));
            let after = parse_part(file, &change.source);
            let new = after.document.expect("reads back");
            assert!(new.features.iter().any(|f| f.id.0 == to), "{file}: {from}");
            assert!(
                !new.features.iter().any(|f| f.id.0 == from),
                "{file}: {from}"
            );
            for head in ["feature", "sketch", "plane", "axis", "point"] {
                let old = format!("{head}.{from}");
                let left = change.source.match_indices(&old).find(|(at, _)| {
                    !change.source[at + old.len()..]
                        .starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_')
                });
                if let Some((at, _)) = left {
                    let line = change.source[..at].lines().count();
                    let text = change.source.lines().nth(line - 1).unwrap_or("");
                    panic!("{file}: `{old}` is left after renaming {from}, line {line}: {text}");
                }
            }
            renamed += 1;
        }
    }
    assert!(renamed > 15, "only {renamed} features renamed");
}

const BRACKET: &str = r#"runmat 0.7
std 1

part bracket {
    input width : length = 40 mm
    sketch s1 on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])
        l2 = line(start = [input.width, 0 mm], end = [input.width, 20 mm])
        l3 = line(start = [input.width, 20 mm], end = [0 mm, 20 mm])
        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s1, distance = 10 mm)
    region mount : face = { select = [feature.body.face.start], expect = exactly_one }
    output bracket = { body = feature.body.body }
}
"#;

/// Another part with an input of the same name: its values never follow.
const SPACER: &str = r#"runmat 0.7
std 1

part spacer {
    input width : length = 5 mm
    sketch s1 on origin.xy {
        c1 = circle(center = [0 mm, 0 mm], radius = input.width)
    }
    feature body = extrude(profile = sketch.s1, distance = 2 mm)
    output spacer = { body = feature.body.body }
}
"#;

const SUB: &str = r#"runmat 0.7
std 1

assembly sub {
    use part bracket as left {
        configuration = { select = inputs, values = { width = 30 mm } }
    }
    use part spacer as gap {
        configuration = { select = inputs, values = { width = 6 mm } }
    }
}
"#;

const TOP: &str = r#"runmat 0.7
std 1

assembly top {
    use assembly sub as unit {
        override left {
            input.width = 25 mm
        }
        override gap {
            input.width = 7 mm
        }
    }
    use part bracket as spare
}
"#;

#[test]
fn an_input_rename_reaches_assemblies_that_set_it() {
    use runmat_cad::syntax::rename::{assemblies_in, rename_input_in_assembly};
    let files = [
        ("/p/bracket.part", BRACKET),
        ("/p/spacer.part", SPACER),
        ("/p/sub.assembly", SUB),
        ("/p/top.assembly", TOP),
    ];
    let scope = ModuleScope::from_files(files);
    let assemblies = assemblies_in(files, &scope);
    assert_eq!(assemblies.len(), 2, "both assemblies read");
    let follow = |path: &str, text: &str| {
        rename_input_in_assembly(
            text,
            path,
            &scope,
            &assemblies,
            "/p/bracket.part",
            "width",
            "span",
        )
        .unwrap()
    };
    // The occurrence's own values follow; the spacer's input of the same name stays.
    let sub = follow("/p/sub.assembly", SUB).expect("sub sets the bracket's width");
    assert!(
        sub.source.contains("values = { span = 30 mm }"),
        "{}",
        sub.source
    );
    assert!(
        sub.source.contains("values = { width = 6 mm }"),
        "{}",
        sub.source
    );
    // An override whose target resolves through the sub-assembly to the bracket follows.
    let top = follow("/p/top.assembly", TOP).expect("top overrides the bracket's width");
    assert!(top.source.contains("input.span = 25 mm"), "{}", top.source);
    assert!(top.source.contains("input.width = 7 mm"), "{}", top.source);
    assert!(!top.reprinted, "{}", top.source);
    // An assembly that sets no input of the part isn't touched.
    let none = rename_input_in_assembly(
        SUB,
        "/p/sub.assembly",
        &scope,
        &assemblies,
        "/p/bracket.part",
        "depth",
        "d",
    )
    .unwrap();
    assert!(none.is_none());
}

#[test]
fn a_region_rename_reaches_studies_whose_geometry_reads_the_part() {
    use runmat_cad::syntax::rename::rename_region_in_study;
    let study = r#"geometry bracket {
    file = "../parts/bracket.part"
    units = mm
    region mount : face { select name "mount"; expect exactly_one }
}

geometry scan {
    file = "./scan.step"
    units = mm
    region mount : face { select name "mount"; expect exactly_one }
}

study bracket_static {
    geometry = geometry.bracket
    analysis static { structural static {} }
}
"#;
    let change =
        rename_region_in_study(study, "/s/load.fea", "/parts/bracket.part", "mount", "base")
            .unwrap()
            .expect("the study reads the bracket's region");
    // The bracket geometry's selection follows; the study's own region name and the STEP
    // geometry's label of the same spelling stay.
    assert_eq!(change.edits.len(), 1, "{}", change.source);
    assert!(
        change.source.contains(
            "region mount : face { select name \"base\"; expect exactly_one }\n}\n\ngeometry scan"
        ),
        "{}",
        change.source
    );
    assert!(change.source.contains("file = \"./scan.step\"\n    units = mm\n    region mount : face { select name \"mount\""), "{}", change.source);
    // A study of another part isn't touched.
    let other =
        rename_region_in_study(study, "/s/load.fea", "/parts/plate.part", "mount", "base").unwrap();
    assert!(other.is_none());
}
