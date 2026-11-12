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
    assert!(parsed.is_valid(), "{:?}\n{}", parsed.diagnostics, change.source);
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
    let scope = ModuleScope::from_files([
        ("/p/plate.part", PLATE),
        ("/p/plate.drawing", drawing),
    ]);
    let change = rename_in_drawing(drawing, "/p/plate.drawing", &scope, "/p/plate.part", "body", "slab")
        .unwrap()
        .expect("the drawing names the feature");
    assert!(change.source.contains("plane = feature.slab.face.start"), "{}", change.source);
    assert!(
        change.source.contains("references = [feature.slab.face.start, feature.slab.face.end]"),
        "{}",
        change.source
    );
    // Another part's drawing isn't touched.
    let other = rename_in_drawing(drawing, "/p/plate.drawing", &scope, "/p/other.part", "body", "slab").unwrap();
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
    for dir in ["tests/corpus"] {
        parts(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(dir), &mut files);
    }
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
            assert!(!new.features.iter().any(|f| f.id.0 == from), "{file}: {from}");
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
