use std::path::{Path, PathBuf};

fn visit_files(root: &Path, extension: &str, visitor: &mut impl FnMut(&Path, &str)) {
    let entries = std::fs::read_dir(root)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", root.display()));
    for entry in entries {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            visit_files(&path, extension, visitor);
        } else if path.extension().and_then(|value| value.to_str()) == Some(extension) {
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
            visitor(&path, &source);
        }
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("runtime crate belongs to the workspace")
        .to_path_buf()
}

fn describes_removed_structure_cell_carrier(line: &str) -> bool {
    const STRUCTURE_ARRAY_NAMES: [&str; 4] = [
        "struct array",
        "struct arrays",
        "structure array",
        "structure arrays",
    ];
    const CELL_CARRIER_PREDICATES: [&str; 10] = [
        "as cell",
        "represented as cell",
        "represented by cell",
        "represented using cell",
        "stored in cell",
        "implemented as cell",
        "backed by cell",
        "uses cells as",
        "use cells as",
        "uses cell arrays as",
    ];

    // A source line can contain multiple prose claims (including a FAQ question
    // and answer). Only a relation within one clause describes the carrier.
    line.split(['.', '?', '!', ';']).any(|clause| {
        let normalized = clause.to_ascii_lowercase().replace('-', " ");
        STRUCTURE_ARRAY_NAMES.iter().any(|identity| {
            normalized.match_indices(identity).any(|(offset, _)| {
                let before = &normalized[..offset];
                let after = normalized[offset + identity.len()..].trim_start();
                let predicate = after
                    .strip_prefix("is ")
                    .or_else(|| after.strip_prefix("are "))
                    .unwrap_or(after);
                let predicate = predicate.strip_prefix("a ").unwrap_or(predicate);
                let cell_noun = matches!(
                    predicate.trim_matches([' ', ',', ':']),
                    "cell" | "cells" | "cell array" | "cell arrays"
                );
                let cells_instead_of_container = predicate
                    .strip_prefix("cells instead of ")
                    .is_some_and(|remainder| {
                        remainder.starts_with("dedicated container")
                            || remainder.starts_with("struct array container")
                            || remainder.starts_with("structure array container")
                    });
                let names_container =
                    after.starts_with("container") || after.starts_with("carrier");
                let denies_container = ((after.contains("does not yet have")
                    || after.contains("do not yet have"))
                    && (after.contains("container") || after.contains("carrier")))
                    || (names_container
                        && (before.ends_with("no ") || after.contains("does not yet exist")));
                denies_container
                    || cell_noun
                    || cells_instead_of_container
                    || CELL_CARRIER_PREDICATES
                        .iter()
                        .any(|carrier| predicate.starts_with(carrier))
                    || before.ends_with("cell backed ")
                    || (before.contains("represent ") && after.starts_with("with cell"))
                    || before.ends_with("cell arrays represent ")
            })
        })
    })
}

#[test]
fn structure_cell_carrier_guard_distinguishes_carriers_from_cell_fallbacks() {
    for claim in [
        "Struct arrays are represented as cells.",
        "We represent structure arrays with cell arrays.",
        "A cell-backed struct array remains a cell.",
        "Struct arrays are backed by cells.",
        "Struct arrays are represented using cells.",
        "A structure array is stored in cells.",
        "A struct array is a cell.",
        "Struct arrays are cells instead of dedicated containers.",
        "Struct arrays do not yet have a dedicated container; use cells.",
        "There is no structure-array container; struct arrays use cells as their carrier.",
        "Struct arrays as cells are an implementation detail.",
    ] {
        assert!(describes_removed_structure_cell_carrier(claim), "{claim}");
    }
    for accurate in [
        "How are JSON objects represented? Objects become scalar structs. Rectangular arrays with common fields become struct arrays; differing fields remain cells.",
        "Rectangular object arrays sharing one field set become typed struct arrays; differing field sets remain cells of scalar structs.",
        "Differing-field objects remain cells instead of struct arrays.",
        "Cell-valued fields contribute to struct-array elements.",
        "Ordinary cell arrays are not structure arrays, regardless of their contents.",
        "Struct array fields may contain cell values.",
        "Struct arrays are cell-compatible containers.",
    ] {
        assert!(!describes_removed_structure_cell_carrier(accurate), "{accurate}");
    }
}

#[test]
fn structure_arrays_do_not_regain_content_based_cell_identity() {
    let root = workspace_root();
    // Guard the retired carrier idioms themselves. Proximity between a cell,
    // an unrelated `all` predicate and a later struct is not evidence that
    // code derives structure identity from that cell's contents.
    let banned = [
        "cell.data.iter().all(|item|matches!(item,Value::Struct",
        "cell.data.iter().all(|value|matches!(value,Value::Struct",
        "cell.data.iter().all(|v|matches!(v,Value::Struct",
        "cell.data().iter().all(|item|matches!(item,Value::Struct",
        "cell.data().iter().all(|value|matches!(value,Value::Struct",
        "cell.data().iter().all(|v|matches!(v,Value::Struct",
    ];
    for relative in [
        "crates/runmat-value/src",
        "crates/runmat-runtime/src",
        "crates/runmat-builtins/src",
    ] {
        visit_files(&root.join(relative), "rs", &mut |path, source| {
            if path.file_name().and_then(|name| name.to_str()) == Some("architecture_tests.rs") {
                return;
            }
            let compact = source
                .chars()
                .filter(|character| !character.is_whitespace())
                .collect::<String>();
            for pattern in banned {
                assert!(
                    !compact.contains(pattern),
                    "{} infers structure-array identity from cell contents",
                    path.display()
                );
            }
        });
    }
}

#[test]
fn structure_surfaces_do_not_describe_the_removed_cell_carrier() {
    let root = workspace_root();
    for relative in [
        "crates/runmat-runtime/src",
        "crates/runmat-builtins/src/catalog/entries",
        "docs/builtins/reference",
    ] {
        visit_files(
            &root.join(relative),
            if relative.ends_with("reference") {
                "json"
            } else {
                "rs"
            },
            &mut |path, source| {
                if path.file_name().and_then(|name| name.to_str()) == Some("architecture_tests.rs")
                {
                    return;
                }
                let lower = source.to_ascii_lowercase();
                for wording in [
                    "cell-backed struct",
                    "cell-backed structure",
                    "represented structure array",
                    "represented struct array",
                    "struct-array cell",
                ] {
                    assert!(
                        !lower.contains(wording),
                        "{} retains obsolete structure-array wording: {wording}",
                        path.display()
                    );
                }
                for line in lower.lines() {
                    assert!(
                        !describes_removed_structure_cell_carrier(line),
                        "{} retains an obsolete cell-carrier claim: {line}",
                        path.display()
                    );
                }
            },
        );
    }
    for relative in ["docs/VALUES.md", "docs/vm/indexing.md"] {
        let path = root.join(relative);
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        let lower = source.to_ascii_lowercase();
        assert!(
            !lower.contains("cell-backed struct")
                && !lower.contains("cell-backed structure")
                && !lower.contains("struct arrays as cells")
                && !lower.contains("structure arrays as cells"),
            "{} retains obsolete structure-array carrier wording",
            path.display()
        );
    }
}
