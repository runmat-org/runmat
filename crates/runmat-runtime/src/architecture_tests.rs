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

#[test]
fn structure_arrays_do_not_regain_content_based_cell_identity() {
    let root = workspace_root();
    let banned = [
        "cell.data.iter().all(|item|matches!(item,Value::Struct",
        "cell.data.iter().all(|value|matches!(value,Value::Struct",
        "cell.data.iter().all(|v|matches!(v,Value::Struct",
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
            for (offset, _) in compact.match_indices(".iter().all(") {
                let start = offset.saturating_sub(240);
                let end = compact.len().min(offset.saturating_add(640));
                let before = String::from_utf8_lossy(&compact.as_bytes()[start..offset]);
                let after = String::from_utf8_lossy(&compact.as_bytes()[offset..end]);
                assert!(
                    !(before.contains("Value::Cell")
                        && after.contains("matches!(")
                        && after.contains("Value::Struct")),
                    "{} infers structure identity by inspecting every cell element",
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
                    let describes_structure_cell_carrier = [
                        "struct array",
                        "structure array",
                        "struct-array",
                        "structure-array",
                    ]
                    .iter()
                    .any(|identity| line.contains(identity))
                        && line.contains("cell")
                        && [
                            "represent",
                            "backed",
                            "instead of",
                            "does not yet have",
                            "structure-array container",
                            "struct array container",
                        ]
                        .iter()
                        .any(|marker| line.contains(marker));
                    assert!(
                        !describes_structure_cell_carrier,
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
