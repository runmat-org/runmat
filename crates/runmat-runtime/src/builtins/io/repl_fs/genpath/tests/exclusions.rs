use std::fs;

use runmat_value::Value;
use tempfile::tempdir;

use super::support::{call, canonical, segments};

#[test]
fn skips_special_folders_and_their_descendants() {
    let directory = tempdir().expect("directory");
    for relative in [
        "keep/child",
        "private/child",
        "resources",
        "@Class",
        "+package",
    ] {
        fs::create_dir_all(directory.path().join(relative)).expect("folder");
    }
    let result = segments(
        call(vec![Value::String(
            directory.path().to_string_lossy().into(),
        )])
        .expect("genpath"),
    );
    assert_eq!(
        result,
        vec![
            canonical(directory.path()),
            canonical(&directory.path().join("keep")),
            canonical(&directory.path().join("keep/child")),
        ]
    );
}

#[test]
fn resolves_relative_exclusions_from_the_root() {
    let directory = tempdir().expect("directory");
    fs::create_dir(directory.path().join("keep")).expect("keep");
    fs::create_dir_all(directory.path().join("skip/child")).expect("skip");
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    let result = segments(
        call(vec![
            Value::String(directory.path().to_string_lossy().into()),
            Value::String("skip".into()),
        ])
        .expect("genpath"),
    );
    assert_eq!(
        result,
        vec![
            canonical(directory.path()),
            canonical(&directory.path().join("keep"))
        ]
    );
}

#[test]
fn excluding_a_parent_omits_its_entire_subtree() {
    let directory = tempdir().expect("directory");
    fs::create_dir_all(directory.path().join("alpha/child")).expect("alpha");
    fs::create_dir(directory.path().join("beta")).expect("beta");
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    let result = segments(
        call(vec![
            Value::String(directory.path().to_string_lossy().into()),
            Value::String(canonical(&directory.path().join("alpha"))),
        ])
        .expect("genpath"),
    );
    assert_eq!(
        result,
        vec![
            canonical(directory.path()),
            canonical(&directory.path().join("beta"))
        ]
    );
}
