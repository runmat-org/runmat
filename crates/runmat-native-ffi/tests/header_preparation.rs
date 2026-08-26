use std::path::PathBuf;
use std::process::Command;

use runmat_native_ffi::{prepare_header, HeaderPreparation, NativeType, ParameterDirection};

fn clang_available() -> bool {
    Command::new("clang")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
}

#[test]
fn compiler_frontend_prepares_c_declarations() {
    if !clang_available() {
        eprintln!("skipping header preparation because clang is unavailable");
        return;
    }
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let metadata = prepare_header(&HeaderPreparation {
        header: crate_root.join("tests/fixtures/interface.h"),
        library_name: "fixture".into(),
        library_path: "libfixture.so".into(),
        target_triple: target_lexicon::HOST.to_string(),
        clang: "clang".into(),
        include_directories: Vec::new(),
        definitions: Vec::new(),
    })
    .expect("prepared metadata");

    assert_eq!(metadata.libraries.len(), 1);
    assert_eq!(metadata.libraries[0].symbols.len(), 6);
    assert_eq!(metadata.structures.len(), 1);
    assert_eq!(metadata.structures[0].fields.len(), 2);
    assert_eq!(metadata.enumerations.len(), 1);
    assert!(metadata.aliases.iter().any(|alias| {
        alias.name == "fixture_record"
            && matches!(
                alias.target,
                NativeType::Structure { ref name } if name == "fixture_record"
            )
    }));
    let tag = metadata.libraries[0]
        .symbols
        .iter()
        .find(|symbol| symbol.name == "fixture_tag")
        .expect("fixture_tag prototype");
    assert_eq!(tag.parameters[0].direction, ParameterDirection::Input);
}
