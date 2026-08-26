use std::path::PathBuf;
use std::process::Command;

use runmat_native_ffi::{
    prepare_header, prepare_header_with_declarations, HeaderPreparation, NativeType,
    ParameterDirection,
};

fn clang_available() -> bool {
    Command::new("clang")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
}

#[test]
fn additional_included_headers_can_join_the_declared_interface() {
    if !clang_available() {
        eprintln!("skipping header preparation because clang is unavailable");
        return;
    }
    let temporary = tempfile::tempdir().unwrap();
    let base = temporary.path().join("base.h");
    let detail = temporary.path().join("detail.h");
    std::fs::write(&base, "#include \"detail.h\"\nint base_value(int input);\n").unwrap();
    std::fs::write(&detail, "int detail_value(int input);\n").unwrap();
    let preparation = HeaderPreparation {
        header: base,
        library_name: "fixture".into(),
        library_path: "libfixture.so".into(),
        target_triple: target_lexicon::HOST.to_string(),
        clang: "clang".into(),
        include_directories: vec![temporary.path().to_path_buf()],
        definitions: Vec::new(),
    };
    let base_only = prepare_header(&preparation).unwrap();
    assert_eq!(base_only.libraries[0].symbols.len(), 1);
    let expanded = prepare_header_with_declarations(&preparation, &["detail".into()]).unwrap();
    assert_eq!(
        expanded.libraries[0]
            .symbols
            .iter()
            .map(|symbol| symbol.name.as_str())
            .collect::<Vec<_>>(),
        ["base_value", "detail_value"]
    );
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
    assert_eq!(metadata.libraries[0].symbols.len(), 7);
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
    let pointer = metadata.libraries[0]
        .symbols
        .iter()
        .find(|symbol| symbol.name == "fixture_borrowed_value")
        .expect("pointer-returning prototype");
    assert_eq!(
        pointer.return_ownership,
        Some(runmat_native_ffi::PointerOwnership::Borrowed)
    );
    assert!(pointer.return_nullable);
}
