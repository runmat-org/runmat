#![cfg(not(target_family = "wasm"))]

use std::path::PathBuf;
use std::process::Command;

use runmat_native_ffi::{
    prepare_native_interface, HeaderPreparationError, NativeInterfacePreparation,
    NativeInterfacePreparationError,
};

fn clang_available() -> bool {
    Command::new("clang")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
}

#[test]
fn shared_preparation_preserves_distinct_names_headers_and_definitions() {
    if !clang_available() {
        eprintln!("skipping interface preparation because clang is unavailable");
        return;
    }
    let temporary = tempfile::tempdir().unwrap();
    let primary = temporary.path().join("primary.h");
    let additional = temporary.path().join("detail.h");
    let library = temporary.path().join("library.bin");
    std::fs::write(
        &primary,
        "#include \"detail.h\"\n#ifdef FEATURE_ENABLED\nint primary_value(int input);\n#endif\n",
    )
    .unwrap();
    std::fs::write(&additional, "int detail_value(int input);\n").unwrap();
    std::fs::write(&library, b"exact native library bytes").unwrap();

    let prepared = prepare_native_interface(&NativeInterfacePreparation {
        interface_name: "public_interface".into(),
        library_name: "logical_library".into(),
        library_path: library.clone(),
        primary_header: primary,
        additional_headers: vec![additional],
        include_directories: vec![temporary.path().to_path_buf()],
        definitions: vec!["FEATURE_ENABLED=1".into()],
        compiler_frontend: PathBuf::from("clang"),
    })
    .unwrap();

    assert_eq!(prepared.manifest.interface_name, "public_interface");
    assert_eq!(
        prepared.manifest.metadata.libraries[0].name,
        "logical_library"
    );
    assert_eq!(
        prepared.manifest.metadata.libraries[0]
            .symbols
            .iter()
            .map(|symbol| symbol.name.as_str())
            .collect::<Vec<_>>(),
        ["detail_value", "primary_value"]
    );
    prepared
        .manifest
        .validate_current_library(&std::fs::read(&library).unwrap())
        .unwrap();

    let repeated = prepare_native_interface(&NativeInterfacePreparation {
        interface_name: "public_interface".into(),
        library_name: "logical_library".into(),
        library_path: library,
        primary_header: temporary.path().join("primary.h"),
        additional_headers: vec![temporary.path().join("detail.h")],
        include_directories: vec![temporary.path().to_path_buf()],
        definitions: vec!["FEATURE_ENABLED=1".into()],
        compiler_frontend: PathBuf::from("clang"),
    })
    .unwrap();
    assert_eq!(prepared.manifest, repeated.manifest);
}

#[test]
fn shared_preparation_reports_frontend_and_library_failures() {
    if !clang_available() {
        eprintln!("skipping interface preparation because clang is unavailable");
        return;
    }
    let temporary = tempfile::tempdir().unwrap();
    let header = temporary.path().join("fixture.h");
    std::fs::write(&header, "int fixture_value(void);\n").unwrap();
    let request = NativeInterfacePreparation {
        interface_name: "fixture".into(),
        library_name: "fixture".into(),
        library_path: temporary.path().join("missing-library"),
        primary_header: header,
        additional_headers: Vec::new(),
        include_directories: Vec::new(),
        definitions: Vec::new(),
        compiler_frontend: PathBuf::from("clang"),
    };
    assert!(matches!(
        prepare_native_interface(&request),
        Err(NativeInterfacePreparationError::Artifact(_))
    ));

    let unavailable_frontend = NativeInterfacePreparation {
        compiler_frontend: temporary.path().join("absent-frontend"),
        ..request
    };
    assert!(matches!(
        prepare_native_interface(&unavailable_frontend),
        Err(NativeInterfacePreparationError::Header(
            HeaderPreparationError::Start { .. }
        ))
    ));
}
