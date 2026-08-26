#![cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]

use std::path::{Path, PathBuf};
use std::process::Command;

use runmat_core::RunMatSession;
use runmat_native_ffi::NativeInterfaceArtifactManifest;
use runmat_value::{IntValue, Value};

fn compile_fixture(directory: &Path) -> Option<PathBuf> {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../runmat-runtime/tests/fixtures/native_ffi/interface.c");
    let output = directory.join(if cfg!(target_os = "macos") {
        "libcore_fixture.dylib"
    } else if cfg!(target_os = "windows") {
        "core_fixture.dll"
    } else {
        "libcore_fixture.so"
    });
    let mut command = Command::new(if cfg!(target_os = "windows") {
        "gcc"
    } else {
        "clang"
    });
    if cfg!(target_os = "macos") {
        command.arg("-dynamiclib");
    } else if cfg!(target_os = "windows") {
        command.arg("-shared");
    } else {
        command.args(["-shared", "-fPIC"]);
    }
    let status = command.arg(source).arg("-o").arg(&output).status().ok()?;
    status.success().then_some(output)
}

#[test]
fn legacy_shared_library_calls_use_the_session_foreign_runtime() {
    if Command::new("clang").arg("--version").output().is_err() {
        eprintln!("skipping native FFI integration test because clang is unavailable");
        return;
    }
    let temporary = tempfile::tempdir().unwrap();
    let Some(library_path) = compile_fixture(temporary.path()) else {
        eprintln!("skipping native FFI integration test because the C compiler is unavailable");
        return;
    };
    let header_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../runmat-runtime/tests/fixtures/native_ffi/interface.h");
    let source = format!(
        "[notfound, header_warnings] = loadlibrary('{}', '{}', 'alias', 'core_fixture');\n\
         total = calllib('core_fixture', 'fixture_add', int32(19), int32(23));\n\
         clibgen.buildInterface('{}', 'Libraries', '{}', 'InterfaceName', 'modern_fixture');\n\
         modern_total = clib.modern_fixture.fixture_add(int32(20), int32(22));\n\
         pointer = libpointer('int32Ptr', int32(40));\n\
         pointer.Value = int32(50);\n\
         pointer_type = pointer.DataType;\n\
         pointer = calllib('core_fixture', 'fixture_increment', pointer);\n\
         pointer_value = pointer.Value;\n\
         opaque = calllib('core_fixture', 'fixture_borrowed_value', int32(1));\n\
         setdatatype(opaque, 'int32Ptr', 1, 1);\n\
         opaque_value = opaque.Value;\n\
         opaque_method = calllib('core_fixture', 'fixture_borrowed_value', int32(1));\n\
         opaque_method.setdatatype('int32Ptr', 1, 1);\n\
         opaque_method_value = opaque_method.Value;\n\
         record = libstruct('fixture_record');\n\
         record.left = int32(19);\n\
         record.right = int32(23);\n\
         record_total = calllib('core_fixture', 'fixture_record_sum', record);\n\
         loaded = libisloaded('core_fixture');\n\
         record_total",
        library_path.display(),
        header_path.display(),
        header_path.display(),
        library_path.display()
    );
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let result = runmat_core::execute_text_request_for_testing(&mut session, &source).unwrap();
    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(result.value, Some(Value::Int(IntValue::I32(42))));
    assert!(result
        .workspace
        .values
        .iter()
        .any(|entry| entry.name == "pointer" && entry.class_name == "lib.pointer"));
    assert!(result
        .workspace
        .values
        .iter()
        .any(|entry| entry.name == "record" && entry.class_name == "lib.fixture_record"));
    assert!(result
        .workspace
        .values
        .iter()
        .any(|entry| entry.name == "loaded" && entry.class_name == "logical"));
    assert!(result
        .workspace
        .values
        .iter()
        .any(|entry| entry.name == "pointer_type" && entry.class_name == "string"));
    assert!(result
        .workspace
        .values
        .iter()
        .any(|entry| entry.name == "notfound" && entry.class_name == "cell"));
    assert!(result
        .workspace
        .values
        .iter()
        .any(|entry| { entry.name == "header_warnings" && entry.class_name == "string" }));
    assert!(result.workspace.values.iter().any(|entry| {
        entry.name == "opaque_value" && entry.class_name == "int32" && entry.shape == vec![1, 1]
    }));
    assert!(result.workspace.values.iter().any(|entry| {
        entry.name == "opaque_method_value"
            && entry.class_name == "int32"
            && entry.shape == vec![1, 1]
    }));
    assert!(library_path
        .with_extension(format!(
            "{}.runmat.json",
            library_path.extension().unwrap().to_string_lossy()
        ))
        .is_file());
}

#[test]
fn prepared_interface_is_installed_and_admitted_before_session_execution() {
    if Command::new("clang").arg("--version").output().is_err() {
        eprintln!("skipping native FFI integration test because clang is unavailable");
        return;
    }
    let temporary = tempfile::tempdir().unwrap();
    let Some(library_path) = compile_fixture(temporary.path()) else {
        eprintln!("skipping native FFI integration test because the C compiler is unavailable");
        return;
    };
    let header_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../runmat-runtime/tests/fixtures/native_ffi/interface.h");
    let preparation = format!(
        "clibgen.buildInterface('{}', 'Libraries', '{}', 'InterfaceName', 'prepared_fixture');",
        header_path.display(),
        library_path.display()
    );
    let mut preparing_session = RunMatSession::with_options(false, false).unwrap();
    let result =
        runmat_core::execute_text_request_for_testing(&mut preparing_session, &preparation)
            .unwrap();
    assert!(result.error.is_none(), "{:?}", result.error);

    let manifest_path = NativeInterfaceArtifactManifest::path_for_library(&library_path);
    let manifest = NativeInterfaceArtifactManifest::read(&manifest_path).unwrap();
    let mut execution_session = RunMatSession::with_options(false, false).unwrap();
    assert_eq!(
        execution_session
            .install_native_interface_artifact(&library_path, &manifest_path)
            .unwrap(),
        "prepared_fixture"
    );
    execution_session
        .admit_interop_manifest(&manifest.interop_manifest())
        .unwrap();
    let result = runmat_core::execute_text_request_for_testing(
        &mut execution_session,
        "total = clib.prepared_fixture.fixture_add(int32(20), int32(22)); total",
    )
    .unwrap();
    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(result.value, Some(Value::Int(IntValue::I32(42))));
}
