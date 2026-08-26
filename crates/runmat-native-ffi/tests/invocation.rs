#![cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]

use std::path::{Path, PathBuf};
use std::process::Command;

use runmat_native_ffi::{
    invoke_symbol, invoke_symbol_with_bindings, invoke_symbol_with_callbacks, prepare_header,
    CallbackBinding, HeaderPreparation, InvocationValue, LoadedLibrary, NativePointerResource,
    NativeType, PointerBinding,
};
use runmat_value::{IntValue, IntegerStorage, StructValue, Tensor, Value};

fn compile_fixture(directory: &Path) -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = root.join("tests/fixtures/interface.c");
    let output = directory.join(if cfg!(target_os = "macos") {
        "libfixture.dylib"
    } else if cfg!(target_os = "windows") {
        "fixture.dll"
    } else {
        "libfixture.so"
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
    let status = command.arg(&source).arg("-o").arg(&output).status().ok()?;
    status.success().then_some(output)
}

fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    let Value::Struct(value) = value else {
        panic!("expected structure output");
    };
    value.fields.get(name).expect("field")
}

#[test]
fn pointer_resources_keep_a_stable_typed_value_across_calls() {
    let temporary = tempfile::tempdir().unwrap();
    let Some(library_path) = compile_fixture(temporary.path()) else {
        eprintln!("skipping native fixture because the C compiler is unavailable");
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let metadata = prepare_header(&HeaderPreparation {
        header: root.join("tests/fixtures/interface.h"),
        library_name: "fixture".into(),
        library_path: library_path.display().to_string(),
        target_triple: target_lexicon::HOST.to_string(),
        clang: "clang".into(),
        include_directories: Vec::new(),
        definitions: Vec::new(),
    })
    .unwrap();
    let prototype = metadata.libraries[0]
        .symbols
        .iter()
        .find(|symbol| symbol.name == "fixture_increment")
        .unwrap();
    let NativeType::Pointer { pointee, .. } = &prototype.parameters[0].ty else {
        panic!("fixture argument must be a pointer");
    };
    let pointer = NativePointerResource::new(
        pointee.as_ref().clone(),
        &Value::Int(IntValue::I32(40)),
        &metadata,
    )
    .unwrap();
    let library = LoadedLibrary::open(&library_path).unwrap();
    for expected in [41, 42] {
        invoke_symbol_with_bindings(
            &library,
            prototype,
            &[Value::Int(IntValue::I32(0))],
            &metadata,
            &[],
            &[PointerBinding::resource(0, &pointer)],
        )
        .unwrap();
        assert_eq!(
            pointer.value(&metadata).unwrap(),
            Value::Int(IntValue::I32(expected))
        );
    }
}

#[test]
fn c_abi_corpus_covers_scalars_arrays_structures_and_outputs() {
    if Command::new("clang").arg("--version").output().is_err() {
        eprintln!("skipping native fixture because clang is unavailable");
        return;
    }
    let temporary = tempfile::tempdir().expect("temporary directory");
    let Some(library_path) = compile_fixture(temporary.path()) else {
        eprintln!("skipping native fixture because the C compiler is unavailable");
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let metadata = prepare_header(&HeaderPreparation {
        header: root.join("tests/fixtures/interface.h"),
        library_name: "fixture".into(),
        library_path: library_path.to_string_lossy().into_owned(),
        target_triple: target_lexicon::HOST.to_string(),
        clang: "clang".into(),
        include_directories: Vec::new(),
        definitions: Vec::new(),
    })
    .expect("prepared metadata");
    let library = LoadedLibrary::open(&library_path).expect("loaded fixture");
    let prototype = |name: &str| {
        metadata.libraries[0]
            .symbols
            .iter()
            .find(|symbol| symbol.name == name)
            .expect("prototype")
    };

    let values =
        Tensor::new_integer(IntegerStorage::I32(vec![3, -2, 8]), vec![1, 3]).expect("typed values");
    let sum = invoke_symbol(
        &library,
        prototype("fixture_sum"),
        &[Value::Tensor(values), Value::Int(IntValue::U32(3))],
        &metadata,
    )
    .expect("array invocation");
    assert_eq!(
        sum.return_value,
        Some(InvocationValue::Value(Value::Int(IntValue::I32(9))))
    );

    let mut record = StructValue::new();
    record.insert("value", Value::Num(2.5));
    record.insert("tag", Value::Int(IntValue::U32(7)));
    let scaled = invoke_symbol(
        &library,
        prototype("fixture_scale"),
        &[Value::Struct(record), Value::Num(4.0)],
        &metadata,
    )
    .expect("structure pointer invocation");
    assert_eq!(
        scaled.return_value,
        Some(InvocationValue::Value(Value::Num(10.0)))
    );
    assert_eq!(
        field(&scaled.output_parameters[0].1, "value"),
        &Value::Num(10.0)
    );
    assert_eq!(
        field(&scaled.output_parameters[0].1, "tag"),
        &Value::Int(IntValue::U32(7))
    );

    let made = invoke_symbol(
        &library,
        prototype("fixture_make"),
        &[Value::Num(6.25), Value::Int(IntValue::U32(12))],
        &metadata,
    )
    .expect("structure return invocation");
    let Some(InvocationValue::Value(made)) = made.return_value else {
        panic!("expected structure return");
    };
    assert_eq!(field(&made, "value"), &Value::Num(6.25));
    assert_eq!(field(&made, "tag"), &Value::Int(IntValue::U32(12)));
}

#[test]
fn integer_conversion_rejects_fractional_and_out_of_range_inputs() {
    let temporary = tempfile::tempdir().expect("temporary directory");
    let Some(library_path) = compile_fixture(temporary.path()) else {
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let metadata = prepare_header(&HeaderPreparation {
        header: root.join("tests/fixtures/interface.h"),
        library_name: "fixture".into(),
        library_path: library_path.to_string_lossy().into_owned(),
        target_triple: target_lexicon::HOST.to_string(),
        clang: "clang".into(),
        include_directories: Vec::new(),
        definitions: Vec::new(),
    })
    .expect("prepared metadata");
    let library = LoadedLibrary::open(&library_path).expect("loaded fixture");
    let sum = metadata.libraries[0]
        .symbols
        .iter()
        .find(|symbol| symbol.name == "fixture_sum")
        .expect("prototype");
    let values =
        Tensor::new_integer(IntegerStorage::I32(vec![1]), vec![1, 1]).expect("typed values");
    let error = invoke_symbol(
        &library,
        sum,
        &[Value::Tensor(values), Value::Num(1.5)],
        &metadata,
    )
    .expect_err("fractional length must fail before native execution");
    assert!(error
        .to_string()
        .contains("not an exactly representable unsigned integer"));
}

#[test]
fn callbacks_reenter_through_an_explicit_dispatch_contract() {
    let temporary = tempfile::tempdir().expect("temporary directory");
    let Some(library_path) = compile_fixture(temporary.path()) else {
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let metadata = prepare_header(&HeaderPreparation {
        header: root.join("tests/fixtures/interface.h"),
        library_name: "fixture".into(),
        library_path: library_path.to_string_lossy().into_owned(),
        target_triple: target_lexicon::HOST.to_string(),
        clang: "clang".into(),
        include_directories: Vec::new(),
        definitions: Vec::new(),
    })
    .expect("prepared metadata");
    let library = LoadedLibrary::open(&library_path).expect("loaded fixture");
    let apply = metadata.libraries[0]
        .symbols
        .iter()
        .find(|symbol| symbol.name == "fixture_apply")
        .expect("prototype");
    let dispatch = |arguments: &[Value]| {
        let Value::Int(value) = &arguments[0] else {
            return Err("expected int32 callback argument".into());
        };
        Ok(Value::Int(IntValue::I32(
            value.try_to_i32().expect("int32 callback value") + 7,
        )))
    };
    let result = invoke_symbol_with_callbacks(
        &library,
        apply,
        &[
            Value::Int(IntValue::I32(5)),
            Value::FunctionHandle("callback".into()),
        ],
        &metadata,
        &[CallbackBinding {
            argument_index: 1,
            dispatch: &dispatch,
        }],
    )
    .expect("callback invocation");
    assert_eq!(
        result.return_value,
        Some(InvocationValue::Value(Value::Int(IntValue::I32(25))))
    );

    let failure = |_arguments: &[Value]| Err("synthetic callback failure".into());
    let error = invoke_symbol_with_callbacks(
        &library,
        apply,
        &[
            Value::Int(IntValue::I32(5)),
            Value::FunctionHandle("callback".into()),
        ],
        &metadata,
        &[CallbackBinding {
            argument_index: 1,
            dispatch: &failure,
        }],
    )
    .expect_err("callback failure must cross the boundary as an error");
    assert!(error.to_string().contains("synthetic callback failure"));
}

#[test]
fn borrowed_pointer_returns_preserve_typed_nullability() {
    let temporary = tempfile::tempdir().expect("temporary directory");
    let Some(library_path) = compile_fixture(temporary.path()) else {
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let metadata = prepare_header(&HeaderPreparation {
        header: root.join("tests/fixtures/interface.h"),
        library_name: "fixture".into(),
        library_path: library_path.to_string_lossy().into_owned(),
        target_triple: target_lexicon::HOST.to_string(),
        clang: "clang".into(),
        include_directories: Vec::new(),
        definitions: Vec::new(),
    })
    .expect("prepared metadata");
    let library = LoadedLibrary::open(&library_path).expect("loaded fixture");
    let prototype = metadata.libraries[0]
        .symbols
        .iter()
        .find(|symbol| symbol.name == "fixture_borrowed_value")
        .expect("pointer-returning prototype");

    for (present, expected_null) in [(0, true), (1, false)] {
        let result = invoke_symbol(
            &library,
            prototype,
            &[Value::Int(IntValue::I32(present))],
            &metadata,
        )
        .expect("pointer invocation");
        let Some(InvocationValue::Pointer(pointer)) = result.return_value else {
            panic!("expected a typed pointer return");
        };
        assert_eq!(pointer.is_null(), expected_null);
        assert_eq!(
            pointer.ownership,
            runmat_native_ffi::PointerOwnership::Borrowed
        );
    }
}
