#![cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::rc::Rc;

use runmat_native_ffi::NativeInterfaceArtifactManifest;
use runmat_runtime::context::{ForeignCall, RuntimeContext, RuntimeServicePorts};
use runmat_runtime::execution::RuntimeExecutionService;
use runmat_runtime::foreign::{
    ForeignPlatform, ForeignRuntime, NativeFfiAdapter, NATIVE_FFI_ADAPTER_ID,
};
use runmat_types::ForeignOwnership;
use runmat_value::{IntValue, Tensor, Value};

fn compile_fixture(directory: &Path) -> Option<PathBuf> {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/native_ffi/interface.c");
    let output = directory.join(if cfg!(target_os = "macos") {
        "libruntime_fixture.dylib"
    } else if cfg!(target_os = "windows") {
        "runtime_fixture.dll"
    } else {
        "libruntime_fixture.so"
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

fn invoke(
    context: &RuntimeContext,
    symbol: &str,
    arguments: Vec<Value>,
    requested_outputs: usize,
) -> Result<Value, Box<runmat_runtime::RuntimeError>> {
    let service = context
        .service_ports()
        .require_foreign(symbol)
        .unwrap()
        .clone();
    futures::executor::block_on(context.scope(service.invoke(
        context.clone(),
        ForeignCall {
            adapter: NATIVE_FFI_ADAPTER_ID.into(),
            symbol: symbol.into(),
            arguments,
            requested_outputs,
        },
    )))
    .map_err(Box::new)
}

#[test]
fn library_and_pointer_state_are_owned_by_one_runtime_session() {
    if Command::new("clang").arg("--version").output().is_err() {
        eprintln!("skipping native FFI session test because clang is unavailable");
        return;
    }
    let temporary = tempfile::tempdir().unwrap();
    let Some(library_path) = compile_fixture(temporary.path()) else {
        eprintln!("skipping native FFI session test because the C compiler is unavailable");
        return;
    };
    let header_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/native_ffi/interface.h");

    let foreign = ForeignRuntime::new(ForeignPlatform::Native);
    let adapter = NativeFfiAdapter::new(foreign.handles().clone()).unwrap();
    foreign.register_adapter(adapter).unwrap();
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_foreign(Rc::new(foreign)));

    let alias = invoke(
        &context,
        "load",
        vec![
            Value::String(library_path.display().to_string()),
            Value::String(header_path.display().to_string()),
            Value::String("session_fixture".into()),
        ],
        1,
    )
    .unwrap();
    assert_eq!(alias, Value::String("session_fixture".into()));
    let signatures = invoke(
        &context,
        "functions",
        vec![
            Value::String("session_fixture".into()),
            Value::String("-full".into()),
        ],
        1,
    )
    .unwrap();
    let Value::Cell(signatures) = signatures else {
        panic!("full function metadata must be returned as a cell array");
    };
    assert!(signatures
        .data
        .iter()
        .any(|value| { value == &Value::String("int32 fixture_add(int32, int32)".into()) }));

    let census_header = temporary.path().join("census.h");
    std::fs::write(
        &census_header,
        "#include <stdint.h>\nint32_t fixture_add(int32_t left, int32_t right);\nint32_t fixture_absent(int32_t value);\n",
    )
    .unwrap();
    let report = invoke(
        &context,
        "load_report",
        vec![
            Value::String(library_path.display().to_string()),
            Value::String(census_header.display().to_string()),
            Value::String("census_fixture".into()),
        ],
        2,
    )
    .expect("legacy load report");
    let Value::OutputList(report) = report else {
        panic!("legacy load report must contain two outputs");
    };
    let Value::Cell(notfound) = &report[0] else {
        panic!("missing functions must be reported as a cell array");
    };
    assert_eq!(notfound.data, vec![Value::String("fixture_absent".into())]);
    assert!(matches!(&report[1], Value::String(_)));
    invoke(
        &context,
        "unload",
        vec![Value::String("census_fixture".into())],
        1,
    )
    .unwrap();

    let prepared_alias = invoke(
        &context,
        "build_interface",
        vec![
            Value::String(header_path.display().to_string()),
            Value::String("Libraries".into()),
            Value::String(library_path.display().to_string()),
            Value::String("InterfaceName".into()),
            Value::String("prepared_fixture".into()),
        ],
        1,
    )
    .expect("prepared interface");
    assert_eq!(prepared_alias, Value::String("prepared_fixture".into()));
    let manifest_path = NativeInterfaceArtifactManifest::path_for_library(&library_path);
    let manifest = NativeInterfaceArtifactManifest::read(&manifest_path).expect("manifest");
    manifest
        .validate_current_library(&std::fs::read(&library_path).unwrap())
        .expect("exact prepared library");
    invoke(
        &context,
        "unload",
        vec![Value::String("prepared_fixture".into())],
        1,
    )
    .unwrap();
    assert_eq!(
        invoke(
            &context,
            "load_prepared",
            vec![
                Value::String(library_path.display().to_string()),
                Value::String(manifest_path.display().to_string()),
                Value::String("restored_fixture".into()),
            ],
            1,
        )
        .expect("load prepared interface"),
        Value::String("restored_fixture".into())
    );
    assert_eq!(
        invoke(
            &context,
            "call",
            vec![
                Value::String("restored_fixture".into()),
                Value::String("fixture_add".into()),
                Value::Int(IntValue::I32(20)),
                Value::Int(IntValue::I32(22)),
            ],
            1,
        )
        .unwrap(),
        Value::Int(IntValue::I32(42))
    );
    assert_eq!(
        invoke(
            &context,
            "call",
            vec![
                Value::String("session_fixture".into()),
                Value::String("fixture_add".into()),
                Value::Int(IntValue::I32(19)),
                Value::Int(IntValue::I32(23)),
            ],
            1,
        )
        .unwrap(),
        Value::Int(IntValue::I32(42))
    );

    for present in [0, 1] {
        let returned = invoke(
            &context,
            "call",
            vec![
                Value::String("session_fixture".into()),
                Value::String("fixture_borrowed_value".into()),
                Value::Int(IntValue::I32(present)),
            ],
            1,
        )
        .expect("nullable pointer return");
        let Value::Foreign(reference) = returned else {
            panic!("pointer returns must remain typed foreign resources");
        };
        assert_eq!(reference.ownership, ForeignOwnership::Borrowed);
        if present == 1 {
            let pointer = Value::Foreign(reference);
            let retyped = invoke(
                &context,
                "invoke_member",
                vec![
                    pointer.clone(),
                    Value::String("setdatatype".into()),
                    Value::String("int32Ptr".into()),
                    Value::Num(1.0),
                    Value::Num(1.0),
                ],
                1,
            )
            .expect("opaque pointer type declaration");
            let Value::Foreign(retyped_reference) = &retyped else {
                panic!("setdatatype must preserve the foreign reference");
            };
            let Value::Foreign(original_reference) = &pointer else {
                unreachable!("pointer is a foreign reference")
            };
            assert!(retyped_reference.is_same_resource(original_reference));
            let copied = invoke(
                &context,
                "get_member",
                vec![pointer, Value::String("Value".into())],
                1,
            )
            .expect("typed opaque pointer copy");
            let Value::Tensor(copied) = copied else {
                panic!("typed pointer copy must retain integer tensor storage");
            };
            assert_eq!(copied.shape, vec![1, 1]);
            assert_eq!(copied.numeric_value_at(0).unwrap().materialize_f64(), 17.0);
        }
    }

    let pointer = invoke(
        &context,
        "pointer",
        vec![
            Value::String("int32Ptr".into()),
            Value::Int(IntValue::I32(40)),
        ],
        1,
    )
    .unwrap();
    let updated = invoke(
        &context,
        "call",
        vec![
            Value::String("session_fixture".into()),
            Value::String("fixture_increment".into()),
            pointer.clone(),
        ],
        1,
    )
    .unwrap();
    let Value::Foreign(updated_reference) = &updated else {
        panic!("mutable pointer output must retain foreign identity");
    };
    let Value::Foreign(original_reference) = &pointer else {
        panic!("pointer constructor must return a foreign reference");
    };
    assert!(updated_reference.is_same_resource(original_reference));
    assert_eq!(
        invoke(&context, "pointer_value", vec![pointer.clone()], 1).unwrap(),
        Value::Int(IntValue::I32(41))
    );
    let assigned = invoke(
        &context,
        "set_member",
        vec![
            pointer.clone(),
            Value::String("Value".into()),
            Value::Int(IntValue::I32(50)),
        ],
        1,
    )
    .unwrap();
    let Value::Foreign(assigned_reference) = &assigned else {
        panic!("member assignment must retain foreign identity");
    };
    assert!(assigned_reference.is_same_resource(original_reference));
    assert_eq!(
        invoke(
            &context,
            "get_member",
            vec![pointer.clone(), Value::String("Value".into())],
            1,
        )
        .unwrap(),
        Value::Int(IntValue::I32(50))
    );
    assert_eq!(
        invoke(
            &context,
            "get_member",
            vec![pointer.clone(), Value::String("DataType".into())],
            1,
        )
        .unwrap(),
        Value::String("int32Ptr".into())
    );
    let shape_error = invoke(
        &context,
        "set_member",
        vec![
            pointer,
            Value::String("Value".into()),
            Value::Tensor(Tensor::new(vec![50.0, 51.0], vec![1, 2]).unwrap()),
        ],
        1,
    )
    .unwrap_err();
    assert_eq!(shape_error.identifier(), Some("RunMat:Foreign:InvalidCall"));

    let null_pointer = invoke(&context, "pointer", Vec::new(), 1).unwrap();
    assert!(matches!(null_pointer, Value::Foreign(_)));
    assert_eq!(
        invoke(
            &context,
            "get_member",
            vec![null_pointer, Value::String("DataType".into())],
            1,
        )
        .unwrap(),
        Value::String("voidPtr".into())
    );

    let typed_null = invoke(
        &context,
        "pointer",
        vec![Value::String("int32Ptr".into())],
        1,
    )
    .unwrap();
    assert_eq!(
        invoke(
            &context,
            "get_member",
            vec![typed_null.clone(), Value::String("Value".into())],
            1,
        )
        .unwrap(),
        Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap())
    );
    invoke(
        &context,
        "set_member",
        vec![
            typed_null.clone(),
            Value::String("Value".into()),
            Value::Int(IntValue::I32(7)),
        ],
        1,
    )
    .unwrap();
    assert_eq!(
        invoke(
            &context,
            "get_member",
            vec![typed_null, Value::String("Value".into())],
            1,
        )
        .unwrap(),
        Value::Int(IntValue::I32(7))
    );

    let record = invoke(
        &context,
        "structure",
        vec![Value::String("fixture_record".into())],
        1,
    )
    .unwrap();
    for (field, value) in [("left", 19), ("right", 23)] {
        invoke(
            &context,
            "set_member",
            vec![
                record.clone(),
                Value::String(field.into()),
                Value::Int(IntValue::I32(value)),
            ],
            1,
        )
        .unwrap();
    }
    assert_eq!(
        invoke(
            &context,
            "call",
            vec![
                Value::String("session_fixture".into()),
                Value::String("fixture_record_sum".into()),
                record,
            ],
            1,
        )
        .unwrap(),
        Value::Int(IntValue::I32(42))
    );

    invoke(
        &context,
        "unload",
        vec![Value::String("session_fixture".into())],
        1,
    )
    .unwrap();
    invoke(
        &context,
        "unload",
        vec![Value::String("restored_fixture".into())],
        1,
    )
    .unwrap();
    assert_eq!(
        invoke(
            &context,
            "is_loaded",
            vec![Value::String("session_fixture".into())],
            1,
        )
        .unwrap(),
        Value::Bool(false)
    );
}
