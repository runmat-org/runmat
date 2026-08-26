#![cfg(any(target_os = "macos", target_os = "linux"))]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::rc::Rc;

use runmat_runtime::context::{ForeignCall, RuntimeContext, RuntimeServicePorts};
use runmat_runtime::execution::RuntimeExecutionService;
use runmat_runtime::foreign::{
    ForeignPlatform, ForeignRuntime, NativeFfiAdapter, NATIVE_FFI_ADAPTER_ID,
};
use runmat_value::{IntValue, Tensor, Value};

fn compile_fixture(directory: &Path) -> Option<PathBuf> {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/native_ffi/interface.c");
    let output = directory.join(if cfg!(target_os = "macos") {
        "libruntime_fixture.dylib"
    } else {
        "libruntime_fixture.so"
    });
    let mut command = Command::new("clang");
    if cfg!(target_os = "macos") {
        command.arg("-dynamiclib");
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
