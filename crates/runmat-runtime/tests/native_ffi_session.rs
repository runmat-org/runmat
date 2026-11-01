#![cfg(any(
    target_arch = "wasm32",
    target_os = "macos",
    target_os = "linux",
    target_os = "windows"
))]

#[cfg(target_arch = "wasm32")]
fn main() {
    // Cargo requires an entry point for this harness=false target on wasm.
    // This does not execute or qualify the native ExtensionHost scenario.
    panic!("native FFI session conformance requires a native host");
}

#[cfg(not(target_arch = "wasm32"))]
use std::ffi::OsStr;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::process::Command;
#[cfg(not(target_arch = "wasm32"))]
use std::rc::Rc;

#[cfg(not(target_arch = "wasm32"))]
use runmat_native_ffi::NativeInterfaceArtifactManifest;
#[cfg(not(target_arch = "wasm32"))]
use runmat_runtime::context::{ForeignCall, RuntimeContext, RuntimeServicePorts};
#[cfg(not(target_arch = "wasm32"))]
use runmat_runtime::execution::RuntimeExecutionService;
#[cfg(not(target_arch = "wasm32"))]
use runmat_runtime::foreign::{
    ForeignPlatform, ForeignRuntime, NativeFfiAdapter, NativeFfiIsolationPolicy,
    NATIVE_FFI_ADAPTER_ID,
};
#[cfg(not(target_arch = "wasm32"))]
use runmat_types::ForeignOwnership;
#[cfg(not(target_arch = "wasm32"))]
use runmat_value::{IntValue, Tensor, Value};

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    if std::env::args_os().nth(1).as_deref()
        == Some(OsStr::new(
            runmat_process_host::HiddenMode::ExtensionHost.marker(),
        ))
    {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("native extension host runtime")
            .block_on(runmat_runtime::foreign::run_extension_host())
            .expect("native extension host");
        return;
    }
    library_and_pointer_state_are_owned_by_one_runtime_session();
}

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
fn invoke(
    context: &RuntimeContext,
    symbol: &str,
    arguments: Vec<Value>,
    requested_outputs: usize,
) -> Result<runmat_runtime::execution::RootedValue, Box<runmat_runtime::RuntimeError>> {
    let outputs = invoke_outputs(context, symbol, arguments, requested_outputs)?;
    if outputs.len() != 1 {
        return Err(Box::new(runmat_runtime::RuntimeError::new(format!(
            "test helper expected one output, received {}",
            outputs.len()
        ))));
    }
    outputs.with_owned_output_storage(|values| {
        let value = values
            .try_into_first()
            .map_err(|_| {
                Box::new(runmat_runtime::sequence::sequence_error_to_runtime(
                    runmat_value::ValueSequenceError::SharedOutputStorage,
                ))
            })?
            .expect("one output");
        runmat_runtime::execution::RootedValue::new(value)
            .map_err(|error| Box::new(error.into_runtime_error()))
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn invoke_outputs(
    context: &RuntimeContext,
    symbol: &str,
    arguments: Vec<Value>,
    requested_outputs: usize,
) -> Result<runmat_runtime::execution::RootedValues, Box<runmat_runtime::RuntimeError>> {
    let service = context
        .service_ports()
        .require_foreign(symbol)
        .unwrap()
        .clone();
    futures::executor::block_on(
        context.scope(
            service.invoke(
                context.clone(),
                ForeignCall {
                    adapter: NATIVE_FFI_ADAPTER_ID.into(),
                    symbol: symbol.into(),
                    arguments: runmat_runtime::call::RootedFunctionArguments::ordinary(
                        runmat_runtime::execution::RootedValues::in_context(arguments, context)
                            .map_err(Box::new)?,
                    ),
                    requested_outputs,
                },
            ),
        ),
    )
    .into_invoked_result()
    .map_err(Box::new)?
    .with_owned_sequence(|sequence| {
        runmat_runtime::execution::RootedValues::from_output_storage(sequence.into_output_storage())
            .map_err(|error| Box::new(error.into_runtime_error()))
    })
}

#[cfg(not(target_arch = "wasm32"))]
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
            Value::String(library_path.display().to_string().into()),
            Value::String(header_path.display().to_string().into()),
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
    let Value::Cell(signatures) = signatures.value() else {
        panic!("full function metadata must be returned as a cell array");
    };
    assert!(signatures
        .data()
        .iter()
        .any(|value| { value == &Value::String("int32 fixture_add(int32, int32)".into()) }));

    let census_header = temporary.path().join("census.h");
    std::fs::write(
        &census_header,
        "#include <stdint.h>\nint32_t fixture_add(int32_t left, int32_t right);\nint32_t fixture_absent(int32_t value);\n",
    )
    .unwrap();
    let report = invoke_outputs(
        &context,
        "load_report",
        vec![
            Value::String(library_path.display().to_string().into()),
            Value::String(census_header.display().to_string().into()),
            Value::String("census_fixture".into()),
        ],
        2,
    )
    .expect("legacy load report");
    assert_eq!(report.len(), 2);
    let Value::Cell(notfound) = &report[0] else {
        panic!("missing functions must be reported as a cell array");
    };
    assert_eq!(
        notfound.data(),
        vec![Value::String("fixture_absent".into())]
    );
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
            Value::String(header_path.display().to_string().into()),
            Value::String("Libraries".into()),
            Value::String(library_path.display().to_string().into()),
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
    let missing_library = temporary.path().join("missing-prepared-library.so");
    let missing = invoke_outputs(
        &context,
        "load_prepared",
        vec![
            Value::String(missing_library.display().to_string().into()),
            Value::String(manifest_path.display().to_string().into()),
            Value::String("missing_fixture".into()),
        ],
        1,
    )
    .expect_err("missing prepared library must retain the file I/O cause");
    assert_eq!(missing.identifier(), Some("RunMat:Foreign:LoadFailed"));
    let source = std::error::Error::source(missing.as_ref()).expect("path-qualified cause");
    assert!(source.to_string().contains("missing-prepared-library.so"));
    assert!(source
        .source()
        .and_then(|nested| nested.downcast_ref::<std::io::Error>())
        .is_some());
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
                Value::String(library_path.display().to_string().into()),
                Value::String(manifest_path.display().to_string().into()),
                Value::String("restored_fixture".into()),
            ],
            1,
        )
        .expect("load prepared interface"),
        Value::String("restored_fixture".into())
    );
    let add_arguments = vec![
        Value::String("restored_fixture".into()),
        Value::String("fixture_add".into()),
        Value::Int(IntValue::I32(20)),
        Value::Int(IntValue::I32(22)),
    ];
    let error = invoke_outputs(&context, "call", add_arguments.clone(), 2)
        .expect_err("signature declares only one public output");
    assert_eq!(error.identifier(), Some("RunMat:Foreign:InvalidCall"));
    assert_eq!(
        invoke(
            &context,
            "call",
            vec![
                Value::String("restored_fixture".into()),
                Value::String("fixture_invocation_count".into()),
            ],
            1,
        )
        .unwrap(),
        Value::Int(IntValue::I32(0)),
        "impossible output request must not invoke the C function"
    );
    assert_eq!(
        invoke(&context, "call", add_arguments, 1).unwrap(),
        Value::Int(IntValue::I32(42))
    );
    assert_eq!(
        invoke(
            &context,
            "call",
            vec![
                Value::String("restored_fixture".into()),
                Value::String("fixture_invocation_count".into()),
            ],
            1,
        )
        .unwrap(),
        Value::Int(IntValue::I32(1))
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
        let Value::Foreign(reference) = returned.value() else {
            panic!("pointer returns must remain typed foreign resources");
        };
        assert_eq!(reference.ownership, ForeignOwnership::Borrowed);
        if present == 1 {
            let pointer = Value::Foreign(reference.clone());
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
            let Value::Foreign(retyped_reference) = retyped.value() else {
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
            let Value::Tensor(copied) = copied.value() else {
                panic!("typed pointer copy must retain integer tensor storage");
            };
            assert_eq!(copied.shape(), &[1, 1]);
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
    assert_eq!(
        invoke(
            &context,
            "get_member",
            vec![pointer.value().clone(), Value::String("dAtAtYpE".into())],
            1,
        )
        .unwrap(),
        Value::String("int32Ptr".into())
    );
    assert_eq!(
        invoke(
            &context,
            "invoke_member",
            vec![pointer.value().clone(), Value::String("IsNuLl".into())],
            1,
        )
        .unwrap(),
        Value::Bool(false)
    );
    let updated = invoke(
        &context,
        "call",
        vec![
            Value::String("session_fixture".into()),
            Value::String("fixture_increment".into()),
            pointer.value().clone(),
        ],
        1,
    )
    .unwrap();
    let Value::Foreign(updated_reference) = updated.value() else {
        panic!("mutable pointer output must retain foreign identity");
    };
    let Value::Foreign(original_reference) = pointer.value() else {
        panic!("pointer constructor must return a foreign reference");
    };
    assert!(updated_reference.is_same_resource(original_reference));
    assert_eq!(
        invoke(&context, "pointer_value", vec![pointer.value().clone()], 1).unwrap(),
        Value::Int(IntValue::I32(41))
    );
    let assigned = invoke(
        &context,
        "set_member",
        vec![
            pointer.value().clone(),
            Value::String("Value".into()),
            Value::Int(IntValue::I32(50)),
        ],
        1,
    )
    .unwrap();
    let Value::Foreign(assigned_reference) = assigned.value() else {
        panic!("member assignment must retain foreign identity");
    };
    assert!(assigned_reference.is_same_resource(original_reference));
    assert_eq!(
        invoke(
            &context,
            "get_member",
            vec![pointer.value().clone(), Value::String("Value".into())],
            1,
        )
        .unwrap(),
        Value::Int(IntValue::I32(50))
    );
    assert_eq!(
        invoke(
            &context,
            "get_member",
            vec![pointer.value().clone(), Value::String("DataType".into())],
            1,
        )
        .unwrap(),
        Value::String("int32Ptr".into())
    );
    let shape_error = invoke(
        &context,
        "set_member",
        vec![
            pointer.value().clone(),
            Value::String("Value".into()),
            Value::Tensor(Tensor::new(vec![50.0, 51.0], vec![1, 2]).unwrap()),
        ],
        1,
    )
    .unwrap_err();
    assert_eq!(shape_error.identifier(), Some("RunMat:Foreign:InvalidCall"));

    let null_pointer = invoke(&context, "pointer", Vec::new(), 1).unwrap();
    assert!(matches!(null_pointer.value(), Value::Foreign(_)));
    assert_eq!(
        invoke(
            &context,
            "get_member",
            vec![
                null_pointer.value().clone(),
                Value::String("DataType".into())
            ],
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
            vec![typed_null.value().clone(), Value::String("Value".into())],
            1,
        )
        .unwrap(),
        Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap())
    );
    invoke(
        &context,
        "set_member",
        vec![
            typed_null.value().clone(),
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
            vec![typed_null.value().clone(), Value::String("Value".into())],
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
                record.value().clone(),
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
                record.value().clone(),
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
    drop(context);

    isolated_direct_capture_retirement_replays(&library_path, &manifest_path, temporary.path());
}

#[cfg(not(target_arch = "wasm32"))]
fn isolated_direct_capture_retirement_replays(
    library_path: &Path,
    manifest_path: &Path,
    scratch: &Path,
) {
    // This same executable is the authenticated hidden-mode child. The
    // public prepared install deliberately has no attached private root;
    // it must capture one before the original paths disappear.
    let isolated = ForeignRuntime::new(ForeignPlatform::Native);
    let isolated_adapter = NativeFfiAdapter::new(isolated.handles().clone()).unwrap();
    isolated_adapter
        .set_isolation_policy(NativeFfiIsolationPolicy::isolated(None))
        .unwrap();
    isolated_adapter
        .install_prepared_artifact(library_path, manifest_path)
        .expect("direct isolated install captures private bytes");
    isolated.register_adapter(isolated_adapter.clone()).unwrap();
    let isolated_context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_foreign(Rc::new(isolated)));
    let append_library = scratch.join("append-library.bin");
    std::fs::copy(library_path, &append_library).unwrap();
    std::fs::remove_file(library_path).unwrap();
    std::fs::remove_file(manifest_path).unwrap();
    assert!(isolated_adapter
        .install_prepared_artifact(&append_library, &scratch.join("missing-manifest.json"))
        .is_err());

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("isolated native fixture runtime");
    runtime.block_on(async {
        assert_eq!(
            isolated_add(&isolated_context).await,
            Value::Int(IntValue::I32(42))
        );
        let staged = isolated_adapter
            .prepare_project_replacement(std::iter::empty())
            .expect("stage exact old native owner");
        let verified = staged
            .retire_owner(Some(&isolated_context))
            .await
            .expect("retire exact child");
        drop(verified);
        drop(staged);
        assert_eq!(
            isolated_add(&isolated_context).await,
            Value::Int(IntValue::I32(42))
        );
        isolated_adapter
            .shutdown()
            .await
            .expect("retire replayed child");
    });
}

#[cfg(not(target_arch = "wasm32"))]
async fn isolated_add(context: &RuntimeContext) -> Value {
    let service = context
        .service_ports()
        .require_foreign("call")
        .expect("native service")
        .clone();
    context
        .scope(
            service.invoke(
                context.clone(),
                ForeignCall {
                    adapter: NATIVE_FFI_ADAPTER_ID.into(),
                    symbol: "call".into(),
                    arguments: runmat_runtime::call::RootedFunctionArguments::ordinary(
                        runmat_runtime::execution::RootedValues::in_context(
                            vec![
                                Value::String("prepared_fixture".into()),
                                Value::String("fixture_add".into()),
                                Value::Int(IntValue::I32(20)),
                                Value::Int(IntValue::I32(22)),
                            ],
                            context,
                        )
                        .expect("root isolated native fixture arguments"),
                    ),
                    requested_outputs: 1,
                },
            ),
        )
        .await
        .into_invoked_result()
        .expect("isolated native fixture call")
        .with_owned_sequence(|sequence| {
            sequence
                .into_output_storage()
                .try_into_first()
                .expect("unique native output")
                .expect("one native output")
        })
}
