use futures::executor::block_on;
use runmat_accelerate_api::{GpuHandleProvenance, GpuTensorHandle};
use runmat_builtins::Value;

const PACKET: [(&str, usize); 9] = [
    ("num2str", 2),
    ("onehotdecode", 4),
    ("onehotencode", 3),
    ("ordinal", 3),
    ("removeLongWords", 1),
    ("removeShortWords", 1),
    ("removeWords", 1),
    ("regexprep", 2),
    ("replaceBetween", 2),
];

fn resident_handle(buffer_id: u64, provenance: GpuHandleProvenance) -> GpuTensorHandle {
    let handle = GpuTensorHandle {
        shape: vec![1, 1],
        device_id: u32::MAX - 440,
        buffer_id,
        descriptor: Default::default(),
    };
    runmat_accelerate_api::set_handle_provenance(&handle, provenance);
    handle
}

#[test]
fn label_and_text_control_packet_has_class_complete_capability_metadata() {
    for (name, expected_forms) in PACKET {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(builtin.integer_capabilities.len(), expected_forms, "{name}");
        assert!(builtin.integer_audit.is_none(), "{name}");
        for capability in builtin.integer_capabilities {
            assert!(
                !capability.inputs.is_empty() || capability.form.contains("integer_typename"),
                "{name}: {}",
                capability.form
            );
            for input in capability.inputs {
                assert!(
                    input.classes.len() == 8
                        || input.availability
                            == runmat_builtins::BuiltinIntegerInputAvailability::Rejected,
                    "{name}: {} input {}",
                    capability.form,
                    input.name
                );
            }
        }
    }
}

#[test]
fn explicit_resident_fallbacks_reject_before_provider_access_in_strict_mode() {
    let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let calls = [
        (
            "num2str",
            vec![Value::GpuTensor(resident_handle(
                1,
                GpuHandleProvenance::Explicit,
            ))],
        ),
        (
            "onehotencode",
            vec![
                Value::GpuTensor(resident_handle(2, GpuHandleProvenance::Explicit)),
                Value::Num(1.0),
            ],
        ),
        (
            "onehotdecode",
            vec![
                Value::GpuTensor(resident_handle(3, GpuHandleProvenance::Explicit)),
                Value::Num(1.0),
                Value::Num(1.0),
            ],
        ),
        (
            "ordinal",
            vec![Value::GpuTensor(resident_handle(
                4,
                GpuHandleProvenance::Explicit,
            ))],
        ),
        (
            "removeShortWords",
            vec![
                Value::Num(0.0),
                Value::GpuTensor(resident_handle(5, GpuHandleProvenance::Explicit)),
            ],
        ),
        (
            "removeLongWords",
            vec![
                Value::Num(0.0),
                Value::GpuTensor(resident_handle(6, GpuHandleProvenance::Explicit)),
            ],
        ),
        (
            "removeWords",
            vec![
                Value::Num(0.0),
                Value::GpuTensor(resident_handle(7, GpuHandleProvenance::Explicit)),
            ],
        ),
        (
            "regexprep",
            vec![
                Value::from("a"),
                Value::from("a"),
                Value::from("b"),
                Value::GpuTensor(resident_handle(8, GpuHandleProvenance::Explicit)),
            ],
        ),
        (
            "replaceBetween",
            vec![
                Value::from("abc"),
                Value::GpuTensor(resident_handle(9, GpuHandleProvenance::Explicit)),
                Value::Num(2.0),
                Value::from("x"),
            ],
        ),
    ];
    for (name, args) in calls {
        let error = block_on(runmat_runtime::call_builtin_async(name, &args))
            .expect_err("explicit resident fallback must be gated");
        assert!(
            error
                .identifier()
                .is_some_and(|identifier| identifier.starts_with("RunMat:compatibility:")),
            "{name}: {error:?}"
        );
        assert!(
            !error.message().to_ascii_lowercase().contains("provider"),
            "{name}: {}",
            error.message()
        );
    }
    for buffer_id in 1..=9 {
        runmat_accelerate_api::clear_handle_metadata(&GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX - 440,
            buffer_id,
            descriptor: Default::default(),
        });
    }
}

#[test]
fn onehotdecode_broad_output_policy_precedes_automatic_provider_gather() {
    let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let handle = resident_handle(10, GpuHandleProvenance::Automatic);
    let error = block_on(runmat_runtime::call_builtin_async(
        "onehotdecode",
        &[
            Value::GpuTensor(handle.clone()),
            Value::Num(1.0),
            Value::Num(1.0),
            Value::from("logical"),
        ],
    ))
    .expect_err("RunMat-only output type must reject before gather");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:OnehotdecodeLogicalCellOutputExtension")
    );
    assert!(!error.message().to_ascii_lowercase().contains("provider"));
    runmat_accelerate_api::clear_handle_metadata(&handle);
}
