use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, GpuHandleProvenance, GpuTensorHandle, GpuTensorStorage,
    HostNumericDataOwned, HostNumericTensorOwned,
};
use runmat_value::Value;

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

fn resident_handle(value: u32, provenance: GpuHandleProvenance) -> GpuTensorHandle {
    let allocation = TestAllocationOwner::new()
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![f64::from(value)]),
            shape: vec![1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert!(!runmat_accelerate_api::registered_providers()
        .iter()
        .any(|provider| provider.device_id() == allocation.handle.device_id()));
    allocation.handle.with_provenance(provenance).unwrap()
}

#[test]
fn label_and_text_control_packet_has_class_complete_capability_metadata() {
    for (name, expected_forms) in PACKET {
        let builtin =
            runmat_builtins::builtin_catalog_entry_by_name(name).expect("canonical builtin");
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
        let caller = args
            .iter()
            .find_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            })
            .unwrap();
        let descriptor = caller.wire_descriptor();
        let physical = caller.allocation_lease().downgrade();
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
        assert_eq!(caller.wire_descriptor(), descriptor);
        drop(args);
        assert!(physical.acquire().is_some());
        drop(caller);
        assert!(physical.acquire().is_none());
    }
}

#[test]
fn onehotdecode_broad_output_policy_precedes_automatic_provider_gather() {
    let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let handle = resident_handle(10, GpuHandleProvenance::Automatic);
    let physical = handle.allocation_lease().downgrade();
    let descriptor = handle.wire_descriptor();
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
    assert_eq!(handle.wire_descriptor(), descriptor);
    drop(handle);
    assert!(physical.acquire().is_none());
}
