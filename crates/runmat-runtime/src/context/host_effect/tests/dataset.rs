use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sealed_dataset_handle_workload_distinguishes_role_key_root_and_output() {
    use crate::context::host_effect::{
        DatasetHandleKey, DatasetHandleOperation, DatasetHandleRole,
    };
    let first = std::path::Path::new("/one/same.data");
    let second = std::path::Path::new("/two/same.data");
    let key = DatasetHandleKey {
        host_identity: "runmat.data.dataset-authority",
        handle: 7,
        generation: 1,
    };
    fn workload<'a>(
        root: &'a std::path::Path,
        key: DatasetHandleKey<'a>,
        role: DatasetHandleRole<'a>,
        inputs: &'a [crate::context::host_effect::DatasetHandleInput],
        output: HostEffectOutput,
    ) -> HostEffectWorkload<'a> {
        HostEffectWorkload::DatasetHandle {
            operation: DatasetHandleOperation::ArrayRead,
            root,
            key,
            role,
            inputs,
            output,
        }
    }
    let selected = workload(
        first,
        key,
        DatasetHandleRole::Array("sample"),
        &[],
        HostEffectOutput::ReturnedValue,
    );
    assert_eq!(selected.output(), HostEffectOutput::ReturnedValue);
    for changed in [
        workload(
            second,
            key,
            DatasetHandleRole::Array("sample"),
            &[],
            HostEffectOutput::ReturnedValue,
        ),
        workload(
            first,
            DatasetHandleKey { handle: 8, ..key },
            DatasetHandleRole::Array("sample"),
            &[],
            HostEffectOutput::ReturnedValue,
        ),
        workload(
            first,
            key,
            DatasetHandleRole::Array("other"),
            &[],
            HostEffectOutput::ReturnedValue,
        ),
        workload(
            first,
            key,
            DatasetHandleRole::Array("sample"),
            &[],
            HostEffectOutput::EffectOnly,
        ),
    ] {
        assert_ne!(selected.identity_bytes(), changed.identity_bytes());
    }
    assert_ne!(
        selected.identity_bytes(),
        workload(
            first,
            key,
            DatasetHandleRole::Transaction("sample"),
            &[],
            HostEffectOutput::ReturnedValue,
        )
        .identity_bytes()
    );
    let incomplete = [crate::context::host_effect::DatasetHandleInput {
        fact: runmat_types::ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
        identity_digest: None,
    }];
    assert!(!workload(
        first,
        key,
        DatasetHandleRole::Array("sample"),
        &incomplete,
        HostEffectOutput::ReturnedValue,
    )
    .identity_complete());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn finite_handle_route_refuses_incomplete_input_identity_before_selection() {
    use crate::context::host_effect::{
        DatasetHandleInput, DatasetHandleKey, DatasetHandleOperation, DatasetHandleRole,
    };
    let placement = Rc::new(Placement::default());
    let runtime = runtime_context(
        placement.clone(),
        Some(bounds(
            ResourceByteBound::Known(1),
            ResourceByteBound::Known(1),
        )),
        Some(32),
    );
    let incomplete = [DatasetHandleInput {
        fact: runmat_types::ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
        identity_digest: None,
    }];
    let request = HostEffectRequest {
        operation: "DataArray.write",
        workload: HostEffectWorkload::DatasetHandle {
            operation: DatasetHandleOperation::ArrayWrite,
            root: Path::new("/fixture.data"),
            key: DatasetHandleKey {
                host_identity: "runmat.data.dataset-authority",
                handle: 1,
                generation: 1,
            },
            role: DatasetHandleRole::Array("A"),
            inputs: &incomplete,
            output: HostEffectOutput::ReturnedValue,
        },
        domain_facts: Digest::sha256(b"incomplete"),
        filesystem: fixture_fs(),
    };
    let error = select_host_effect(Some(&runtime), request)
        .err()
        .expect("class and shape alone cannot certify a finite route");
    assert_eq!(
        error.identifier.as_deref(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(placement.selections.get(), 0);
    assert!(placement.observations.borrow().is_empty());
}

#[test]
fn dataset_identity_carries_typed_operation_both_paths_and_output() {
    let source = Path::new("/source.data");
    let destination = Path::new("/destination.data");
    let selected = HostEffectWorkload::DatasetPath {
        operation: DatasetPathOperation::Copy,
        source,
        destination: Some(destination),
        output: HostEffectOutput::ReturnedValue,
    };
    assert_eq!(selected.output(), HostEffectOutput::ReturnedValue);
    for other in [
        HostEffectWorkload::DatasetPath {
            operation: DatasetPathOperation::Move,
            source,
            destination: Some(destination),
            output: HostEffectOutput::ReturnedValue,
        },
        HostEffectWorkload::DatasetPath {
            operation: DatasetPathOperation::Copy,
            source,
            destination: Some(Path::new("/other.data")),
            output: HostEffectOutput::ReturnedValue,
        },
        HostEffectWorkload::DatasetPath {
            operation: DatasetPathOperation::Copy,
            source,
            destination: Some(destination),
            output: HostEffectOutput::EffectOnly,
        },
    ] {
        assert_ne!(selected.identity_bytes(), other.identity_bytes());
    }
    assert_ne!(
        selected.identity_bytes(),
        HostEffectWorkload::ArchiveExtract {
            source: HostEffectSource::LocalArchive(source),
            destination: Some(destination),
            output: HostEffectOutput::ReturnedValue,
        }
        .identity_bytes()
    );
}

#[cfg(unix)]
#[test]
fn dataset_identity_keeps_non_utf8_path_bytes() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    use std::path::PathBuf;

    let path = |byte| {
        let mut raw = b"/dataset-".to_vec();
        raw.push(byte);
        PathBuf::from(OsString::from_vec(raw))
    };
    let first = path(0xff);
    let second = path(0xfe);
    assert_eq!(first.to_string_lossy(), second.to_string_lossy());
    fn workload(source: &Path) -> HostEffectWorkload<'_> {
        HostEffectWorkload::DatasetPath {
            operation: DatasetPathOperation::Open,
            source,
            destination: None,
            output: HostEffectOutput::ReturnedValue,
        }
    }
    assert_ne!(
        workload(&first).identity_bytes(),
        workload(&second).identity_bytes()
    );
}

#[test]
fn finite_dataset_workload_without_certified_producer_rejects() {
    let placement = Rc::new(Placement::default());
    let runtime = runtime_context(placement.clone(), None, Some(64));
    let workload = HostEffectWorkload::DatasetPath {
        operation: DatasetPathOperation::Create,
        source: Path::new("/created.data"),
        destination: None,
        output: HostEffectOutput::ReturnedValue,
    };
    let error = select_host_effect(
        Some(&runtime),
        HostEffectRequest {
            operation: "data.create",
            workload,
            domain_facts: Digest::sha256(b"typed-schema"),
            filesystem: fixture_fs(),
        },
    )
    .err()
    .expect("finite budget must reject unknown peak");
    assert_eq!(
        error.identifier.as_deref(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(placement.selections.get(), 0);
}
