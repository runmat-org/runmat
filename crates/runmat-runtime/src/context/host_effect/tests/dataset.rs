use super::*;

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
