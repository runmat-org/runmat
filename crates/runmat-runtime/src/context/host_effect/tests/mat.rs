use super::*;

fn workload<'a>(
    source: &'a Path,
    names: &'a [String],
    patterns: &'a [String],
    output: MatDecodeOutput,
) -> HostEffectWorkload<'a> {
    HostEffectWorkload::MatDecode {
        source,
        selection: MatDecodeSelection::Selected { names, patterns },
        output,
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mat_identity_separates_exact_path_selectors_and_publication_mode() {
    let names = vec!["A".to_owned()];
    let other = vec!["B".to_owned()];
    let patterns = vec!["^sensor_".to_owned()];
    let first = workload(
        Path::new("/data/one.mat"),
        &names,
        &patterns,
        MatDecodeOutput::Struct,
    )
    .identity_bytes();
    assert_ne!(
        first,
        workload(
            Path::new("/data/two.mat"),
            &names,
            &patterns,
            MatDecodeOutput::Struct
        )
        .identity_bytes()
    );
    assert_ne!(
        first,
        workload(
            Path::new("/data/one.mat"),
            &other,
            &patterns,
            MatDecodeOutput::Struct
        )
        .identity_bytes()
    );
    assert_ne!(
        first,
        workload(
            Path::new("/data/one.mat"),
            &names,
            &patterns,
            MatDecodeOutput::Workspace
        )
        .identity_bytes()
    );
    assert_eq!(
        HostEffectWorkload::MatDecode {
            source: Path::new("/data/one.mat"),
            selection: MatDecodeSelection::All,
            output: MatDecodeOutput::Workspace,
        }
        .output(),
        HostEffectOutput::EffectOnly
    );
}

#[cfg(unix)]
#[test]
fn mat_identity_preserves_non_utf8_path_bytes() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    use std::path::PathBuf;
    let source = |suffix| {
        let mut raw = b"/mat-".to_vec();
        raw.push(suffix);
        PathBuf::from(OsString::from_vec(raw))
    };
    let first = source(0xff);
    let second = source(0xfe);
    assert_eq!(first.to_string_lossy(), second.to_string_lossy());
    let identity = |path: &Path| {
        HostEffectWorkload::MatDecode {
            source: path,
            selection: MatDecodeSelection::All,
            output: MatDecodeOutput::Struct,
        }
        .identity_bytes()
    };
    assert_ne!(identity(&first), identity(&second));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mat_encode_identity_separates_destination_controls_and_output() {
    let names = vec!["A".to_owned()];
    let other = vec!["B".to_owned()];
    let structs = [MatStructSelection {
        source: "record",
        fields: None,
    }];
    let patterns = vec!["^sensor_".to_owned()];
    let workload = |destination: &Path, names: &[String], output| {
        HostEffectWorkload::MatEncode {
            destination,
            mode: MatEncodeMode::Save {
                append: true,
                names,
                structs: &structs,
                patterns: &patterns,
            },
            output,
        }
        .identity_bytes()
    };
    let first = workload(Path::new("/data/one.mat"), &names, MatEncodeOutput::Status);
    assert_ne!(
        first,
        workload(Path::new("/data/two.mat"), &names, MatEncodeOutput::Status)
    );
    assert_ne!(
        first,
        workload(Path::new("/data/one.mat"), &other, MatEncodeOutput::Status)
    );
    assert_ne!(
        first,
        workload(
            Path::new("/data/one.mat"),
            &names,
            MatEncodeOutput::EffectOnly
        )
    );
    assert_eq!(
        HostEffectWorkload::MatEncode {
            destination: Path::new("/data/one.mat"),
            mode: MatEncodeMode::AssignMember {
                member: "x",
                suffix: MatAssignmentSuffix::Exact(Digest::sha256(b"whole member")),
            },
            output: MatEncodeOutput::UpdatedObject,
        }
        .output(),
        HostEffectOutput::ReturnedValue
    );
}

#[test]
fn finite_mat_decode_without_a_resource_producer_rejects_before_selection() {
    let placement = Rc::new(Placement::default());
    let runtime = runtime_context(placement.clone(), None, Some(64));
    let workload = HostEffectWorkload::MatDecode {
        source: Path::new("/fixture.mat"),
        selection: MatDecodeSelection::All,
        output: MatDecodeOutput::Workspace,
    };
    let error = select_host_effect(
        Some(&runtime),
        HostEffectRequest {
            operation: "load",
            workload,
            domain_facts: Digest::sha256(workload.identity_bytes()),
            filesystem: fixture_fs(),
        },
    )
    .err()
    .expect("unknown MAT codec peak rejects finite quota");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(placement.selections.get(), 0);
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mat_handle_and_member_suffix_identity_are_exact_or_explicitly_unavailable() {
    let source = Path::new("/fixture.mat");
    let readonly = [MatHandleControl::KnownWritable(false)];
    let writable = [MatHandleControl::KnownWritable(true)];
    let handle = |controls: &[MatHandleControl]| {
        HostEffectWorkload::MatHandle { source, controls }.identity_bytes()
    };
    assert_ne!(handle(&readonly), handle(&writable));

    let exact = MatAssignmentSuffix::Exact(Digest::sha256(b"index 1"));
    let other = MatAssignmentSuffix::Exact(Digest::sha256(b"index 2"));
    let assignment = |suffix| HostEffectWorkload::MatEncode {
        destination: source,
        mode: MatEncodeMode::AssignMember {
            member: "A",
            suffix,
        },
        output: MatEncodeOutput::UpdatedObject,
    };
    assert_ne!(
        assignment(exact).identity_bytes(),
        assignment(other).identity_bytes()
    );
    assert!(assignment(exact).identity_complete());
    assert!(!assignment(MatAssignmentSuffix::Unavailable).identity_complete());
    assert!(!HostEffectWorkload::MatDecode {
        source,
        selection: MatDecodeSelection::MemberPath {
            member: "A",
            suffix: MatAssignmentSuffix::Unavailable,
        },
        output: MatDecodeOutput::SingleMember,
    }
    .identity_complete());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn finite_mat_member_with_unavailable_suffix_rejects_before_selection() {
    let placement = Rc::new(Placement::default());
    let runtime = runtime_context(placement.clone(), None, Some(64));
    let workload = HostEffectWorkload::MatEncode {
        destination: Path::new("/fixture.mat"),
        mode: MatEncodeMode::AssignMember {
            member: "A",
            suffix: MatAssignmentSuffix::Unavailable,
        },
        output: MatEncodeOutput::UpdatedObject,
    };
    let error = select_host_effect(
        Some(&runtime),
        HostEffectRequest {
            operation: "matlab.io.MatFile.subsasgn",
            workload,
            domain_facts: Digest::sha256(workload.identity_bytes()),
            filesystem: fixture_fs(),
        },
    )
    .err()
    .expect("unavailable selector identity rejects finite route");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(placement.selections.get(), 0);
}
