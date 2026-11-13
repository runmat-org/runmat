use super::*;

#[test]
fn hardware_profile_does_not_capture_local_account_selection() {
    let mut original = NativeExecutionConfig::for_current_executable().unwrap();
    original.local_fea_organization =
        runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization::Selected(
            uuid::Uuid::from_u128(1),
        );
    let encoded = serde_json::to_vec(&original).unwrap();
    let restored: NativeExecutionConfig = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(restored.local_fea_organization, Default::default());
    let mut other = original.clone();
    other.local_fea_organization =
        runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization::Selected(
            uuid::Uuid::from_u128(2),
        );
    assert_eq!(serde_json::to_vec(&other).unwrap(), encoded);
}

#[test]
fn default_profile_reuses_original_authority_and_converges_on_racing_publication() {
    let original = NativeExecutionConfig::for_current_executable().unwrap();
    let executable = original.executable.clone();
    let temp = tempfile::tempdir().unwrap();
    let paths = SupervisorPaths::new(temp.path().join("authority")).unwrap();
    let first =
        resolve_default_profile(paths.clone(), &executable, || Ok(original.clone())).unwrap();
    let bytes = std::fs::read(paths.root.join("supervisor-profile.json")).unwrap();
    let cold = resolve_default_profile(paths.clone(), &executable, || {
        panic!("existing authority must not resample memory or device epochs")
    })
    .unwrap();
    assert_eq!(first.execution, cold.execution);
    assert_eq!(first.identity().unwrap(), cold.identity().unwrap());
    assert_eq!(
        std::fs::read(paths.root.join("supervisor-profile.json")).unwrap(),
        bytes
    );
    // Exercise the journal boundary too, without launching any process.
    drop(crate::supervisor::LocalSupervisor::open(first).unwrap());
    drop(crate::supervisor::LocalSupervisor::open(cold).unwrap());

    let race = tempfile::tempdir().unwrap();
    let paths = SupervisorPaths::new(race.path().join("authority")).unwrap();
    let barrier = std::sync::Barrier::new(2);
    let resolved = std::thread::scope(|scope| {
        let mut threads = Vec::new();
        for ordinal in 0..2_u64 {
            let paths = paths.clone();
            let executable = &executable;
            let barrier = &barrier;
            let mut candidate = original.clone();
            candidate.local_memory_capacity_bytes = Some(4 * 1024 * 1024 * 1024 + ordinal * 1024);
            threads.push(scope.spawn(move || {
                resolve_default_profile(paths, executable, || {
                    // Both readers have observed genuine absence before either
                    // can publish a different sampled budget.
                    barrier.wait();
                    Ok(candidate)
                })
                .unwrap()
            }));
        }
        threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(resolved[0].execution, resolved[1].execution);
    assert_eq!(
        resolved[0].identity().unwrap(),
        resolved[1].identity().unwrap()
    );
    assert_eq!(
        read_profile(&paths.root.join("supervisor-profile.json"))
            .unwrap()
            .execution,
        resolved[0].execution
    );
}

#[test]
fn default_profile_rejects_untrusted_or_foreign_existing_state_without_discovery() {
    let original = NativeExecutionConfig::for_current_executable().unwrap();
    let executable = original.executable.clone();
    #[derive(Clone, Copy, Debug)]
    enum Case {
        Corrupt,
        ForeignRoot,
        DifferentExecutable,
        Oversize,
        Directory,
        #[cfg(unix)]
        Symlink,
        #[cfg(unix)]
        WideFile,
        #[cfg(unix)]
        WideRoot,
    }
    let cases = [
        Case::Corrupt,
        Case::ForeignRoot,
        Case::DifferentExecutable,
        Case::Oversize,
        Case::Directory,
        #[cfg(unix)]
        Case::Symlink,
        #[cfg(unix)]
        Case::WideFile,
        #[cfg(unix)]
        Case::WideRoot,
    ];
    for case in cases {
        let temp = tempfile::tempdir().unwrap();
        let paths = SupervisorPaths::new(temp.path().join("authority")).unwrap();
        let config = LocalSupervisorConfig::new(original.clone(), paths.clone()).unwrap();
        let path = publish_profile(&config).unwrap();
        match case {
            Case::Corrupt => std::fs::write(&path, b"not a profile").unwrap(),
            Case::ForeignRoot | Case::DifferentExecutable => {
                let mut foreign = config.execution.clone();
                if matches!(case, Case::ForeignRoot) {
                    foreign.store_root = temp.path().join("different-root");
                } else {
                    foreign.executable = executable.with_file_name("different-runmat-executable");
                }
                std::fs::write(
                    &path,
                    serde_json::to_vec(&Profile {
                        schema_version: 1,
                        execution: foreign,
                    })
                    .unwrap(),
                )
                .unwrap();
            }
            Case::Oversize => std::fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap()
                .set_len(PROFILE_MAX_BYTES + 1)
                .unwrap(),
            Case::Directory => {
                std::fs::remove_file(&path).unwrap();
                std::fs::create_dir(&path).unwrap();
            }
            #[cfg(unix)]
            Case::Symlink => {
                std::fs::remove_file(&path).unwrap();
                std::os::unix::fs::symlink(temp.path().join("absent"), &path).unwrap();
            }
            #[cfg(unix)]
            Case::WideFile | Case::WideRoot => {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(
                    if matches!(case, Case::WideFile) {
                        &path
                    } else {
                        &paths.root
                    },
                    std::fs::Permissions::from_mode(if matches!(case, Case::WideFile) {
                        0o644
                    } else {
                        0o777
                    }),
                )
                .unwrap();
            }
        }
        let before = std::fs::read(&path).ok();
        assert!(
            resolve_default_profile(paths, &executable, || {
                panic!("{case:?}: invalid existing state is not absence")
            })
            .is_err(),
            "{case:?}"
        );
        assert_eq!(
            std::fs::read(&path).ok(),
            before,
            "{case:?}: must not replace state"
        );
    }
}
