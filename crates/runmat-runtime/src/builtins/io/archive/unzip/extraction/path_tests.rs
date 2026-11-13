use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn repeated_same_target_temporary_paths_are_distinct() {
    let target = Path::new("/unzip/output/entry.txt");
    let paths = (0..32)
        .map(|_| temporary_output_path(target))
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(paths.len(), 32);
    for path in paths {
        assert_eq!(path.parent(), target.parent());
        assert!(path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with(".entry.txt.runmat-unzip-tmp-"));
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn exclusive_temp_collision_preserves_existing_file_and_guard_cleans_own_file() {
    let provider = std::sync::Arc::new(vfs::MemoryFsProvider::new());
    let target = Path::new("/unzip/output/entry.txt");
    let temporary = temporary_output_path(target);
    futures::executor::block_on(vfs::FsProvider::create_dir_all(
        provider.as_ref(),
        target.parent().unwrap(),
    ))
    .unwrap();
    provider
        .write_project_path(&temporary, b"unrelated existing bytes")
        .unwrap();

    let error =
        futures::executor::block_on(PartialEntry::create(temporary.clone(), provider.clone()))
            .err()
            .expect("create_new must reject a collision");
    assert_eq!(error.identifier(), UNZIP_ERROR_IO.identifier);
    assert_eq!(
        provider.read_project_path(&temporary).unwrap(),
        b"unrelated existing bytes"
    );

    let own_path = temporary_output_path(target);
    assert_ne!(own_path, temporary);
    futures::executor::block_on(async {
        let mut partial = PartialEntry::create(own_path.clone(), provider.clone())
            .await
            .expect("exclusive own partial");
        partial.write_all(b"unpublished bytes").unwrap();
        drop(partial);
    });
    assert!(provider.read_project_path(&own_path).is_err());
    assert_eq!(
        provider.read_project_path(&temporary).unwrap(),
        b"unrelated existing bytes"
    );
}
