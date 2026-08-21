use futures::executor::block_on;
use runmat_package::NormalizedRelativePath;
use runmat_package_cache::{
    BlobMetadata, CacheBackend, CacheObject, CacheTransaction, ObjectWrite, TreeEntry, TreeManifest,
};
use runmat_package_cache_native::filesystem::CacheLayout;
use runmat_package_cache_native::materialize::materialize_tree;
use runmat_package_cache_native::SqliteCacheBackend;
use std::sync::{Arc, Barrier};

#[test]
fn concurrent_materializers_publish_one_complete_tree() {
    let directory = tempfile::tempdir().unwrap();
    let layout = CacheLayout::new(directory.path().join("cache"));
    layout.create().unwrap();
    let backend = SqliteCacheBackend::open_path(&layout.database, None).unwrap();
    let blob = BlobMetadata::from_bytes(b"shared");
    let tree = TreeManifest::new(vec![TreeEntry::file(
        NormalizedRelativePath::new("main.m").unwrap(),
        blob.digest.clone(),
        6,
        false,
    )])
    .unwrap();
    block_on(async {
        let mut state = backend.snapshot().await.unwrap().state;
        state
            .objects
            .insert(blob.digest.clone(), CacheObject::Blob(blob.clone()));
        state
            .objects
            .insert(tree.digest.clone(), CacheObject::Tree(tree.clone()));
        let mut transaction = CacheTransaction::metadata_only(0, state);
        transaction.writes.insert(
            blob.digest.clone(),
            ObjectWrite::new(CacheObject::Blob(blob), Some(b"shared".to_vec())).unwrap(),
        );
        backend.commit(transaction).await.unwrap();
    });
    drop(backend);

    let barrier = Arc::new(Barrier::new(2));
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let barrier = barrier.clone();
            let layout = layout.clone();
            let tree = tree.clone();
            std::thread::spawn(move || {
                let backend = SqliteCacheBackend::open_path(&layout.database, None).unwrap();
                barrier.wait();
                block_on(materialize_tree(&backend, &layout, &tree)).unwrap()
            })
        })
        .collect();
    let paths: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();
    assert_eq!(paths[0], paths[1]);
    assert_eq!(std::fs::read(paths[0].join("main.m")).unwrap(), b"shared");
}
