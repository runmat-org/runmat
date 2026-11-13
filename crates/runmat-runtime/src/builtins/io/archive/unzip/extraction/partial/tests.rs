use super::*;
use runmat_execution::profiling::{CommandProfilingRoot, DeferredWorkCounts};
use std::sync::atomic::Ordering;

#[path = "tests/provider.rs"]
mod provider;
use provider::GatedRemoveProvider;

async fn until_removed(provider: &dyn vfs::FsProvider, path: &Path) {
    for _ in 0..16 {
        if provider.read(path).await.is_err() {
            return;
        }
        wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(
            &wasm_bindgen::JsValue::UNDEFINED,
        ))
        .await
        .unwrap();
    }
    assert!(provider.read(path).await.is_err(), "rollback completed");
}

#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn detached_rollback_has_exact_origin_and_immutable_early_snapshot() {
    let (provider, entered, release, closed) = GatedRemoveProvider::new();
    vfs::FsProvider::create_dir_all(provider.as_ref(), Path::new("/output"))
        .await
        .unwrap();
    let path = PathBuf::from("/output/.partial-early");
    let command = CommandProfilingRoot::new();
    let partial =
        PartialEntry::create_with_origin(path.clone(), provider.clone(), Some(command.receipt()))
            .await
            .unwrap();
    drop(partial);
    entered.await.expect("rollback entered provider removal");
    assert_eq!(
        closed.load(Ordering::SeqCst),
        1,
        "output handle closed before detached removal"
    );
    assert!(
        provider.read_project_path(&path).is_ok(),
        "removal is still gated"
    );
    let published = command.seal();
    assert_eq!(
        published.deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            pending: 1,
            ..DeferredWorkCounts::default()
        })
    );
    release.send(()).expect("release pending removal");
    until_removed(provider.as_ref(), &path).await;
    assert_eq!(published.deferred_work.unwrap().pending, 1);
}

#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn rollback_settles_after_delete_but_commit_and_sealed_origin_do_not_admit() {
    let (gated, entered, release, closed) = GatedRemoveProvider::new();
    let provider = gated.clone();
    vfs::FsProvider::create_dir_all(provider.as_ref(), Path::new("/output"))
        .await
        .unwrap();
    let path = PathBuf::from("/output/.partial-terminal");
    let command = CommandProfilingRoot::new();
    let partial =
        PartialEntry::create_with_origin(path.clone(), provider.clone(), Some(command.receipt()))
            .await
            .unwrap();
    drop(partial);
    entered.await.expect("rollback entered provider removal");
    assert_eq!(closed.load(Ordering::SeqCst), 1);
    assert!(
        provider.read_project_path(&path).is_ok(),
        "removal remains pending"
    );
    release.send(()).expect("release pending removal");
    until_removed(provider.as_ref(), &path).await;
    assert_eq!(
        command.seal().deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            completed: 1,
            ..DeferredWorkCounts::default()
        })
    );

    let path = PathBuf::from("/output/.partial-committed");
    let output = PathBuf::from("/output/committed.txt");
    let command = CommandProfilingRoot::new();
    PartialEntry::create_with_origin(path, provider.clone(), Some(command.receipt()))
        .await
        .unwrap()
        .commit_to(&output)
        .await
        .unwrap();
    assert!(provider.read_project_path(&output).is_ok());
    assert_eq!(
        command.seal().deferred_work,
        Some(DeferredWorkCounts::default()),
    );

    let path = PathBuf::from("/output/.partial-sealed");
    let command = CommandProfilingRoot::new();
    let partial =
        PartialEntry::create_with_origin(path.clone(), provider.clone(), Some(command.receipt()))
            .await
            .unwrap();
    assert_eq!(
        command.seal().deferred_work,
        Some(DeferredWorkCounts::default()),
    );
    drop(partial);
    until_removed(provider.as_ref(), &path).await;
}
