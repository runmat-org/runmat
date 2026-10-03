use super::*;
use std::future::Future;
use std::io::{Cursor, Write};
use std::sync::Arc;
use std::task::{Context, Poll};
use zip::write::SimpleFileOptions;

fn archive_bytes(contents: &[u8]) -> Vec<u8> {
    let mut archive = zip::ZipWriter::new(Cursor::new(Vec::new()));
    archive
        .start_file("owner.txt", SimpleFileOptions::default())
        .expect("start archive entry");
    archive.write_all(contents).expect("write archive entry");
    archive.finish().expect("finish archive").into_inner()
}

#[test]
fn selected_filesystem_owns_local_archive_input_and_output_after_global_swap() {
    let _lock = vfs::provider_override_lock();
    let selected = Arc::new(vfs::MemoryFsProvider::new());
    let replacement = Arc::new(vfs::MemoryFsProvider::new());
    let archive = Path::new("/archive.zip");
    let output = Path::new("/selected-output");
    selected
        .write_project_path(archive, &archive_bytes(b"selected owner"))
        .expect("selected archive");
    replacement
        .write_project_path(archive, &archive_bytes(b"replacement owner"))
        .expect("replacement archive at the same path");

    // Capture the exact provider that admission would retain. Suspend the
    // invocation before its open, then replace the ambient owner while it is
    // awaiting continuation.
    let _selected_guard = vfs::replace_provider(selected.clone());
    let selected_provider = vfs::current_provider();
    let (resume, suspended) = futures::channel::oneshot::channel::<()>();
    let mut invocation = Box::pin(async move {
        suspended.await.expect("resume selected invocation");
        extract_local_archive(archive, output, None, selected_provider).await
    });
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(
        invocation.as_mut().poll(&mut context),
        Poll::Pending
    ));
    let _replacement_guard = vfs::replace_provider(replacement.clone());
    resume.send(()).expect("resume suspended invocation");
    let files =
        futures::executor::block_on(invocation).expect("local extraction through selected owner");

    assert_eq!(files, vec!["/selected-output/owner.txt"]);
    assert_eq!(
        selected
            .read_project_path("/selected-output/owner.txt")
            .unwrap(),
        b"selected owner"
    );
    assert!(replacement
        .read_project_path("/selected-output/owner.txt")
        .is_err());
}
