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
fn captured_relative_input_and_default_output_survive_provider_and_cwd_swap() {
    let _lock = vfs::provider_override_lock();
    let selected = Arc::new(vfs::MemoryFsProvider::with_current_dir("/selected-cwd"));
    let replacement = Arc::new(vfs::MemoryFsProvider::with_current_dir("/replacement-cwd"));
    selected
        .write_project_path(
            "/selected-cwd/archive.zip",
            &archive_bytes(b"selected owner"),
        )
        .expect("selected archive");
    replacement
        .write_project_path(
            "/replacement-cwd/archive.zip",
            &archive_bytes(b"replacement owner"),
        )
        .expect("replacement archive at the same relative path");

    // Capture the exact provider that admission would retain. Suspend the
    // invocation before its open, then replace the ambient owner while it is
    // awaiting continuation.
    let _selected_guard = vfs::replace_provider(selected.clone());
    vfs::set_current_dir("/selected-cwd").expect("selected working directory");
    let mut captured = None;
    let output = resolve_output_folder(None, &mut captured).expect("default output path");
    let archive = resolve_archive_path("archive.zip", &mut captured).expect("relative input path");
    assert_eq!(output, Path::new("/selected-cwd"));
    assert_eq!(archive, Path::new("/selected-cwd/archive.zip"));
    let selected_provider = captured.expect("selected owner and CWD captured").provider;
    let (resume, suspended) = futures::channel::oneshot::channel::<()>();
    let mut invocation = Box::pin(async move {
        suspended.await.expect("resume selected invocation");
        extract_local_archive(&archive, &output, None, None, selected_provider, &None).await
    });
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(
        invocation.as_mut().poll(&mut context),
        Poll::Pending
    ));
    let _replacement_guard = vfs::replace_provider(replacement.clone());
    vfs::set_current_dir("/replacement-cwd").expect("replacement working directory");
    resume.send(()).expect("resume suspended invocation");
    let files =
        futures::executor::block_on(invocation).expect("local extraction through selected owner");

    let expected = Path::new("/selected-cwd").join("owner.txt");
    assert_eq!(files, vec![expected.to_str().expect("UTF-8 fixture path")]);
    assert_eq!(
        selected
            .read_project_path("/selected-cwd/owner.txt")
            .unwrap(),
        b"selected owner"
    );
    assert!(replacement
        .read_project_path("/replacement-cwd/owner.txt")
        .is_err());
}
