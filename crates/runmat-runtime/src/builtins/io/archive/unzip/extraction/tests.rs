use super::*;
#[cfg(not(target_arch = "wasm32"))]
use futures::executor::block_on;
#[cfg(not(target_arch = "wasm32"))]
use std::io::Cursor;

#[cfg(not(target_arch = "wasm32"))]
fn two_entry_archive() -> Vec<u8> {
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = zip::ZipWriter::new(&mut cursor);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        writer.start_file("good.txt", options).unwrap();
        writer.write_all(b"first entry stays").unwrap();
        writer.start_file("bad.txt", options).unwrap();
        writer.write_all(b"second entry fails CRC").unwrap();
        writer.finish().unwrap();
    }
    cursor.into_inner()
}

#[cfg(not(target_arch = "wasm32"))]
fn native_control() -> ExtractionControl {
    ExtractionControl::native(
        None,
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        None,
    )
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn failed_later_entry_removes_only_its_partial_output() {
    let _provider_lock = vfs::provider_override_lock();
    let output = tempfile::tempdir().unwrap();
    let mut bytes = two_entry_archive();
    let needle = b"second entry fails CRC";
    let offset = bytes
        .windows(needle.len())
        .position(|part| part == needle)
        .unwrap();
    bytes[offset] ^= 1;

    let error = block_on(extract_archive_impl(
        Cursor::new(bytes),
        output.path(),
        native_control(),
        vfs::current_provider(),
    ))
    .expect_err("CRC failure in second entry");
    assert_eq!(error.identifier(), UNZIP_ERROR_ZIP.identifier);
    assert_eq!(
        std::fs::read(output.path().join("good.txt")).unwrap(),
        b"first entry stays"
    );
    assert!(!output.path().join("bad.txt").exists());
    let names = std::fs::read_dir(output.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect::<Vec<_>>();
    assert_eq!(names, vec![std::ffi::OsString::from("good.txt")]);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn abandonment_and_token_stop_before_publishing_an_entry() {
    let output = tempfile::tempdir().unwrap();
    let lease = control::Abandonment::new();
    let worker_flag = lease.worker_flag();
    drop(lease);
    let error = block_on(extract_archive_impl(
        Cursor::new(two_entry_archive()),
        output.path(),
        ExtractionControl::native(None, worker_flag, None),
        vfs::current_provider(),
    ))
    .expect_err("abandoned worker");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(std::fs::read_dir(output.path()).unwrap().next().is_none());

    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let control = ExtractionControl::native(
        Some(cancellation.token()),
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        None,
    );
    cancellation.cancel(runmat_execution::CancellationReason::User);
    let error = block_on(extract_archive_impl(
        Cursor::new(two_entry_archive()),
        output.path(),
        control,
        vfs::current_provider(),
    ))
    .expect_err("cancelled worker");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(std::fs::read_dir(output.path()).unwrap().next().is_none());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn dropped_partial_and_source_leases_remove_unpublished_files() {
    let _provider_lock = vfs::provider_override_lock();
    let output = tempfile::tempdir().unwrap();
    let partial_path = output.path().join(".partial");
    block_on(async {
        let mut partial = PartialEntry::create(partial_path.clone(), vfs::current_provider())
            .await
            .unwrap();
        partial.write_all(b"not published").unwrap();
        drop(partial);
    });
    assert!(!partial_path.exists());

    let mut source = super::super::source::TempArchive::create().unwrap();
    let source_path = source.path().to_path_buf();
    source.write_all(b"not consumed").unwrap();
    drop(source);
    assert!(!source_path.exists());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn worker_uses_captured_output_provider_not_thread_ambient_provider() {
    let _provider_lock = vfs::provider_override_lock();
    let provider = std::sync::Arc::new(vfs::MemoryFsProvider::new());
    block_on(vfs::FsProvider::create_dir_all(
        provider.as_ref(),
        Path::new("/captured-output"),
    ))
    .unwrap();
    let mut input = super::super::source::TempArchive::create().unwrap();
    input.write_all(&two_entry_archive()).unwrap();
    input.flush().unwrap();
    input.close();
    let input_path = input.path().to_path_buf();

    let names = block_on(extract_archive_file_on_worker(
        input,
        PathBuf::from("/captured-output"),
        None,
        None,
        provider.clone(),
        None,
    ))
    .expect("worker extracts through captured memory provider");
    assert_eq!(names.len(), 2);
    assert_eq!(
        provider
            .read_project_path("/captured-output/good.txt")
            .unwrap(),
        b"first entry stays"
    );
    assert_eq!(
        provider
            .read_project_path("/captured-output/bad.txt")
            .unwrap(),
        b"second entry fails CRC"
    );
    assert!(!input_path.exists());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn revoked_worker_lease_rejects_before_opening_or_parsing_source() {
    use std::sync::atomic::{AtomicBool, Ordering};
    struct Revoked(std::sync::Arc<AtomicBool>);
    impl crate::context::host_effect::HostEffectResourceLivenessLease for Revoked {
        fn is_current(&self) -> bool {
            self.0.load(Ordering::SeqCst)
        }
    }

    let output = tempfile::tempdir().unwrap();
    let mut input = super::super::source::TempArchive::create().unwrap();
    input.write_all(b"invalid ZIP must not be parsed").unwrap();
    input.close();
    let valid = std::sync::Arc::new(AtomicBool::new(false));
    let error = block_on(extract_archive_file_on_worker(
        input,
        output.path().to_path_buf(),
        None,
        None,
        vfs::current_provider(),
        Some(std::sync::Arc::new(Revoked(valid))),
    ))
    .expect_err("revoked lease must precede source parse");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:StaleResourceAuthority")
    );
    assert!(std::fs::read_dir(output.path()).unwrap().next().is_none());
}
