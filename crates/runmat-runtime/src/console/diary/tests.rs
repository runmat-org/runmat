use super::*;
use futures::executor::block_on;
use std::sync::atomic::{AtomicBool, Ordering};

#[path = "tests/reentrant_provider.rs"]
mod reentrant_provider;

struct RevocableResourceLease(Arc<AtomicBool>, FsResourceContract);

impl FsResourceLease for RevocableResourceLease {
    fn contract(&self) -> FsResourceContract {
        self.1
    }

    fn is_valid(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

#[test]
fn active_diary_keeps_original_provider_and_directory_across_ambient_swap() {
    let _lock = fs::provider_override_lock();
    set_diary_enabled(false);
    let first = Arc::new(fs::MemoryFsProvider::with_current_dir("/first"));
    let first_guard = fs::replace_provider(first.clone());
    fs::set_current_dir("/first").expect("first fixture directory");
    set_diary_filename_checked("session.log").expect("enable diary");
    record_diary_command("first command");

    let second = Arc::new(fs::MemoryFsProvider::with_current_dir("/second"));
    let second_guard = fs::replace_provider(second.clone());
    fs::set_current_dir("/second").expect("second fixture directory");
    write_diary_entry(&ConsoleEntry {
        stream: ConsoleStream::Stdout,
        text: "second output\n".into(),
        timestamp_ms: 0,
    });
    assert_eq!(diary_filename(), PathBuf::from("session.log"));
    assert!(diary_enabled());
    set_diary_enabled(false);

    let original = block_on(first.read(Path::new("/first/session.log"))).expect("original log");
    assert_eq!(
        String::from_utf8(original).expect("text"),
        "first command\nsecond output\n"
    );
    assert!(block_on(second.read(Path::new("/second/session.log"))).is_err());
    drop(second_guard);
    drop(first_guard);
}

#[test]
fn off_then_on_captures_a_new_owner_and_appends_without_replaying_old_sink() {
    let _lock = fs::provider_override_lock();
    set_diary_enabled(false);
    let first = Arc::new(fs::MemoryFsProvider::with_current_dir("/first"));
    let first_guard = fs::replace_provider(first.clone());
    fs::set_current_dir("/first").expect("first fixture directory");
    set_diary_filename_checked("session.log").expect("first enable");
    record_diary_command("one");
    set_diary_enabled(false);

    let second = Arc::new(fs::MemoryFsProvider::with_current_dir("/second"));
    let second_guard = fs::replace_provider(second.clone());
    fs::set_current_dir("/second").expect("second fixture directory");
    set_diary_enabled_checked(true).expect("second enable");
    record_diary_command("two");
    set_diary_enabled(false);

    assert_eq!(
        block_on(first.read(Path::new("/first/session.log")))
            .expect("first log")
            .as_slice(),
        b"one\n"
    );
    assert_eq!(
        block_on(second.read(Path::new("/second/session.log")))
            .expect("second log")
            .as_slice(),
        b"two\n"
    );
    drop(second_guard);
    drop(first_guard);
}

#[test]
fn deferred_write_failure_retains_original_io_error_object_once() {
    let _lock = fs::provider_override_lock();
    set_diary_enabled(false);
    let _ = take_diary_error();
    record_diary_open_failure(io::Error::new(io::ErrorKind::BrokenPipe, "first failure"));
    record_diary_open_failure(io::Error::new(
        io::ErrorKind::PermissionDenied,
        "later failure",
    ));
    let error = take_diary_error().expect("original failure");
    assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
    assert_eq!(error.to_string(), "first failure");
    assert!(take_diary_error().is_none());
}

#[test]
fn invalidated_provider_resource_lease_stops_before_next_write() {
    let _lock = fs::provider_override_lock();
    set_diary_enabled(false);
    let _ = take_diary_error();
    let provider = Arc::new(fs::MemoryFsProvider::with_current_dir("/journal"));
    block_on(provider.write(Path::new("/journal/session.log"), b""))
        .expect("persist empty fixture before installing revocable sink");
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    let file = options
        .open_on(provider.as_ref(), Path::new("/journal/session.log"))
        .expect("open fixture");
    let valid = Arc::new(AtomicBool::new(true));
    let contract = FsResourceContract {
        revision: fs::FsResourceRevision::for_provider("diary-test", b"one"),
        opened_input_handle: None,
        opened_output_handle: None,
    };
    let lease: Arc<dyn FsResourceLease> = Arc::new(RevocableResourceLease(valid.clone(), contract));
    super::super::with_console_state(|state| {
        state.diary.sink = Some(Rc::new(RefCell::new(DiarySink {
            _provider: provider.clone(),
            _resolved_path: PathBuf::from("/journal/session.log"),
            resource_lease: Some((lease, contract)),
            file,
        })));
    });

    valid.store(false, Ordering::SeqCst);
    record_diary_command("must not be written");
    assert!(!diary_enabled());
    let error = take_diary_error().expect("resource lease cause");
    assert!(error.to_string().contains("invalidated"));
    let bytes = block_on(provider.read(Path::new("/journal/session.log"))).expect("log");
    assert!(bytes.is_empty());
}
