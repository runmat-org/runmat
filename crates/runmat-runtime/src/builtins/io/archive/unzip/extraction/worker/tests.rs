use super::*;
use runmat_execution::profiling::{
    with_command_profiling, CommandProfilingRoot, CommandProfilingSnapshot,
    CommandThreadCpuSampler, DeferredWorkCounts,
};
use std::future::Future;
use std::io::Write;
use std::sync::{mpsc, Condvar, Mutex};

#[derive(Default)]
struct ThreadSamples {
    counts: std::collections::HashMap<std::thread::ThreadId, u64>,
    origins: Vec<std::thread::ThreadId>,
}

struct RecordingThreadClock(Mutex<ThreadSamples>);

impl CommandThreadCpuSampler for RecordingThreadClock {
    fn sample_current_thread(&self) -> Option<std::time::Duration> {
        let mut samples = self.0.lock().ok()?;
        let origin = std::thread::current().id();
        let count = samples.counts.entry(origin).or_default();
        *count = count.checked_add(1)?;
        let elapsed = count.checked_mul(10)?;
        samples.origins.push(origin);
        Some(std::time::Duration::from_nanos(elapsed))
    }
}

#[test]
fn worker_samples_explicit_origin_even_while_caller_selects_other_command() {
    let provider = std::sync::Arc::new(vfs::MemoryFsProvider::new());
    futures::executor::block_on(vfs::FsProvider::create_dir_all(
        provider.as_ref(),
        Path::new("/output"),
    ))
    .unwrap();
    let clock = std::sync::Arc::new(RecordingThreadClock(Mutex::new(ThreadSamples::default())));
    let worker_root = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(clock.clone()));
    let caller_root = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(clock.clone()));
    let caller_thread = std::thread::current().id();
    let (input, _) = source(&archive());
    with_command_profiling(Some(caller_root.receipt()), || {
        futures::executor::block_on(extract_archive_file_on_worker(
            input,
            PathBuf::from("/output"),
            None,
            Some(worker_root.receipt()),
            provider,
            None,
        ))
        .expect("extraction");
    });
    let samples = clock.0.lock().unwrap();
    assert_eq!(samples.origins.len(), 4);
    assert_eq!(
        samples
            .origins
            .iter()
            .filter(|thread| **thread == caller_thread)
            .count(),
        2
    );
    assert_eq!(
        samples
            .origins
            .iter()
            .filter(|thread| **thread != caller_thread)
            .count(),
        2
    );
    drop(samples);
    assert!(worker_root.seal().cpu_thread_attributed.is_some());
    assert!(caller_root.seal().cpu_thread_attributed.is_some());
}

fn source(bytes: &[u8]) -> (super::super::super::source::TempArchive, PathBuf) {
    let mut input = super::super::super::source::TempArchive::create().unwrap();
    let path = input.path().to_path_buf();
    input.write_all(bytes).unwrap();
    input.flush().unwrap();
    input.close();
    (input, path)
}

fn archive() -> Vec<u8> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = zip::ZipWriter::new(&mut cursor);
        writer
            .start_file("entry.txt", zip::write::SimpleFileOptions::default())
            .unwrap();
        writer.write_all(b"owned extraction").unwrap();
        writer.finish().unwrap();
    }
    cursor.into_inner()
}

fn handoff(
    input: super::super::super::source::TempArchive,
    origin: &CommandProfilingRoot,
    resource_lease: ArchiveResourceLease,
) -> WorkerHandoff {
    WorkerHandoff {
        phase: WorkerPhase::Unstarted,
        job: Some(WorkerJob {
            input,
            output_folder: PathBuf::from("/unused"),
            cancellation: None,
            abandonment: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            provider: std::sync::Arc::new(vfs::MemoryFsProvider::new()),
            resource_lease,
        }),
        token: origin.receipt().begin_deferred_work(),
        command_origin: Some(origin.receipt()),
        sender: None,
    }
}

struct SealWhileDropping {
    root: std::sync::Arc<Mutex<Option<CommandProfilingRoot>>>,
    report: mpsc::Sender<CommandProfilingSnapshot>,
}

impl crate::context::host_effect::HostEffectResourceLivenessLease for SealWhileDropping {
    fn is_current(&self) -> bool {
        true
    }
}

impl Drop for SealWhileDropping {
    fn drop(&mut self) {
        let root = self.root.lock().unwrap().take().expect("one root");
        self.report.send(root.seal()).expect("observe during Drop");
    }
}

struct BlockingLease {
    first: std::sync::atomic::AtomicBool,
    gate: std::sync::Arc<(Mutex<bool>, Condvar)>,
    entered: mpsc::Sender<()>,
    retired: mpsc::Sender<()>,
}

impl crate::context::host_effect::HostEffectResourceLivenessLease for BlockingLease {
    fn is_current(&self) -> bool {
        if !self.first.swap(true, std::sync::atomic::Ordering::AcqRel) {
            self.entered
                .send(())
                .expect("worker reached provider lease");
            let (lock, wake) = &*self.gate;
            let mut open = lock.lock().unwrap();
            while !*open {
                open = wake.wait(open).unwrap();
            }
        }
        true
    }
}

impl Drop for BlockingLease {
    fn drop(&mut self) {
        self.retired.send(()).expect("worker lease retired");
    }
}

#[test]
fn cancelled_caller_cannot_retire_running_worker_before_immutable_early_seal() {
    let root = CommandProfilingRoot::new();
    let other = CommandProfilingRoot::new();
    let gate = std::sync::Arc::new((Mutex::new(false), Condvar::new()));
    let (entered, reached) = mpsc::channel();
    let (retired, done) = mpsc::channel();
    let lease: ArchiveResourceLease = Some(std::sync::Arc::new(BlockingLease {
        first: std::sync::atomic::AtomicBool::new(false),
        gate: gate.clone(),
        entered,
        retired,
    }));
    let (input, path) = source(&archive());
    let mut call = Box::pin(extract_archive_file_on_worker(
        input,
        PathBuf::from("/output"),
        None,
        Some(root.receipt()),
        std::sync::Arc::new(vfs::MemoryFsProvider::new()),
        lease,
    ));
    let waker = futures::task::noop_waker();
    let mut context = std::task::Context::from_waker(&waker);
    assert!(matches!(
        call.as_mut().poll(&mut context),
        std::task::Poll::Pending
    ));
    reached
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("worker reached captured lease");
    drop(call);
    let published = root.seal();
    assert_eq!(
        published.deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            pending: 1,
            ..DeferredWorkCounts::default()
        })
    );
    assert_eq!(
        other.seal().deferred_work,
        Some(DeferredWorkCounts::default())
    );
    let (lock, wake) = &*gate;
    *lock.lock().unwrap() = true;
    wake.notify_one();
    done.recv_timeout(std::time::Duration::from_secs(2))
        .expect("worker retired its captured lease");
    assert_eq!(published.deferred_work.unwrap().pending, 1);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while path.exists() && std::time::Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert!(
        !path.exists(),
        "worker source retired after caller cancellation"
    );
}

#[test]
fn rejected_unstarted_spawn_drops_resources_before_failed_settlement() {
    let root = CommandProfilingRoot::new();
    let (input, path) = source(b"unstarted");
    let rejected = spawn_owned(handoff(input, &root, None), |handoff| {
        drop(handoff);
        Err(std::io::Error::other("test spawn rejection"))
    });
    assert!(rejected.is_err());
    assert!(!path.exists(), "source retired before result");
    assert_eq!(
        root.seal().deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            failed: 1,
            ..DeferredWorkCounts::default()
        })
    );

    let root = std::sync::Arc::new(Mutex::new(Some(CommandProfilingRoot::new())));
    let (report, observed) = mpsc::channel();
    let lease: ArchiveResourceLease = Some(std::sync::Arc::new(SealWhileDropping {
        root: root.clone(),
        report,
    }));
    let (input, _) = source(b"drop-order");
    let handoff = handoff(input, root.lock().unwrap().as_ref().unwrap(), lease);
    // The handoff's token is the admitted receipt, not a new root.
    assert!(handoff.token.is_some());
    let rejected = spawn_owned(handoff, |handoff| {
        drop(handoff);
        Err(std::io::Error::other("test spawn rejection"))
    });
    assert!(rejected.is_err());
    assert_eq!(
        observed.recv().unwrap().deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            pending: 1,
            ..DeferredWorkCounts::default()
        }),
        "the resource destructor must run before the token can settle",
    );
}

#[test]
fn started_unwind_is_missing_after_job_resource_destruction() {
    let root = CommandProfilingRoot::new();
    let (input, path) = source(b"panic");
    let handoff = handoff(input, &root, None);
    let worker = std::thread::spawn(move || {
        let mut handoff = handoff;
        handoff.phase = WorkerPhase::Running;
        panic!("worker panic witness");
    });
    assert!(worker.join().is_err());
    assert!(!path.exists());
    assert_eq!(
        root.seal().deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            missing: 1,
            ..DeferredWorkCounts::default()
        })
    );
}

#[test]
fn actual_worker_settles_before_result_delivery() {
    let provider = std::sync::Arc::new(vfs::MemoryFsProvider::new());
    futures::executor::block_on(vfs::FsProvider::create_dir_all(
        provider.as_ref(),
        Path::new("/output"),
    ))
    .unwrap();
    let root = CommandProfilingRoot::new();
    let (input, path) = source(&archive());
    let names = futures::executor::block_on(extract_archive_file_on_worker(
        input,
        PathBuf::from("/output"),
        None,
        Some(root.receipt()),
        provider.clone(),
        None,
    ))
    .unwrap();
    assert_eq!(names.len(), 1);
    assert_eq!(
        provider.read_project_path("/output/entry.txt").unwrap(),
        b"owned extraction"
    );
    assert!(!path.exists());
    assert_eq!(
        root.seal().deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            completed: 1,
            ..DeferredWorkCounts::default()
        })
    );

    let root = std::sync::Arc::new(Mutex::new(Some(CommandProfilingRoot::new())));
    let (report, observed) = mpsc::channel();
    let lease: ArchiveResourceLease = Some(std::sync::Arc::new(SealWhileDropping {
        root: root.clone(),
        report,
    }));
    let (input, path) = source(&archive());
    let receipt = root.lock().unwrap().as_ref().unwrap().receipt();
    let names = futures::executor::block_on(extract_archive_file_on_worker(
        input,
        PathBuf::from("/output"),
        None,
        Some(receipt),
        provider.clone(),
        lease,
    ))
    .unwrap();
    assert_eq!(
        names,
        vec![PathBuf::from("/output")
            .join("entry.txt")
            .to_string_lossy()
            .into_owned()]
    );
    assert!(!path.exists());
    assert_eq!(
        observed.recv().unwrap().deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            pending: 1,
            ..DeferredWorkCounts::default()
        }),
        "normal worker resource destruction precedes terminal settlement",
    );

    let root = CommandProfilingRoot::new();
    let (input, path) = source(b"not a ZIP");
    let error = futures::executor::block_on(extract_archive_file_on_worker(
        input,
        PathBuf::from("/output"),
        None,
        Some(root.receipt()),
        provider,
        None,
    ))
    .expect_err("invalid archive");
    assert_eq!(error.identifier(), UNZIP_ERROR_ZIP.identifier);
    assert!(!path.exists());
    assert_eq!(
        root.seal().deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            failed: 1,
            ..DeferredWorkCounts::default()
        })
    );
}
