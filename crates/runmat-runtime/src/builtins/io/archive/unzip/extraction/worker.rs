//! Linear ownership of an admitted native extraction until its actual worker
//! resources have retired, even when the awaiting invocation is abandoned.

use super::*;
use runmat_execution::profiling::{
    with_command_profiling, CommandDeferredWork, CommandProfilingReceipt, DeferredWorkTerminal,
};

enum WorkerPhase {
    Unstarted,
    Running,
}

struct WorkerJob {
    input: super::super::source::TempArchive,
    output_folder: PathBuf,
    cancellation: Option<runmat_execution::cancellation::CancellationToken>,
    abandonment: std::sync::Arc<std::sync::atomic::AtomicBool>,
    provider: std::sync::Arc<dyn vfs::FsProvider>,
    resource_lease: ArchiveResourceLease,
}

impl WorkerJob {
    fn run(self) -> BuiltinResult<Vec<String>> {
        let Self {
            input,
            output_folder,
            cancellation,
            abandonment,
            provider,
            resource_lease,
        } = self;
        let result = (|| {
            check_resource_lease(&resource_lease)?;
            let file = std::fs::File::open(input.path()).map_err(|err| {
                unzip_error_with_source(
                    &UNZIP_ERROR_IO,
                    format!("unzip: unable to open '{}': {err}", input.path().display()),
                    err,
                )
            })?;
            check_resource_lease(&resource_lease)?;
            let control = ExtractionControl::native(cancellation, abandonment, resource_lease);
            futures::executor::block_on(extract_archive_impl(
                file,
                &output_folder,
                control,
                provider,
            ))
        })();
        // The temporary source is a worker capture, not an invocation capture.
        // On either return path it retires before the outer token is settled.
        drop(input);
        result
    }
}

struct WorkerHandoff {
    phase: WorkerPhase,
    job: Option<WorkerJob>,
    token: Option<CommandDeferredWork>,
    command_origin: Option<CommandProfilingReceipt>,
    sender: Option<futures::channel::oneshot::Sender<BuiltinResult<Vec<String>>>>,
}

impl WorkerHandoff {
    fn run(mut self) {
        // A rejected Builder::spawn never executes this first instruction.
        self.phase = WorkerPhase::Running;
        let result = with_command_profiling(self.command_origin.take(), || {
            self.job.take().expect("one extraction job").run()
        });
        if let Some(token) = self.token.take() {
            token.complete(if result.is_ok() {
                DeferredWorkTerminal::Completed
            } else {
                DeferredWorkTerminal::Failed
            });
        }
        // Delivery can wake and seal the caller immediately. It must follow
        // the worker resource retirement and deferred-work terminal fact.
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(result);
        }
    }
}

impl Drop for WorkerHandoff {
    fn drop(&mut self) {
        // Builder::spawn consumes and drops its closure before returning Err.
        // An unstarted closure therefore classifies its own rejection. On a
        // started unwind, the inner job frame has already dropped its captures.
        // If destruction here panics, the still-owned token drops as Missing.
        drop(self.job.take());
        if let Some(token) = self.token.take() {
            let terminal = match self.phase {
                WorkerPhase::Unstarted => DeferredWorkTerminal::Failed,
                WorkerPhase::Running => DeferredWorkTerminal::Missing,
            };
            token.complete(terminal);
        }
    }
}

fn spawn_owned(
    handoff: WorkerHandoff,
    spawn: impl FnOnce(WorkerHandoff) -> std::io::Result<()>,
) -> std::io::Result<()> {
    spawn(handoff)
}

pub(in super::super) async fn extract_archive_file_on_worker(
    input: super::super::source::TempArchive,
    output_folder: PathBuf,
    cancellation: Option<runmat_execution::cancellation::CancellationToken>,
    command_origin: Option<CommandProfilingReceipt>,
    provider: std::sync::Arc<dyn vfs::FsProvider>,
    resource_lease: ArchiveResourceLease,
) -> BuiltinResult<Vec<String>> {
    let (sender, receiver) = futures::channel::oneshot::channel();
    let abandonment = control::Abandonment::new();
    let token = command_origin
        .as_ref()
        .and_then(CommandProfilingReceipt::begin_deferred_work);
    let handoff = WorkerHandoff {
        phase: WorkerPhase::Unstarted,
        job: Some(WorkerJob {
            input,
            output_folder,
            cancellation,
            abandonment: abandonment.worker_flag(),
            provider,
            resource_lease,
        }),
        token,
        command_origin,
        sender: Some(sender),
    };
    spawn_owned(handoff, |handoff| {
        std::thread::Builder::new()
            .name("runmat-unzip-extract".to_string())
            .spawn(move || handoff.run())
            .map(|_| ())
    })
    .map_err(|err| {
        unzip_error_with_source(
            &UNZIP_ERROR_IO,
            format!("unzip: unable to spawn extraction worker: {err}"),
            err,
        )
    })?;
    let result = receiver.await.map_err(|err| {
        unzip_error_with_source(
            &UNZIP_ERROR_IO,
            format!("unzip: extraction worker did not return a result: {err}"),
            err,
        )
    })?;
    abandonment.disarm();
    result
}

#[cfg(test)]
#[path = "worker/tests.rs"]
mod tests;
