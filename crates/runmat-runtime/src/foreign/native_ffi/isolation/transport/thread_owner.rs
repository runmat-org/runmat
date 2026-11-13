//! Join evidence without blocking the evaluator on native TLS destructors.

use std::{sync::Arc, thread::JoinHandle};

use super::{HostExit, NativeChildResources};
use futures::channel::oneshot;

pub(in super::super) struct ThreadOwner {
    pub(super) resources: Arc<NativeChildResources>,
    completion: oneshot::Receiver<Result<(), String>>,
    outcome: Option<Result<(), String>>,
}

impl ThreadOwner {
    pub(super) fn new(
        thread: JoinHandle<Result<(), String>>,
        resources: Arc<NativeChildResources>,
    ) -> Result<Self, String> {
        let (send, completion) = oneshot::channel();
        let retirement_resources = Arc::clone(&resources);
        std::thread::Builder::new()
            .name("runmat-native-ffi-retirement".into())
            .spawn(move || {
                let result = thread
                    .join()
                    .unwrap_or_else(|_| Err("native host thread panicked".into()));
                // Parent abandonment cannot delete private native bytes
                // while the native thread is still running TLS destructors.
                drop(retirement_resources);
                // Only an actual join, including native thread-local destruction,
                // can publish this completion. No evaluator thread blocks here.
                let _ = send.send(result);
            })
            .map_err(|error| format!("could not start native host retirement owner: {error}"))?;
        Ok(Self {
            resources,
            completion,
            outcome: None,
        })
    }

    pub(super) fn diagnostic(&self) -> String {
        self.outcome
            .as_ref()
            .and_then(|result| result.as_ref().err())
            .cloned()
            .unwrap_or_default()
    }

    fn exit(&self) -> HostExit {
        HostExit {
            success: self
                .outcome
                .as_ref()
                .expect("native host is joined")
                .is_ok(),
            code: None,
        }
    }

    pub(super) fn try_wait(&mut self) -> Result<Option<HostExit>, String> {
        if self.outcome.is_none() {
            self.outcome = self
                .completion
                .try_recv()
                .map_err(|_| "native host retirement owner was lost before joining".to_string())?;
        }
        Ok(self.outcome.as_ref().map(|_| self.exit()))
    }

    pub(super) async fn wait(&mut self) -> Result<HostExit, String> {
        if self.outcome.is_none() {
            self.outcome =
                Some((&mut self.completion).await.map_err(|_| {
                    "native host retirement owner was lost before joining".to_string()
                })?);
        }
        Ok(self.exit())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, sync::mpsc};

    struct NativeThreadDestructor {
        entered: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
    }
    impl Drop for NativeThreadDestructor {
        fn drop(&mut self) {
            let _ = self.entered.send(());
            let _ = self.release.recv();
        }
    }

    thread_local! {
        static NATIVE_DESTRUCTOR: RefCell<Option<NativeThreadDestructor>> = const { RefCell::new(None) };
    }

    #[test]
    fn abandoned_owner_keeps_bytes_until_actual_native_thread_exit() {
        let snapshots = runmat_process_host::shared_memory::SharedSnapshotStore::create().unwrap();
        let root = snapshots.root_path().to_path_buf();
        let resources = Arc::new(NativeChildResources {
            snapshots,
            project_bytes: None,
        });
        let (release, blocked) = mpsc::channel();
        let (entered, destructing) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            NATIVE_DESTRUCTOR.with(|slot| {
                *slot.borrow_mut() = Some(NativeThreadDestructor {
                    entered,
                    release: blocked,
                });
            });
            Ok(())
        });
        let mut owner = ThreadOwner::new(thread, resources).unwrap();
        destructing.recv().unwrap();
        assert!(owner.try_wait().unwrap().is_none());
        let (_unused, replacement) = oneshot::channel();
        let completion = std::mem::replace(&mut owner.completion, replacement);
        drop(owner);
        assert!(
            root.is_dir(),
            "private bytes must outlive native TLS destruction"
        );
        release.send(()).unwrap();
        futures::executor::block_on(completion).unwrap().unwrap();
        assert!(
            !root.exists(),
            "joined abandoned native owner releases its bytes"
        );
    }
}
