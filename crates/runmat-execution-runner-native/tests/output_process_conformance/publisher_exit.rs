//! Test-owned physical publisher loss, independent of executor progress.
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::JoinHandle,
    time::Duration,
};

pub(super) struct Observer {
    stopped: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Observer {
    pub(super) fn start(marker: PathBuf) -> Self {
        let stopped = Arc::new(AtomicBool::new(false));
        let watch_stop = Arc::clone(&stopped);
        let thread = std::thread::Builder::new()
            .name("fixture-publisher-loss".into())
            .spawn(move || {
                while !watch_stop.load(Ordering::Acquire) {
                    if marker.exists() {
                        // Deliberate original-process loss: no completion footer
                        // or normal resource/result publication may run.
                        std::process::exit(19);
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            })
            .expect("start independent publisher-loss observer");
        Self {
            stopped,
            thread: Some(thread),
        }
    }
}

impl Drop for Observer {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::Release);
        self.thread
            .take()
            .expect("owned observer thread")
            .join()
            .expect("publisher-loss observer must stop without a panic");
    }
}
