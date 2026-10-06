//! Public print-future source/scene ownership through a suspended stage.

use super::*;
use async_trait::async_trait;
use futures::channel::oneshot;
use futures::task::noop_waker;
use runmat_filesystem::{
    self as fs, DirEntry, FileHandle, FsMetadata, FsProvider, FsPublicationFailure, FsStagedOutput,
    OpenFlags,
};
use std::future::Future;
use std::io;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

struct PausedProvider {
    gate: Mutex<Option<oneshot::Receiver<()>>>,
    published: Arc<Mutex<Option<Vec<u8>>>>,
}

struct PausedStage {
    gate: Option<oneshot::Receiver<()>>,
    bytes: Vec<u8>,
    published: Arc<Mutex<Option<Vec<u8>>>>,
}

#[async_trait(?Send)]
impl FsStagedOutput for PausedStage {
    async fn write_chunk(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if let Some(gate) = self.gate.take() {
            gate.await
                .map_err(|_| io::Error::other("test stage abandoned"))?;
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    async fn flush_and_sync(&mut self) -> io::Result<()> {
        Ok(())
    }
    async fn publish(self: Box<Self>) -> Result<(), FsPublicationFailure> {
        *self.published.lock().unwrap() = Some(self.bytes);
        Ok(())
    }
}

fn unused() -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, "unused fixture operation")
}

#[async_trait(?Send)]
impl FsProvider for PausedProvider {
    fn path_namespace(&self) -> fs::FsPathNamespace {
        fs::FsPathNamespace::VirtualRooted
    }
    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(PathBuf::from("/"))
    }
    fn begin_staged_replace(&self, _path: &Path) -> io::Result<Option<Box<dyn FsStagedOutput>>> {
        let gate = self
            .gate
            .lock()
            .unwrap()
            .take()
            .expect("one public print stage");
        Ok(Some(Box::new(PausedStage {
            gate: Some(gate),
            bytes: Vec::new(),
            published: self.published.clone(),
        })))
    }
    fn open(&self, _: &Path, _: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        Err(unused())
    }
    async fn read(&self, _: &Path) -> io::Result<Vec<u8>> {
        Err(unused())
    }
    async fn write(&self, _: &Path, _: &[u8]) -> io::Result<()> {
        Err(unused())
    }
    async fn remove_file(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unused())
    }
    async fn symlink_metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unused())
    }
    async fn read_dir(&self, _: &Path) -> io::Result<Vec<DirEntry>> {
        Err(unused())
    }
    async fn canonicalize(&self, _: &Path) -> io::Result<PathBuf> {
        Err(unused())
    }
    async fn create_dir(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn create_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn remove_dir(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn remove_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn rename(&self, _: &Path, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn set_readonly(&self, _: &Path, _: bool) -> io::Result<()> {
        Err(unused())
    }
}

#[test]
fn public_print_keeps_selected_snapshot_across_pending_stage_and_scene_clear() {
    let _plot = crate::builtins::plotting::tests::lock_plot_registry();
    crate::builtins::plotting::tests::ensure_plot_test_env();
    super::super::state::reset_hold_state_for_run();
    let _ = super::super::state::clear_figure(None);
    let _fs = fs::provider_override_lock();
    futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![
        Value::Tensor(runmat_value::Tensor::new(vec![0.0, 1.0, 2.0], vec![3]).unwrap()),
        Value::Tensor(runmat_value::Tensor::new(vec![1.0, 3.0, 2.0], vec![3]).unwrap()),
    ]))
    .expect("plot selected scene");
    let original = clone_figure(current_figure_handle()).expect("original scene");
    let expected =
        futures::executor::block_on(render_png(original, 800, 600)).expect("reference render");
    let (sender, receiver) = oneshot::channel();
    let published = Arc::new(Mutex::new(None));
    let provider = Arc::new(PausedProvider {
        gate: Mutex::new(Some(receiver)),
        published: published.clone(),
    });
    let _selected = fs::replace_provider(provider);
    let _statement = crate::output_count::push_output_count(Some(0));
    let mut call = Box::pin(print_builtin(None, vec![Value::from("/selected.png")]));
    let waker = noop_waker();
    assert!(matches!(
        call.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    super::super::state::clear_figure(None).expect("clear ambient scene during pending call");
    sender.send(()).expect("retained stage receiver");
    assert!(futures::executor::block_on(call).expect("public print completes"));
    assert_eq!(
        published.lock().unwrap().as_deref(),
        Some(expected.as_slice())
    );
}
