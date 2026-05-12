use std::fmt;
use std::io::{self, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use runmat_filesystem::FileHandle;

use super::bindings::JsFsFuncs;

pub(super) struct JsFileState {
    pub(super) funcs: JsFsFuncs,
    pub(super) path: PathBuf,
    pub(super) buffer: Vec<u8>,
    pub(super) cursor: usize,
    pub(super) can_read: bool,
    pub(super) can_write: bool,
    pub(super) append: bool,
    pub(super) dirty: bool,
}

impl JsFileState {
    fn write_back(&mut self) -> io::Result<()> {
        if self.can_write && self.dirty {
            self.dirty = false;
            let data = self.buffer.clone();
            let path = self.path.clone();
            self.funcs.write_file(&path, &data)?;
        }
        Ok(())
    }

    async fn write_back_async(inner: Arc<Mutex<Self>>) -> io::Result<()> {
        let Some((funcs, path, data)) = ({
            let mut state = inner.lock().unwrap();
            if state.can_write && state.dirty {
                state.dirty = false;
                Some((
                    state.funcs.clone(),
                    state.path.clone(),
                    state.buffer.clone(),
                ))
            } else {
                None
            }
        }) else {
            return Ok(());
        };

        funcs.write_file_async(&path, &data).await
    }
}

pub(super) struct JsFileHandle {
    #[allow(clippy::arc_with_non_send_sync)]
    pub(super) inner: Arc<Mutex<JsFileState>>,
}

unsafe impl Send for JsFileHandle {}
unsafe impl Sync for JsFileHandle {}

impl Read for JsFileHandle {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let mut state = self.inner.lock().unwrap();
        if !state.can_read {
            return Err(io::Error::new(
                ErrorKind::PermissionDenied,
                "File not opened for reading",
            ));
        }
        let remaining = state.buffer.len().saturating_sub(state.cursor);
        let to_read = remaining.min(buf.len());
        buf[..to_read].copy_from_slice(&state.buffer[state.cursor..state.cursor + to_read]);
        state.cursor += to_read;
        Ok(to_read)
    }
}

impl Write for JsFileHandle {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut state = self.inner.lock().unwrap();
        if !state.can_write {
            return Err(io::Error::new(
                ErrorKind::PermissionDenied,
                "File not opened for writing",
            ));
        }
        if state.append {
            state.cursor = state.buffer.len();
        }
        let cursor = state.cursor;
        if cursor > state.buffer.len() {
            state.buffer.resize(cursor, 0);
        }
        let end = cursor + buf.len();
        if end > state.buffer.len() {
            state.buffer.resize(end, 0);
        }
        state.buffer[cursor..end].copy_from_slice(buf);
        state.cursor = end;
        state.dirty = true;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        let mut state = self.inner.lock().unwrap();
        state.write_back()
    }
}

impl Seek for JsFileHandle {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let mut state = self.inner.lock().unwrap();
        let new_pos = match pos {
            SeekFrom::Start(offset) => offset as i64,
            SeekFrom::End(offset) => state.buffer.len() as i64 + offset,
            SeekFrom::Current(offset) => state.cursor as i64 + offset,
        };
        if new_pos < 0 {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "Seek before start of file",
            ));
        }
        state.cursor = new_pos as usize;
        Ok(state.cursor as u64)
    }
}

#[async_trait(?Send)]
impl FileHandle for JsFileHandle {
    async fn flush_async(&mut self) -> io::Result<()> {
        JsFileState::write_back_async(self.inner.clone()).await
    }
}

impl Drop for JsFileHandle {
    fn drop(&mut self) {
        if let Ok(mut state) = self.inner.lock() {
            let _ = state.write_back();
        }
    }
}

impl fmt::Debug for JsFileHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JsFileHandle").finish_non_exhaustive()
    }
}
