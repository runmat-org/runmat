use std::cell::RefCell;
use std::fmt;
use std::io::{self, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::rc::Rc;

use crate::FileHandle;
use async_trait::async_trait;

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
            let data = self.buffer.clone();
            let path = self.path.clone();
            self.funcs.write_file(&path, &data)?;
            self.dirty = false;
        }
        Ok(())
    }

    async fn write_back_async(inner: Rc<RefCell<Self>>) -> io::Result<()> {
        let Some((funcs, path, data)) = ({
            let mut state = inner.try_borrow_mut().map_err(state_busy)?;
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

        match funcs.write_file_async(&path, &data).await {
            Ok(()) => Ok(()),
            Err(err) => {
                if let Ok(mut state) = inner.try_borrow_mut() {
                    state.dirty = true;
                }
                Err(err)
            }
        }
    }
}

pub(super) struct JsFileHandle {
    pub(super) inner: Rc<RefCell<JsFileState>>,
}

// SAFETY: the enclosing JS adapter rejects shared-memory/atomics builds. All
// handles, their local shared state and callbacks stay on the same JS agent.
unsafe impl Send for JsFileHandle {}
unsafe impl Sync for JsFileHandle {}

impl Read for JsFileHandle {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let mut state = self.inner.try_borrow_mut().map_err(state_busy)?;
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
        let mut state = self.inner.try_borrow_mut().map_err(state_busy)?;
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
        let mut state = self.inner.try_borrow_mut().map_err(state_busy)?;
        state.write_back()
    }
}

impl Seek for JsFileHandle {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let mut state = self.inner.try_borrow_mut().map_err(state_busy)?;
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
        if let Ok(mut state) = self.inner.try_borrow_mut() {
            let _ = state.write_back();
        }
    }
}

impl fmt::Debug for JsFileHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JsFileHandle").finish_non_exhaustive()
    }
}
fn state_busy(error: std::cell::BorrowMutError) -> io::Error {
    io::Error::new(ErrorKind::WouldBlock, error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[wasm_bindgen_test::wasm_bindgen_test]
    fn reentrant_handle_access_reports_a_busy_state() {
        let bindings = js_sys::Object::new();
        let inner = Rc::new(RefCell::new(JsFileState {
            funcs: JsFsFuncs::new(&bindings.into()).unwrap(),
            path: PathBuf::from("/file"),
            buffer: vec![42],
            cursor: 0,
            can_read: true,
            can_write: true,
            append: false,
            dirty: false,
        }));
        let mut handle = JsFileHandle {
            inner: inner.clone(),
        };
        let borrowed = inner.borrow_mut();
        assert_eq!(
            handle.read(&mut [0]).unwrap_err().kind(),
            ErrorKind::WouldBlock
        );
        assert_eq!(
            handle.write(&[1]).unwrap_err().kind(),
            ErrorKind::WouldBlock
        );
        assert_eq!(
            handle.seek(SeekFrom::Start(0)).unwrap_err().kind(),
            ErrorKind::WouldBlock
        );
        assert_eq!(handle.flush().unwrap_err().kind(), ErrorKind::WouldBlock);
        drop(borrowed);
        let mut byte = [0];
        assert_eq!(handle.read(&mut byte).unwrap(), 1);
        assert_eq!(byte, [42]);
    }
}
