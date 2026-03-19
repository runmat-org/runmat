use async_trait::async_trait;
use js_sys::{Array, ArrayBuffer, Function, Promise, Reflect, Uint8Array};
use runmat_filesystem::{
    DirEntry, FileHandle, FsFileType, FsMetadata, FsProvider, OpenFlags, ReadManyEntry,
};
use std::ffi::OsString;
use std::fmt;
use std::io::{self, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

#[derive(Clone)]
struct JsFsFuncs {
    bindings: JsValue,
    read_file: Option<Function>,
    read_many: Option<Function>,
    write_file: Option<Function>,
    remove_file: Option<Function>,
    metadata: Option<Function>,
    symlink_metadata: Option<Function>,
    read_dir: Option<Function>,
    canonicalize: Option<Function>,
    create_dir: Option<Function>,
    create_dir_all: Option<Function>,
    remove_dir: Option<Function>,
    remove_dir_all: Option<Function>,
    rename: Option<Function>,
    set_readonly: Option<Function>,
}

impl JsFsFuncs {
    fn new(bindings: &JsValue) -> Result<Self, JsValue> {
        if !bindings.is_object() {
            return Err(js_error("fsProvider must be an object"));
        }
        Ok(Self {
            bindings: bindings.clone(),
            read_file: get_fn(bindings, "readFile")?,
            read_many: get_fn(bindings, "readMany")?,
            write_file: get_fn(bindings, "writeFile")?,
            remove_file: get_fn(bindings, "removeFile")?,
            metadata: get_fn(bindings, "metadata")?,
            symlink_metadata: get_fn(bindings, "symlinkMetadata")?,
            read_dir: get_fn(bindings, "readDir")?,
            canonicalize: get_fn(bindings, "canonicalize")?,
            create_dir: get_fn(bindings, "createDir")?,
            create_dir_all: get_fn(bindings, "createDirAll")?,
            remove_dir: get_fn(bindings, "removeDir")?,
            remove_dir_all: get_fn(bindings, "removeDirAll")?,
            rename: get_fn(bindings, "rename")?,
            set_readonly: get_fn(bindings, "setReadonly")?,
        })
    }

    fn read_file(&self, path: &Path) -> io::Result<Vec<u8>> {
        let func = self.require_fn(&self.read_file, "readFile")?;
        let js_path = JsValue::from(path_to_string(path));
        let result = func.call1(&self.bindings, &js_path);
        match result {
            Ok(value) => js_value_to_bytes(value, "readFile"),
            Err(err) => Err(map_js_error("readFile", err)),
        }
    }

    async fn read_file_async(&self, path: &Path) -> io::Result<Vec<u8>> {
        let func = self.require_fn(&self.read_file, "readFile")?;
        let js_path = JsValue::from(path_to_string(path));
        let value = func
            .call1(&self.bindings, &js_path)
            .map_err(|err| map_js_error("readFile", err))?;
        let resolved = resolve_maybe_promise(value, "readFile").await?;
        js_value_to_bytes(resolved, "readFile")
    }

    fn write_file(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        let func = self.require_fn(&self.write_file, "writeFile")?;
        let js_path = JsValue::from(path_to_string(path));
        let array = Uint8Array::new_with_length(data.len() as u32);
        array.copy_from(data);
        func.call2(&self.bindings, &js_path, &array.into())
            .map_err(|err| map_js_error("writeFile", err))?;
        Ok(())
    }

    async fn write_file_async(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        let func = self.require_fn(&self.write_file, "writeFile")?;
        let js_path = JsValue::from(path_to_string(path));
        let array = Uint8Array::new_with_length(data.len() as u32);
        array.copy_from(data);
        let value = func
            .call2(&self.bindings, &js_path, &array.into())
            .map_err(|err| map_js_error("writeFile", err))?;
        let _ = resolve_maybe_promise(value, "writeFile").await?;
        Ok(())
    }

    async fn read_many(&self, paths: &[PathBuf]) -> io::Result<Vec<ReadManyEntry>> {
        let Some(func) = &self.read_many else {
            let mut entries = Vec::with_capacity(paths.len());
            for path in paths {
                let bytes = self.read_file_async(path).await.ok();
                entries.push(ReadManyEntry::new(path.clone(), bytes));
            }
            return Ok(entries);
        };
        let js_paths = Array::new();
        for path in paths {
            js_paths.push(&JsValue::from(path_to_string(path)));
        }
        let value = func
            .call1(&self.bindings, &js_paths.into())
            .map_err(|err| map_js_error("readMany", err))?;
        let value = resolve_maybe_promise(value, "readMany").await?;
        if !Array::is_array(&value) {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "readMany must return an array",
            ));
        }
        let items = Array::from(&value);
        let mut out = Vec::with_capacity(paths.len());
        for (index, path) in paths.iter().enumerate() {
            let item = items.get(index as u32);
            if item.is_null() || item.is_undefined() {
                out.push(ReadManyEntry::new(path.clone(), None));
            } else {
                let bytes = js_value_to_bytes(item, "readMany")?;
                out.push(ReadManyEntry::new(path.clone(), Some(bytes)));
            }
        }
        Ok(out)
    }

    async fn remove_file_async(&self, path: &Path) -> io::Result<()> {
        let func = self.require_fn(&self.remove_file, "removeFile")?;
        let js_path = JsValue::from(path_to_string(path));
        let value = func
            .call1(&self.bindings, &js_path)
            .map_err(|err| map_js_error("removeFile", err))?;
        let _ = resolve_maybe_promise(value, "removeFile").await?;
        Ok(())
    }

    async fn metadata_async(&self, path: &Path) -> io::Result<FsMetadata> {
        let func = self.require_fn(&self.metadata, "metadata")?;
        let js_path = JsValue::from(path_to_string(path));
        let value = func
            .call1(&self.bindings, &js_path)
            .map_err(|err| map_js_error("metadata", err))?;
        let value = resolve_maybe_promise(value, "metadata").await?;
        parse_metadata(value).ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidData,
                "fsProvider.metadata returned invalid payload",
            )
        })
    }

    async fn symlink_metadata_async(&self, path: &Path) -> io::Result<FsMetadata> {
        if let Some(func) = &self.symlink_metadata {
            let js_path = JsValue::from(path_to_string(path));
            let value = func
                .call1(&self.bindings, &js_path)
                .map_err(|err| map_js_error("symlinkMetadata", err))?;
            let value = resolve_maybe_promise(value, "symlinkMetadata").await?;
            parse_metadata(value).ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidData,
                    "fsProvider.symlinkMetadata returned invalid payload",
                )
            })
        } else {
            self.metadata_async(path).await
        }
    }

    async fn read_dir_async(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        let func = self.require_fn(&self.read_dir, "readDir")?;
        let js_path = JsValue::from(path_to_string(path));
        let value = func
            .call1(&self.bindings, &js_path)
            .map_err(|err| map_js_error("readDir", err))?;
        let value = resolve_maybe_promise(value, "readDir").await?;
        parse_dir_entries(value)
    }

    async fn canonicalize_async(&self, path: &Path) -> io::Result<PathBuf> {
        if let Some(func) = &self.canonicalize {
            let js_path = JsValue::from(path_to_string(path));
            let value = func
                .call1(&self.bindings, &js_path)
                .map_err(|err| map_js_error("canonicalize", err))?;
            let value = resolve_maybe_promise(value, "canonicalize").await?;
            value.as_string().map(PathBuf::from).ok_or_else(|| {
                io::Error::new(ErrorKind::InvalidData, "canonicalize must return a string")
            })
        } else {
            Ok(path.to_path_buf())
        }
    }

    async fn create_dir_async(&self, path: &Path) -> io::Result<()> {
        if let Some(func) = &self.create_dir {
            let js_path = JsValue::from(path_to_string(path));
            let value = func
                .call1(&self.bindings, &js_path)
                .map_err(|err| map_js_error("createDir", err))?;
            let _ = resolve_maybe_promise(value, "createDir").await?;
            Ok(())
        } else {
            Err(io::Error::new(
                ErrorKind::Unsupported,
                "fsProvider.createDir not implemented",
            ))
        }
    }

    async fn create_dir_all_async(&self, path: &Path) -> io::Result<()> {
        if let Some(func) = &self.create_dir_all {
            let js_path = JsValue::from(path_to_string(path));
            let value = func
                .call1(&self.bindings, &js_path)
                .map_err(|err| map_js_error("createDirAll", err))?;
            let _ = resolve_maybe_promise(value, "createDirAll").await?;
            Ok(())
        } else {
            self.create_dir_async(path).await
        }
    }

    async fn remove_dir_async(&self, path: &Path) -> io::Result<()> {
        if let Some(func) = &self.remove_dir {
            let js_path = JsValue::from(path_to_string(path));
            let value = func
                .call1(&self.bindings, &js_path)
                .map_err(|err| map_js_error("removeDir", err))?;
            let _ = resolve_maybe_promise(value, "removeDir").await?;
            Ok(())
        } else {
            Err(io::Error::new(
                ErrorKind::Unsupported,
                "fsProvider.removeDir not implemented",
            ))
        }
    }

    async fn remove_dir_all_async(&self, path: &Path) -> io::Result<()> {
        if let Some(func) = &self.remove_dir_all {
            let js_path = JsValue::from(path_to_string(path));
            let value = func
                .call1(&self.bindings, &js_path)
                .map_err(|err| map_js_error("removeDirAll", err))?;
            let _ = resolve_maybe_promise(value, "removeDirAll").await?;
            Ok(())
        } else {
            self.remove_dir_async(path).await
        }
    }

    async fn rename_async(&self, from: &Path, to: &Path) -> io::Result<()> {
        if let Some(func) = &self.rename {
            let js_from = JsValue::from(path_to_string(from));
            let js_to = JsValue::from(path_to_string(to));
            let value = func
                .call2(&self.bindings, &js_from, &js_to)
                .map_err(|err| map_js_error("rename", err))?;
            let _ = resolve_maybe_promise(value, "rename").await?;
            Ok(())
        } else {
            Err(io::Error::new(
                ErrorKind::Unsupported,
                "fsProvider.rename not implemented",
            ))
        }
    }

    async fn set_readonly_async(&self, path: &Path, readonly: bool) -> io::Result<()> {
        if let Some(func) = &self.set_readonly {
            let js_path = JsValue::from(path_to_string(path));
            let js_flag = JsValue::from(readonly);
            let value = func
                .call2(&self.bindings, &js_path, &js_flag)
                .map_err(|err| map_js_error("setReadonly", err))?;
            let _ = resolve_maybe_promise(value, "setReadonly").await?;
            Ok(())
        } else {
            Err(io::Error::new(
                ErrorKind::Unsupported,
                "fsProvider.setReadonly not implemented",
            ))
        }
    }

    fn require_fn<'a>(
        &'a self,
        func: &'a Option<Function>,
        name: &'static str,
    ) -> io::Result<&'a Function> {
        func.as_ref().ok_or_else(|| {
            io::Error::new(
                ErrorKind::Unsupported,
                format!("fsProvider.{name} is not implemented"),
            )
        })
    }
}

pub(crate) fn install_js_fs_provider(bindings: &JsValue) -> Result<(), JsValue> {
    let funcs = JsFsFuncs::new(bindings)?;
    let provider: Arc<dyn FsProvider> = Arc::new(JsFsProvider { funcs });
    runmat_filesystem::set_provider(provider);
    Ok(())
}

pub(crate) struct JsFsProvider {
    funcs: JsFsFuncs,
}

unsafe impl Send for JsFsProvider {}
unsafe impl Sync for JsFsProvider {}

#[async_trait(?Send)]
impl FsProvider for JsFsProvider {
    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        let mut initial = Vec::new();
        let mut exists = false;
        let mut dirty = false;

        if flags.read || (!flags.create && !flags.create_new) || flags.append {
            match self.funcs.read_file(path) {
                Ok(bytes) => {
                    initial = bytes;
                    exists = true;
                }
                Err(err) if err.kind() == ErrorKind::NotFound => {
                    exists = false;
                }
                Err(err) => return Err(err),
            }
        }

        if flags.create_new && exists {
            return Err(io::Error::new(
                ErrorKind::AlreadyExists,
                format!("File already exists: {}", path.display()),
            ));
        }

        if flags.create && !exists {
            exists = true;
            dirty = true;
        }

        if !exists && !flags.create {
            return Err(io::Error::new(
                ErrorKind::NotFound,
                format!("File not found: {}", path.display()),
            ));
        }

        if flags.truncate {
            initial.clear();
            dirty = true;
        }

        let cursor = if flags.append { initial.len() } else { 0 };

        let inner = JsFileState {
            funcs: self.funcs.clone(),
            path: path.to_path_buf(),
            buffer: initial,
            cursor,
            can_read: flags.read,
            can_write: flags.write || flags.append,
            append: flags.append,
            dirty,
        };
        #[allow(clippy::arc_with_non_send_sync)]
        let inner = Arc::new(Mutex::new(inner));
        Ok(Box::new(JsFileHandle { inner }))
    }

    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.funcs.read_file_async(path).await
    }

    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        self.funcs.write_file_async(path, data).await
    }

    async fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.funcs.remove_file_async(path).await
    }

    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.funcs.metadata_async(path).await
    }

    async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.funcs.symlink_metadata_async(path).await
    }

    async fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        self.funcs.read_dir_async(path).await
    }

    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        self.funcs.canonicalize_async(path).await
    }

    async fn create_dir(&self, path: &Path) -> io::Result<()> {
        self.funcs.create_dir_async(path).await
    }

    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.funcs.create_dir_all_async(path).await
    }

    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        self.funcs.remove_dir_async(path).await
    }

    async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        self.funcs.remove_dir_all_async(path).await
    }

    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.funcs.rename_async(from, to).await
    }

    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        self.funcs.set_readonly_async(path, readonly).await
    }

    async fn read_many(&self, paths: &[PathBuf]) -> io::Result<Vec<ReadManyEntry>> {
        self.funcs.read_many(paths).await
    }
}

async fn resolve_maybe_promise(value: JsValue, op: &'static str) -> io::Result<JsValue> {
    if value.is_instance_of::<Promise>() {
        let promise: Promise = value.unchecked_into();
        JsFuture::from(promise)
            .await
            .map_err(|err| map_js_error(op, err))
    } else {
        Ok(value)
    }
}

struct JsFileState {
    funcs: JsFsFuncs,
    path: PathBuf,
    buffer: Vec<u8>,
    cursor: usize,
    can_read: bool,
    can_write: bool,
    append: bool,
    dirty: bool,
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
}

struct JsFileHandle {
    #[allow(clippy::arc_with_non_send_sync)]
    inner: Arc<Mutex<JsFileState>>,
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

fn get_fn(obj: &JsValue, name: &str) -> Result<Option<Function>, JsValue> {
    let value = Reflect::get(obj, &JsValue::from_str(name))?;
    if value.is_undefined() || value.is_null() {
        Ok(None)
    } else if value.is_function() {
        Ok(Some(value.unchecked_into()))
    } else {
        Err(js_error(&format!(
            "fsProvider.{} must be a function if provided",
            name
        )))
    }
}

fn js_error(msg: &str) -> JsValue {
    JsValue::from_str(msg)
}

fn map_js_error(op: &str, err: JsValue) -> io::Error {
    if is_not_found_error(&err) {
        return io::Error::new(ErrorKind::NotFound, format!("{op}: not found"));
    }
    let message = err.as_string().unwrap_or_else(|| format!("{:?}", err));
    io::Error::other(format!("{op}: {message}"))
}

fn is_not_found_error(err: &JsValue) -> bool {
    if let Some(text) = err.as_string() {
        if text.to_ascii_lowercase().contains("notfound") {
            return true;
        }
    }
    if err.is_object() {
        if let Ok(code) = Reflect::get(err, &JsValue::from_str("code")) {
            if let Some(text) = code.as_string() {
                if text.eq_ignore_ascii_case("notfound") {
                    return true;
                }
            }
        }
        if let Ok(name) = Reflect::get(err, &JsValue::from_str("name")) {
            if let Some(text) = name.as_string() {
                if text.eq_ignore_ascii_case("notfounderror") {
                    return true;
                }
            }
        }
    }
    false
}

fn js_value_to_bytes(value: JsValue, op: &str) -> io::Result<Vec<u8>> {
    if let Some(array) = value.dyn_ref::<Uint8Array>() {
        let mut buffer = vec![0u8; array.length() as usize];
        array.copy_to(&mut buffer);
        Ok(buffer)
    } else if let Some(buffer) = value.dyn_ref::<ArrayBuffer>() {
        let view = Uint8Array::new(buffer);
        let mut data = vec![0u8; view.length() as usize];
        view.copy_to(&mut data);
        Ok(data)
    } else {
        Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("{op} must return a Uint8Array or ArrayBuffer"),
        ))
    }
}

fn parse_metadata(value: JsValue) -> Option<FsMetadata> {
    if !value.is_object() {
        return None;
    }
    let file_type = Reflect::get(&value, &JsValue::from_str("fileType"))
        .ok()
        .and_then(|v| v.as_string())
        .and_then(map_file_type)?;
    let len = Reflect::get(&value, &JsValue::from_str("len"))
        .ok()?
        .as_f64()? as u64;
    let modified = Reflect::get(&value, &JsValue::from_str("modified"))
        .ok()
        .and_then(|v| v.as_f64())
        .map(|millis| {
            let secs = (millis / 1000.0) as u64;
            let nanos = ((millis % 1000.0) * 1_000_000.0) as u32;
            std::time::UNIX_EPOCH + std::time::Duration::new(secs, nanos)
        });
    let readonly = Reflect::get(&value, &JsValue::from_str("readonly"))
        .ok()
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    Some(FsMetadata::new(file_type, len, modified, readonly))
}

fn parse_dir_entries(value: JsValue) -> io::Result<Vec<DirEntry>> {
    if !value.is_object() {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "readDir must return an array",
        ));
    }
    let array = Array::from(&value);
    let mut entries = Vec::with_capacity(array.length() as usize);
    for item in array.iter() {
        if !item.is_object() {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "Invalid directory entry",
            ));
        }
        let path = Reflect::get(&item, &JsValue::from_str("path"))
            .ok()
            .and_then(|v| v.as_string())
            .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "dir entry missing path"))?;
        let file_name = Reflect::get(&item, &JsValue::from_str("fileName"))
            .ok()
            .and_then(|v| v.as_string())
            .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "dir entry missing fileName"))?;
        let file_type = Reflect::get(&item, &JsValue::from_str("fileType"))
            .ok()
            .and_then(|v| v.as_string())
            .and_then(map_file_type)
            .unwrap_or(FsFileType::Unknown);
        entries.push(DirEntry::new(
            PathBuf::from(path),
            OsString::from(file_name),
            file_type,
        ));
    }
    Ok(entries)
}

fn map_file_type(text: String) -> Option<FsFileType> {
    match text.as_str() {
        "file" => Some(FsFileType::File),
        "dir" | "directory" => Some(FsFileType::Directory),
        "symlink" => Some(FsFileType::Symlink),
        "other" => Some(FsFileType::Other),
        _ => Some(FsFileType::Unknown),
    }
}

fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}
