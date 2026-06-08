use js_sys::{Array, Function, Promise, Reflect};
use runmat_filesystem::{DirEntry, FsMetadata, ReadManyEntry};
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use super::convert::{
    js_error, js_value_to_bytes, map_js_error, parse_dir_entries, parse_metadata, path_to_string,
};

#[derive(Clone)]
pub(super) struct JsFsFuncs {
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
    pub(super) fn new(bindings: &JsValue) -> Result<Self, JsValue> {
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

    pub(super) fn read_file(&self, path: &Path) -> io::Result<Vec<u8>> {
        let func = self.require_fn(&self.read_file, "readFile")?;
        let js_path = JsValue::from(path_to_string(path));
        let result = func.call1(&self.bindings, &js_path);
        match result {
            Ok(value) => js_value_to_bytes(value, "readFile"),
            Err(err) => Err(map_js_error("readFile", err)),
        }
    }

    pub(super) async fn read_file_async(&self, path: &Path) -> io::Result<Vec<u8>> {
        let func = self.require_fn(&self.read_file, "readFile")?;
        let js_path = JsValue::from(path_to_string(path));
        let value = func
            .call1(&self.bindings, &js_path)
            .map_err(|err| map_js_error("readFile", err))?;
        let resolved = resolve_maybe_promise(value, "readFile").await?;
        js_value_to_bytes(resolved, "readFile")
    }

    pub(super) fn write_file(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        let func = self.require_fn(&self.write_file, "writeFile")?;
        let js_path = JsValue::from(path_to_string(path));
        let array = js_sys::Uint8Array::new_with_length(data.len() as u32);
        array.copy_from(data);
        func.call2(&self.bindings, &js_path, &array.into())
            .map_err(|err| map_js_error("writeFile", err))?;
        Ok(())
    }

    pub(super) async fn write_file_async(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        let func = self.require_fn(&self.write_file, "writeFile")?;
        let js_path = JsValue::from(path_to_string(path));
        let array = js_sys::Uint8Array::new_with_length(data.len() as u32);
        array.copy_from(data);
        let value = func
            .call2(&self.bindings, &js_path, &array.into())
            .map_err(|err| map_js_error("writeFile", err))?;
        let _ = resolve_maybe_promise(value, "writeFile").await?;
        Ok(())
    }

    pub(super) async fn read_many(&self, paths: &[PathBuf]) -> io::Result<Vec<ReadManyEntry>> {
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

    pub(super) async fn remove_file_async(&self, path: &Path) -> io::Result<()> {
        let func = self.require_fn(&self.remove_file, "removeFile")?;
        let js_path = JsValue::from(path_to_string(path));
        let value = func
            .call1(&self.bindings, &js_path)
            .map_err(|err| map_js_error("removeFile", err))?;
        let _ = resolve_maybe_promise(value, "removeFile").await?;
        Ok(())
    }

    pub(super) async fn metadata_async(&self, path: &Path) -> io::Result<FsMetadata> {
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

    pub(super) async fn symlink_metadata_async(&self, path: &Path) -> io::Result<FsMetadata> {
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

    pub(super) async fn read_dir_async(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        let func = self.require_fn(&self.read_dir, "readDir")?;
        let js_path = JsValue::from(path_to_string(path));
        let value = func
            .call1(&self.bindings, &js_path)
            .map_err(|err| map_js_error("readDir", err))?;
        let value = resolve_maybe_promise(value, "readDir").await?;
        parse_dir_entries(value)
    }

    pub(super) async fn canonicalize_async(&self, path: &Path) -> io::Result<PathBuf> {
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

    pub(super) async fn create_dir_async(&self, path: &Path) -> io::Result<()> {
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

    pub(super) async fn create_dir_all_async(&self, path: &Path) -> io::Result<()> {
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

    pub(super) async fn remove_dir_async(&self, path: &Path) -> io::Result<()> {
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

    pub(super) async fn remove_dir_all_async(&self, path: &Path) -> io::Result<()> {
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

    pub(super) async fn rename_async(&self, from: &Path, to: &Path) -> io::Result<()> {
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

    pub(super) async fn set_readonly_async(&self, path: &Path, readonly: bool) -> io::Result<()> {
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

fn get_fn(obj: &JsValue, name: &str) -> Result<Option<Function>, JsValue> {
    let value = Reflect::get(obj, &JsValue::from_str(name))?;
    if value.is_undefined() || value.is_null() {
        Ok(None)
    } else if value.is_function() {
        Ok(Some(value.unchecked_into()))
    } else {
        Err(js_error(&format!(
            "fsProvider.{name} must be a function if provided",
        )))
    }
}
