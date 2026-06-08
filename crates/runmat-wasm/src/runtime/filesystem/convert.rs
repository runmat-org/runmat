use js_sys::{Array, ArrayBuffer, Uint8Array};
use runmat_filesystem::{DirEntry, FsFileType, FsMetadata};
use std::ffi::OsString;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

pub(super) fn js_error(msg: &str) -> JsValue {
    JsValue::from_str(msg)
}

pub(super) fn map_js_error(op: &str, err: JsValue) -> io::Error {
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
        if let Ok(code) = js_sys::Reflect::get(err, &JsValue::from_str("code")) {
            if let Some(text) = code.as_string() {
                if text.eq_ignore_ascii_case("notfound") {
                    return true;
                }
            }
        }
        if let Ok(name) = js_sys::Reflect::get(err, &JsValue::from_str("name")) {
            if let Some(text) = name.as_string() {
                if text.eq_ignore_ascii_case("notfounderror") {
                    return true;
                }
            }
        }
    }
    false
}

pub(super) fn js_value_to_bytes(value: JsValue, op: &str) -> io::Result<Vec<u8>> {
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

pub(super) fn parse_metadata(value: JsValue) -> Option<FsMetadata> {
    if !value.is_object() {
        return None;
    }
    let file_type = js_sys::Reflect::get(&value, &JsValue::from_str("fileType"))
        .ok()
        .and_then(|v| v.as_string())
        .and_then(map_file_type)?;
    let len = js_sys::Reflect::get(&value, &JsValue::from_str("len"))
        .ok()?
        .as_f64()? as u64;
    let modified = js_sys::Reflect::get(&value, &JsValue::from_str("modified"))
        .ok()
        .and_then(|v| v.as_f64())
        .map(|millis| {
            let secs = (millis / 1000.0) as u64;
            let nanos = ((millis % 1000.0) * 1_000_000.0) as u32;
            std::time::UNIX_EPOCH + std::time::Duration::new(secs, nanos)
        });
    let readonly = js_sys::Reflect::get(&value, &JsValue::from_str("readonly"))
        .ok()
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    Some(FsMetadata::new(file_type, len, modified, readonly))
}

pub(super) fn parse_dir_entries(value: JsValue) -> io::Result<Vec<DirEntry>> {
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
        let path = js_sys::Reflect::get(&item, &JsValue::from_str("path"))
            .ok()
            .and_then(|v| v.as_string())
            .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "dir entry missing path"))?;
        let file_name = js_sys::Reflect::get(&item, &JsValue::from_str("fileName"))
            .ok()
            .and_then(|v| v.as_string())
            .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "dir entry missing fileName"))?;
        let file_type = js_sys::Reflect::get(&item, &JsValue::from_str("fileType"))
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

pub(super) fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}
