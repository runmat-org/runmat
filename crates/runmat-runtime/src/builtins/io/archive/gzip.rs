//! MATLAB-compatible `gzip` and `gunzip` file helpers.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use glob::glob;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, Value};
use url::Url;

use crate::builtins::common::fs::{expand_user_path, path_to_string};
use crate::builtins::io::http::transport::{self, HttpMethod, HttpRequest};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

fn archive_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

const INPUTS_ARCHIVE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input file.",
    },
    BuiltinParamDescriptor {
        name: "outputfolder",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Destination folder.",
    },
];
const OUTPUT_FILES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "files",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell array of output file paths.",
}];
const GZIP_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "files = gzip(filename, outputfolder)",
    inputs: &INPUTS_ARCHIVE,
    outputs: &OUTPUT_FILES,
}];
pub const GZIP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GZIP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
const GUNZIP_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "files = gunzip(filename, outputfolder)",
    inputs: &INPUTS_ARCHIVE,
    outputs: &OUTPUT_FILES,
}];
pub const GUNZIP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GUNZIP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

async fn gather_args(name: &str, args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(
            gather_if_needed_async(value)
                .await
                .map_err(|err| archive_error(name, format!("{name}: {}", err.message())))?,
        );
    }
    Ok(out)
}

fn scalar_text(value: &Value, name: &str, arg: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        _ => Err(archive_error(
            name,
            format!("{name}: {arg} must be a string scalar or character vector"),
        )),
    }
}

fn path_from_value(value: &Value, name: &str, arg: &str) -> BuiltinResult<PathBuf> {
    let text = scalar_text(value, name, arg)?;
    Ok(PathBuf::from(
        expand_user_path(text.trim(), name).map_err(|err| archive_error(name, err))?,
    ))
}

fn paths_from_value(value: &Value, name: &str, arg: &str) -> BuiltinResult<Vec<PathBuf>> {
    match value {
        Value::StringArray(array) => array
            .data
            .iter()
            .map(|text| path_from_text(text, name))
            .collect(),
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|value| path_from_value(value, name, arg))
            .collect(),
        _ => Ok(vec![path_from_value(value, name, arg)?]),
    }
}

fn path_from_text(text: &str, name: &str) -> BuiltinResult<PathBuf> {
    Ok(PathBuf::from(
        expand_user_path(text.trim(), name).map_err(|err| archive_error(name, err))?,
    ))
}

fn expand_input_paths(paths: Vec<PathBuf>, name: &str) -> BuiltinResult<Vec<PathBuf>> {
    let mut out = Vec::new();
    for path in paths {
        let text = path_to_string(&path);
        if text.contains('*') || text.contains('?') {
            for entry in glob(&text).map_err(|err| archive_error(name, err.to_string()))? {
                out.push(entry.map_err(|err| archive_error(name, err.to_string()))?);
            }
            continue;
        }
        if vfs::metadata(&path).is_ok_and(|meta| meta.is_dir()) {
            for entry in
                vfs::read_dir(&path).map_err(|err| archive_error(name, format!("{name}: {err}")))?
            {
                if !entry.is_dir() {
                    out.push(entry.path().to_path_buf());
                }
            }
        } else {
            out.push(path);
        }
    }
    Ok(out)
}

fn char_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

fn cellstr(paths: Vec<PathBuf>, name: &str) -> BuiltinResult<Value> {
    let len = paths.len();
    Ok(Value::Cell(
        CellArray::new(
            paths
                .iter()
                .map(|p| char_value(&path_to_string(p)))
                .collect(),
            len,
            1,
        )
        .map_err(|err| archive_error(name, err))?,
    ))
}

#[runtime_builtin(
    name = "gzip",
    category = "io/archive",
    summary = "Compress files using gzip.",
    keywords = "gzip,compress,archive,file",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::unzip_type),
    descriptor(crate::builtins::io::archive::gzip::GZIP_DESCRIPTOR),
    builtin_path = "crate::builtins::io::archive::gzip"
)]
async fn gzip_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_args("gzip", &args).await?;
    if args.is_empty() || args.len() > 2 {
        return Err(archive_error(
            "gzip",
            "gzip: filename and optional output folder are expected",
        ));
    }
    let inputs = expand_input_paths(paths_from_value(&args[0], "gzip", "filename")?, "gzip")?;
    let output_folder = if let Some(value) = args.get(1) {
        Some(path_from_value(value, "gzip", "outputfolder")?)
    } else {
        None
    };
    let mut outputs = Vec::with_capacity(inputs.len());
    for input in inputs {
        outputs.push(gzip_one(&input, output_folder.as_deref()).await?);
    }
    cellstr(outputs, "gzip")
}

async fn gzip_one(input: &Path, output_folder: Option<&Path>) -> BuiltinResult<PathBuf> {
    let bytes = vfs::read_async(input)
        .await
        .map_err(|err| archive_error("gzip", format!("gzip: {err}")))?;
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(&bytes)
        .map_err(|err| archive_error("gzip", format!("gzip: {err}")))?;
    let compressed = encoder
        .finish()
        .map_err(|err| archive_error("gzip", format!("gzip: {err}")))?;
    let filename = input
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| archive_error("gzip", "gzip: invalid filename"))?;
    let output = output_folder
        .map(|folder| folder.join(format!("{filename}.gz")))
        .unwrap_or_else(|| input.with_file_name(format!("{filename}.gz")));
    if let Some(parent) = output.parent() {
        vfs::create_dir_all_async(parent)
            .await
            .map_err(|err| archive_error("gzip", format!("gzip: {err}")))?;
    }
    vfs::write_async(&output, compressed)
        .await
        .map_err(|err| archive_error("gzip", format!("gzip: {err}")))?;
    Ok(output)
}

#[runtime_builtin(
    name = "gunzip",
    category = "io/archive",
    summary = "Decompress gzip files.",
    keywords = "gunzip,decompress,archive,file",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::unzip_type),
    descriptor(crate::builtins::io::archive::gzip::GUNZIP_DESCRIPTOR),
    builtin_path = "crate::builtins::io::archive::gzip"
)]
async fn gunzip_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_args("gunzip", &args).await?;
    if args.is_empty() || args.len() > 2 {
        return Err(archive_error(
            "gunzip",
            "gunzip: filename and optional output folder are expected",
        ));
    }
    let inputs = gunzip_sources_from_value(&args[0])?;
    let output_folder = if let Some(value) = args.get(1) {
        Some(path_from_value(value, "gunzip", "outputfolder")?)
    } else {
        None
    };
    let mut outputs = Vec::with_capacity(inputs.len());
    for input in inputs {
        outputs.push(extract_gunzip_source(input, output_folder.as_deref()).await?);
    }
    cellstr(outputs, "gunzip")
}

enum GunzipSource {
    Path(PathBuf),
    Url(Url),
}

fn gunzip_sources_from_value(value: &Value) -> BuiltinResult<Vec<GunzipSource>> {
    match value {
        Value::StringArray(array) => array
            .data
            .iter()
            .map(|text| parse_gunzip_source(text))
            .collect(),
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|value| {
                scalar_text(value, "gunzip", "filename").and_then(|text| parse_gunzip_source(&text))
            })
            .collect(),
        _ => {
            let text = scalar_text(value, "gunzip", "filename")?;
            Ok(vec![parse_gunzip_source(&text)?])
        }
    }
}

fn parse_gunzip_source(text: &str) -> BuiltinResult<GunzipSource> {
    if let Ok(url) = Url::parse(text) {
        if matches!(url.scheme(), "http" | "https") {
            return Ok(GunzipSource::Url(url));
        }
    }
    Ok(GunzipSource::Path(path_from_text(text, "gunzip")?))
}

async fn extract_gunzip_source(
    source: GunzipSource,
    output_folder: Option<&Path>,
) -> BuiltinResult<PathBuf> {
    match source {
        GunzipSource::Path(path) => gunzip_one(&path, output_folder).await,
        GunzipSource::Url(url) => {
            let response = transport::send_request(&HttpRequest {
                url: url.clone(),
                method: HttpMethod::Get,
                headers: Vec::new(),
                body: None,
                timeout: DEFAULT_TIMEOUT,
                user_agent: "RunMat gunzip/0.0".to_string(),
            })
            .map_err(|err| archive_error("gunzip", err.message_with_prefix("gunzip")))?;
            let name = url
                .path_segments()
                .and_then(|mut segments| segments.next_back())
                .filter(|name| !name.is_empty())
                .unwrap_or("download.gz");
            gunzip_bytes(&response.body, name, output_folder).await
        }
    }
}

async fn gunzip_one(input: &Path, output_folder: Option<&Path>) -> BuiltinResult<PathBuf> {
    let bytes = vfs::read_async(input)
        .await
        .map_err(|err| archive_error("gunzip", format!("gunzip: {err}")))?;
    let filename = input
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| archive_error("gunzip", "gunzip: invalid filename"))?;
    gunzip_bytes(&bytes, filename, output_folder.or_else(|| input.parent())).await
}

async fn gunzip_bytes(
    bytes: &[u8],
    filename: &str,
    output_folder: Option<&Path>,
) -> BuiltinResult<PathBuf> {
    let mut decoder = GzDecoder::new(bytes);
    let mut decoded = Vec::new();
    decoder
        .read_to_end(&mut decoded)
        .map_err(|err| archive_error("gunzip", format!("gunzip: {err}")))?;
    let output_name = filename.strip_suffix(".gz").unwrap_or(filename);
    let output = output_folder
        .map(|folder| folder.join(output_name))
        .unwrap_or_else(|| PathBuf::from(output_name));
    if let Some(parent) = output.parent() {
        vfs::create_dir_all_async(parent)
            .await
            .map_err(|err| archive_error("gunzip", format!("gunzip: {err}")))?;
    }
    vfs::write_async(&output, decoded)
        .await
        .map_err(|err| archive_error("gunzip", format!("gunzip: {err}")))?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;

    fn run(value: impl std::future::Future<Output = BuiltinResult<Value>>) -> BuiltinResult<Value> {
        futures::executor::block_on(value)
    }

    #[test]
    fn gzip_and_gunzip_round_trip_file() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join("runmat_gzip_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("data.txt");
        std::fs::write(&input, "hello").unwrap();
        run(gzip_builtin(vec![Value::String(path_to_string(&input))])).unwrap();
        let gz = dir.join("data.txt.gz");
        assert!(gz.exists());
        std::fs::remove_file(&input).unwrap();
        run(gunzip_builtin(vec![Value::String(path_to_string(&gz))])).unwrap();
        assert_eq!(std::fs::read_to_string(&input).unwrap(), "hello");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn gzip_accepts_string_array_inputs() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join("runmat_gzip_array_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.txt");
        let b = dir.join("b.txt");
        std::fs::write(&a, "a").unwrap();
        std::fs::write(&b, "b").unwrap();
        let inputs = runmat_value::StringArray::new(
            vec![path_to_string(&a), path_to_string(&b)],
            vec![2, 1],
        )
        .unwrap();
        let value = run(gzip_builtin(vec![Value::StringArray(inputs)])).unwrap();
        let Value::Cell(files) = value else {
            panic!("expected cell output");
        };
        assert_eq!(files.data.len(), 2);
        assert!(dir.join("a.txt.gz").exists());
        assert!(dir.join("b.txt.gz").exists());
        let _ = std::fs::remove_dir_all(dir);
    }
}
