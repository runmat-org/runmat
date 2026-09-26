//! MATLAB-compatible `gzip` and `gunzip` file helpers.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, Value};
use url::Url;

use crate::builtins::common::fs::{enumeration, expand_user_path, path_to_string};
use crate::builtins::common::shape::effective_rank;
use crate::builtins::io::http::transport::{self, HttpMethod, HttpRequest};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

fn archive_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(name)
        .with_identifier(format!("RunMat:{name}:RuntimeError"))
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

pub use runmat_builtins::{
    GUNZIP_DESCRIPTOR, GUNZIP_ERROR_OUTPUTS, GUNZIP_INTEGER_AUDIT, GZIP_DESCRIPTOR,
    GZIP_ERROR_OUTPUTS, GZIP_INTEGER_AUDIT,
};

fn scalar_text(value: &Value, name: &str, arg: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) if array.rows == 1 && effective_rank(&array.shape) <= 2 => {
            Ok(array.data.iter().collect())
        }
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

async fn expand_input_paths(paths: Vec<PathBuf>, name: &str) -> BuiltinResult<Vec<PathBuf>> {
    let mut out = Vec::new();
    for path in paths {
        let text = path_to_string(&path);
        if text.contains('*') || text.contains('?') {
            let mut matches: Vec<_> = enumeration::matching(&path)
                .await
                .map_err(|err| archive_error(name, format!("{name}: {err}")))?
                .into_iter()
                .map(|entry| entry.path)
                .collect();
            // glob's historical ordered results must not depend on a provider's
            // directory iteration order or the traversal's deduplication set.
            matches.sort();
            out.extend(matches);
            continue;
        }
        if vfs::metadata_async(&path)
            .await
            .is_ok_and(|meta| meta.is_dir())
        {
            for entry in enumeration::directory(&path)
                .await
                .map_err(|err| archive_error(name, format!("{name}: {err}")))?
            {
                if !entry.metadata.is_dir() {
                    out.push(entry.path);
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
    binding_variant = "default",
    builtin_path = "crate::builtins::io::archive::gzip"
)]
async fn gzip_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(&GZIP_DESCRIPTOR, &GZIP_ERROR_OUTPUTS, "gzip")?;
    if args.is_empty() || args.len() > 2 {
        return Err(archive_error(
            "gzip",
            "gzip: filename and optional output folder are expected",
        ));
    }
    let inputs = paths_from_value(&args[0], "gzip", "filename")?;
    let output_folder = if let Some(value) = args.get(1) {
        Some(path_from_value(value, "gzip", "outputfolder")?)
    } else {
        None
    };
    // Admit every public control before the first filesystem operation.
    let inputs = expand_input_paths(inputs, "gzip").await?;
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
    binding_variant = "default",
    builtin_path = "crate::builtins::io::archive::gzip"
)]
async fn gunzip_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &GUNZIP_DESCRIPTOR,
        &GUNZIP_ERROR_OUTPUTS,
        "gunzip",
    )?;
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
    let output_name = filename.strip_suffix(".gz").unwrap_or(filename);
    if output_name.is_empty()
        || matches!(output_name, "." | "..")
        || output_name.contains(['/', '\\'])
    {
        return Err(archive_error("gunzip", "gunzip: invalid output filename"));
    }
    let mut decoder = GzDecoder::new(bytes);
    let mut decoded = Vec::new();
    decoder
        .read_to_end(&mut decoded)
        .map_err(|err| archive_error("gunzip", format!("gunzip: {err}")))?;
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

    #[test]
    fn wildcard_round_trip_uses_active_memory_filesystem_and_sorted_paths() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let filesystem = vfs::MemoryFsProvider::with_current_dir("/workspace");
        filesystem
            .write_project_path("/workspace/in/b.txt", b"second")
            .unwrap();
        filesystem
            .write_project_path("/workspace/in/a.txt", b"first")
            .unwrap();
        vfs::with_provider_override(std::sync::Arc::new(filesystem), || {
            let output = run(gzip_builtin(vec![
                Value::String("in/*.txt".into()),
                Value::String("compressed".into()),
            ]))
            .unwrap();
            let Value::Cell(files) = output else {
                panic!("expected cell paths")
            };
            let names: Vec<_> = files
                .data
                .iter()
                .map(|value| scalar_text(value, "gzip", "output").unwrap())
                .collect();
            assert_eq!(names, vec!["compressed/a.txt.gz", "compressed/b.txt.gz"]);
            run(gunzip_builtin(vec![
                Value::Cell(files),
                Value::String("restored".into()),
            ]))
            .unwrap();
            assert_eq!(
                futures::executor::block_on(vfs::read_async("restored/a.txt")).unwrap(),
                b"first"
            );
            assert_eq!(
                futures::executor::block_on(vfs::read_async("restored/b.txt")).unwrap(),
                b"second"
            );
        });
    }

    #[test]
    fn invalid_output_folder_is_admitted_before_wildcard_filesystem_access() {
        let error = run(gzip_builtin(vec![
            Value::String("/missing/gzip-admission/*.txt".into()),
            Value::Num(1.0),
        ]))
        .unwrap_err();
        assert!(error.message().contains("outputfolder must be"), "{error}");
    }

    #[test]
    fn requested_outputs_reject_before_invalid_input_or_provider_access() {
        let _outputs = crate::output_count::push_output_count(Some(2));
        for (name, result) in [
            ("gzip", run(gzip_builtin(vec![Value::Num(1.0)]))),
            ("gunzip", run(gunzip_builtin(vec![Value::Num(1.0)]))),
        ] {
            assert_eq!(
                result.unwrap_err().identifier(),
                Some(format!("RunMat:{name}:TooManyOutputs").as_str())
            );
        }
    }

    #[test]
    fn decompression_rejects_non_basename_destinations_before_decode_or_write() {
        for name in [".gz", "..gz", "...gz", "../escape.gz", "nested\\escape.gz"] {
            let error = futures::executor::block_on(gunzip_bytes(b"not a gzip stream", name, None))
                .unwrap_err();
            assert!(
                error.message().contains("invalid output filename"),
                "{name}: {error}"
            );
        }
    }
}
