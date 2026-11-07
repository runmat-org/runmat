//! MATLAB-compatible `gzip` and `gunzip` file helpers.

use std::path::{Path, PathBuf};
use std::time::Duration;

#[cfg(test)]
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, Value};
use url::Url;

use crate::builtins::common::fs::{expand_user_path, path_to_string};
use crate::builtins::common::shape::effective_rank;
use crate::builtins::io::http::transport::{self, HttpMethod, HttpRequest};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

#[path = "gzip/codec.rs"]
mod codec;
#[path = "gzip/filesystem.rs"]
mod filesystem;

#[cfg(test)]
#[path = "gzip/cooperation_tests.rs"]
mod cooperation_tests;

fn archive_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(name)
        .with_identifier(format!("RunMat:{name}:RuntimeError"))
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

fn archive_error_with_source(
    name: &str,
    message: impl Into<String>,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(name)
        .with_identifier(format!("RunMat:{name}:RuntimeError"))
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .with_source(source)
        .build()
}

pub use runmat_builtins::{
    GUNZIP_DESCRIPTOR, GUNZIP_ERROR_OUTPUTS, GUNZIP_INTEGER_AUDIT, GZIP_DESCRIPTOR,
    GZIP_ERROR_OUTPUTS, GZIP_INTEGER_AUDIT,
};

fn scalar_text(value: &Value, name: &str, arg: &str) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text.as_str().to_owned()),
        Value::CharArray(array) if array.rows == 1 && effective_rank(&array.shape) <= 2 => {
            crate::builtins::io::utf16::decode_utf16_with_error(array.data(), || {
                archive_error(
                    name,
                    format!("{name}: {arg} must contain valid UTF-16 text"),
                )
            })
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(array.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .to_owned())
        }
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
            .data()
            .iter()
            .map(|text| {
                path_from_text(
                    text.as_str().ok_or_else(|| {
                        archive_error(name, format!("{name}: missing string is not a filename"))
                    })?,
                    name,
                )
            })
            .collect(),
        Value::Cell(cell) => cell
            .data()
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

async fn expand_input_paths(
    filesystem: &filesystem::ArchiveFilesystem,
    paths: Vec<PathBuf>,
    name: &str,
) -> BuiltinResult<Vec<PathBuf>> {
    let mut out = Vec::new();
    for path in paths {
        let text = path_to_string(&path);
        if text.contains('*') || text.contains('?') {
            let mut matches: Vec<_> = filesystem
                .matching(&path)
                .await?
                .map_err(|source| {
                    archive_error_with_source(name, format!("{name}: {source}"), source)
                })?
                .into_iter()
                .map(|entry| entry.path)
                .collect();
            // glob's historical ordered results must not depend on a provider's
            // directory iteration order or the traversal's deduplication set.
            matches.sort();
            out.extend(matches);
            continue;
        }
        if filesystem
            .metadata(&path)
            .await?
            .is_ok_and(|meta| meta.is_dir())
        {
            for entry in filesystem.directory(&path).await?.map_err(|source| {
                archive_error_with_source(name, format!("{name}: {source}"), source)
            })? {
                if !entry.metadata.is_dir() {
                    out.push(entry.path);
                }
            }
        } else {
            out.push(path);
        }
    }
    for path in &out {
        gzip_filename(path)?;
    }
    Ok(out)
}

fn gzip_filename(path: &Path) -> BuiltinResult<&str> {
    path.file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| archive_error("gzip", "gzip: invalid filename"))
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
    context = true,
    builtin_path = "crate::builtins::io::archive::gzip"
)]
async fn gzip_builtin(
    context: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
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
    // Capture only after every public control has been admitted, and before
    // the first suspension or filesystem operation.
    let mut filesystem =
        filesystem::ArchiveFilesystem::capture(codec::Operation::Compress, context.as_ref())?;
    filesystem.checkpoint(0).await?;
    let inputs = expand_input_paths(&filesystem, inputs, "gzip").await?;
    let mut outputs = Vec::with_capacity(inputs.len());
    for input in inputs {
        outputs.push(gzip_one(&mut filesystem, &input, output_folder.as_deref()).await?);
    }
    cellstr(outputs, "gzip")
}

async fn gzip_one(
    filesystem: &mut filesystem::ArchiveFilesystem,
    input: &Path,
    output_folder: Option<&Path>,
) -> BuiltinResult<PathBuf> {
    let filename = gzip_filename(input)?;
    let bytes = filesystem.read(input).await?;
    let compressed = filesystem.transform(&bytes).await?;
    let output = output_folder
        .map(|folder| folder.join(format!("{filename}.gz")))
        .unwrap_or_else(|| input.with_file_name(format!("{filename}.gz")));
    filesystem.checkpoint(0).await?;
    if let Some(parent) = output.parent() {
        filesystem.create_dir_all(parent).await?;
    }
    filesystem.write(&output, &compressed).await?;
    Ok(output)
}

#[runtime_builtin(
    name = "gunzip",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::archive::gzip"
)]
async fn gunzip_builtin(
    context: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
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
    let mut filesystem =
        filesystem::ArchiveFilesystem::capture(codec::Operation::Decompress, context.as_ref())?;
    filesystem.checkpoint(0).await?;
    let mut outputs = Vec::with_capacity(inputs.len());
    for input in inputs {
        outputs.push(
            extract_gunzip_source(
                &mut filesystem,
                input,
                output_folder.as_deref(),
                context.as_ref(),
            )
            .await?,
        );
    }
    cellstr(outputs, "gunzip")
}

enum GunzipSource {
    Path { path: PathBuf, filename: String },
    Url { url: Url, filename: String },
}

fn gunzip_sources_from_value(value: &Value) -> BuiltinResult<Vec<GunzipSource>> {
    match value {
        Value::StringArray(array) => array
            .data()
            .iter()
            .map(|text| {
                parse_gunzip_source(text.as_str().ok_or_else(|| {
                    archive_error("gunzip", "gunzip: missing string is not a filename")
                })?)
            })
            .collect(),
        Value::Cell(cell) => cell
            .data()
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
            let filename = url
                .path_segments()
                .and_then(|mut segments| segments.next_back())
                .filter(|name| !name.is_empty())
                .unwrap_or("download.gz")
                .to_owned();
            gunzip_output_name(&filename)?;
            return Ok(GunzipSource::Url { url, filename });
        }
    }
    let path = path_from_text(text, "gunzip")?;
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| archive_error("gunzip", "gunzip: invalid filename"))?
        .to_owned();
    gunzip_output_name(&filename)?;
    Ok(GunzipSource::Path { path, filename })
}

async fn extract_gunzip_source(
    filesystem: &mut filesystem::ArchiveFilesystem,
    source: GunzipSource,
    output_folder: Option<&Path>,
    context: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<PathBuf> {
    match source {
        GunzipSource::Path { path, filename } => {
            gunzip_one(filesystem, &path, &filename, output_folder).await
        }
        GunzipSource::Url { url, filename } => {
            let response = transport::send_request(
                HttpRequest {
                    url: url.clone(),
                    method: HttpMethod::Get,
                    headers: Vec::new(),
                    body: None,
                    timeout: DEFAULT_TIMEOUT,
                    user_agent: "RunMat gunzip/0.0".to_string(),
                },
                context,
            )
            .await
            .map_err(|err| match err {
                transport::TransportCallError::Transport(err) => {
                    archive_error_with_source("gunzip", err.message_with_prefix("gunzip"), err)
                }
                transport::TransportCallError::Runtime(err) => err,
            })?;
            filesystem.check()?;
            gunzip_bytes(filesystem, &response.body, &filename, output_folder).await
        }
    }
}

async fn gunzip_one(
    filesystem: &mut filesystem::ArchiveFilesystem,
    input: &Path,
    filename: &str,
    output_folder: Option<&Path>,
) -> BuiltinResult<PathBuf> {
    let bytes = filesystem.read(input).await?;
    gunzip_bytes(
        filesystem,
        &bytes,
        filename,
        output_folder.or_else(|| input.parent()),
    )
    .await
}

async fn gunzip_bytes(
    filesystem: &mut filesystem::ArchiveFilesystem,
    bytes: &[u8],
    filename: &str,
    output_folder: Option<&Path>,
) -> BuiltinResult<PathBuf> {
    let output_name = gunzip_output_name(filename)?;
    let decoded = filesystem.transform(bytes).await?;
    let output = output_folder
        .map(|folder| folder.join(output_name))
        .unwrap_or_else(|| PathBuf::from(output_name));
    filesystem.checkpoint(0).await?;
    if let Some(parent) = output.parent() {
        filesystem.create_dir_all(parent).await?;
    }
    filesystem.write(&output, &decoded).await?;
    Ok(output)
}

fn gunzip_output_name(filename: &str) -> BuiltinResult<&str> {
    let output_name = filename.strip_suffix(".gz").unwrap_or(filename);
    if output_name.is_empty()
        || matches!(output_name, "." | "..")
        || output_name.contains(['/', '\\'])
    {
        return Err(archive_error("gunzip", "gunzip: invalid output filename"));
    }
    Ok(output_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(value: impl std::future::Future<Output = BuiltinResult<Value>>) -> BuiltinResult<Value> {
        futures::executor::block_on(value)
    }

    #[test]
    fn gzip_and_gunzip_round_trip_file() {
        let _guard = runmat_filesystem::provider_override_lock();
        let dir = std::env::temp_dir().join("runmat_gzip_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("data.txt");
        std::fs::write(&input, "hello").unwrap();
        run(gzip_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                path_to_string(&input).into(),
            )]),
        ))
        .unwrap();
        let gz = dir.join("data.txt.gz");
        assert!(gz.exists());
        std::fs::remove_file(&input).unwrap();
        run(gunzip_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                path_to_string(&gz).into(),
            )]),
        ))
        .unwrap();
        assert_eq!(std::fs::read_to_string(&input).unwrap(), "hello");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn gzip_accepts_string_array_inputs() {
        let _guard = runmat_filesystem::provider_override_lock();
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
        let value = run(gzip_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::StringArray(
                inputs,
            )]),
        ))
        .unwrap();
        let Value::Cell(files) = value else {
            panic!("expected cell output");
        };
        assert_eq!(files.data().len(), 2);
        assert!(dir.join("a.txt.gz").exists());
        assert!(dir.join("b.txt.gz").exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn wildcard_round_trip_uses_active_memory_filesystem_and_sorted_paths() {
        let _guard = runmat_filesystem::provider_override_lock();
        let filesystem = vfs::MemoryFsProvider::with_current_dir("/workspace");
        filesystem
            .write_project_path("/workspace/in/b.txt", b"second")
            .unwrap();
        filesystem
            .write_project_path("/workspace/in/a.txt", b"first")
            .unwrap();
        vfs::with_provider_override(std::sync::Arc::new(filesystem), || {
            let output = run(gzip_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::String("in/*.txt".into()),
                    Value::String("compressed".into()),
                ]),
            ))
            .unwrap();
            let Value::Cell(files) = output else {
                panic!("expected cell paths")
            };
            let names: Vec<_> = files
                .data()
                .iter()
                .map(|value| scalar_text(value, "gzip", "output").unwrap())
                .collect();
            let expected: Vec<_> = ["a.txt.gz", "b.txt.gz"]
                .into_iter()
                .map(|name| {
                    std::path::Path::new("compressed")
                        .join(name)
                        .to_string_lossy()
                        .into_owned()
                })
                .collect();
            assert_eq!(names, expected);
            run(gunzip_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Cell(files),
                    Value::String("restored".into()),
                ]),
            ))
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
        let error = run(gzip_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("/missing/gzip-admission/*.txt".into()),
                Value::Num(1.0),
            ]),
        ))
        .unwrap_err();
        assert!(error.message().contains("outputfolder must be"), "{error}");
    }

    #[test]
    fn requested_outputs_reject_before_invalid_input_or_provider_access() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        for (name, result) in [
            (
                "gzip",
                run(gzip_builtin(
                    None,
                    crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
                )),
            ),
            (
                "gunzip",
                run(gunzip_builtin(
                    None,
                    crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
                )),
            ),
        ] {
            assert_eq!(
                result.unwrap_err().identifier(),
                Some(format!("RunMat:{name}:TooManyOutputs").as_str())
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn decompression_rejects_non_basename_destinations_before_decode_or_write() {
        let filesystem = std::sync::Arc::new(vfs::MemoryFsProvider::with_current_dir("/workspace"));
        let mut owner = vfs::with_provider_override(filesystem, || {
            filesystem::ArchiveFilesystem::capture(codec::Operation::Decompress, None)
                .expect("archive owner")
        });
        for name in [".gz", "..gz", "...gz", "../escape.gz", "nested\\escape.gz"] {
            let error = futures::executor::block_on(gunzip_bytes(
                &mut owner,
                b"not a gzip stream",
                name,
                None,
            ))
            .unwrap_err();
            assert!(
                error.message().contains("invalid output filename"),
                "{name}: {error}"
            );
        }
    }
}
