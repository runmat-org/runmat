use std::path::PathBuf;

#[cfg(target_os = "windows")]
use runmat_aot::archive::prepare_msvc_runtime_archive;
use runmat_aot::archive::{
    build_runtime_archive, RuntimeArchiveCapabilities, RuntimeArchiveEncoding,
    RuntimeArchiveManifest,
};

struct Arguments {
    archive: PathBuf,
    payload_out: PathBuf,
    manifest_out: PathBuf,
    notices_out: PathBuf,
    native_link_tokens: Vec<String>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("runmat-aot-pack: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut inputs = std::env::args_os().skip(1);
    match inputs.next().as_deref() {
        Some(value) if value == std::ffi::OsStr::new("--selected-pack") => {
            return run_selected(inputs);
        }
        Some(value) if value == std::ffi::OsStr::new("--selected-tree") => {
            return run_selected_tree(inputs);
        }
        _ => {}
    }
    let arguments = parse_arguments(std::env::args_os().skip(1))?;
    if arguments.archive == arguments.payload_out
        || arguments.archive == arguments.manifest_out
        || arguments.archive == arguments.notices_out
        || arguments.payload_out == arguments.manifest_out
        || arguments.payload_out == arguments.notices_out
        || arguments.manifest_out == arguments.notices_out
    {
        return Err("runtime archive inputs and outputs must be distinct paths".to_string());
    }
    reject_existing(&arguments.payload_out)?;
    reject_existing(&arguments.manifest_out)?;
    reject_existing(&arguments.notices_out)?;
    let bytes = std::fs::read(&arguments.archive)
        .map_err(|error| format!("read `{}`: {error}", arguments.archive.display()))?;
    let native_link_tokens = arguments.native_link_tokens;
    #[cfg(target_os = "windows")]
    let (bytes, native_link_tokens) = {
        let prepared = prepare_msvc_runtime_archive(&bytes, native_link_tokens)
            .map_err(|error| error.to_string())?;
        eprintln!(
            "runmat-aot-pack: normalized MSVC archive ({} duplicate members and {} bundled link tokens removed)",
            prepared.duplicate_members_removed, prepared.bundled_link_tokens_removed
        );
        (prepared.archive, prepared.native_link_tokens)
    };
    let environment = runmat_core::program_environment(runmat_core::CompatMode::Matlab);
    let mut capabilities = RuntimeArchiveCapabilities::standalone_host();
    capabilities.accelerators = runmat_aot::archive::read_native_accelerator_profile(&bytes)
        .map_err(|error| error.to_string())?;
    let product = build_runtime_archive(
        &bytes,
        &environment,
        native_link_tokens,
        RuntimeArchiveEncoding::Zstd,
        capabilities,
    )
    .map_err(|error| error.to_string())?;
    let manifest = canonical_manifest_json(&product.manifest)?;
    write_new(&arguments.payload_out, product.payload())?;
    if let Err(error) = write_new(&arguments.manifest_out, manifest.as_bytes()) {
        let _ = std::fs::remove_file(&arguments.payload_out);
        return Err(error);
    }
    if let Err(error) = write_new(
        &arguments.notices_out,
        runmat_runtime::third_party_notices::bytes(),
    ) {
        let _ = std::fs::remove_file(&arguments.manifest_out);
        let _ = std::fs::remove_file(&arguments.payload_out);
        return Err(error);
    }
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct HostedTreeDeclaration {
    schema_version: u16,
    archive: runmat_package_cache::ArchiveLimits,
    metadata_bytes: u64,
    entries: Vec<HostedTreeInput>,
}

#[derive(serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum HostedTreeInput {
    File {
        path: runmat_package::NormalizedRelativePath,
        source: PathBuf,
        executable: bool,
    },
    Directory {
        path: runmat_package::NormalizedRelativePath,
    },
    Symlink {
        path: runmat_package::NormalizedRelativePath,
        target: runmat_package::NormalizedRelativePath,
    },
}

fn run_selected_tree(mut inputs: impl Iterator<Item = std::ffi::OsString>) -> Result<(), String> {
    use runmat_aot::link::NativeStorageGrant;
    use runmat_aot::target_pack::{build_hosted_tree, HostedTreeEntry};
    let mut declaration_path = None;
    let mut metadata_out = None;
    let mut digest_out = None;
    let mut blob_root = None;
    while let Some(option) = inputs.next() {
        let option = option
            .into_string()
            .map_err(|_| "selected tree options must be UTF-8".to_string())?;
        let value = PathBuf::from(
            inputs
                .next()
                .ok_or_else(|| format!("{option} requires a value"))?,
        );
        let slot = match option.as_str() {
            "--declaration" => &mut declaration_path,
            "--metadata-out" => &mut metadata_out,
            "--digest-out" => &mut digest_out,
            "--blob-root" => &mut blob_root,
            _ => return Err(format!("unknown selected tree option `{option}`")),
        };
        if slot.replace(value).is_some() {
            return Err(format!("duplicate selected tree option `{option}`"));
        }
    }
    let declaration_path = declaration_path.ok_or("--declaration is required")?;
    let metadata_out = metadata_out.ok_or("--metadata-out is required")?;
    let digest_out = digest_out.ok_or("--digest-out is required")?;
    let blob_root = blob_root.ok_or("--blob-root is required")?;
    reject_existing(&metadata_out)?;
    reject_existing(&digest_out)?;
    reject_existing(&blob_root)?;
    let declaration_bytes = read_bounded(&declaration_path, 1024 * 1024)?;
    let declaration: HostedTreeDeclaration = serde_json::from_slice(&declaration_bytes)
        .map_err(|error| format!("decode explicit hosted tree declaration: {error}"))?;
    if declaration.schema_version != 1 {
        return Err("unsupported explicit hosted tree declaration schema".to_string());
    }
    let grant = NativeStorageGrant {
        archive: declaration.archive,
        metadata_bytes: declaration.metadata_bytes,
    };
    grant.archive.validate()?;
    let base = declaration_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let mut entries = Vec::with_capacity(declaration.entries.len());
    let mut file_count = 0u64;
    let mut total_bytes = 0u64;
    for entry in declaration.entries {
        entries.push(match entry {
            HostedTreeInput::File {
                path,
                source,
                executable,
            } => {
                let source = if source.is_relative() {
                    base.join(source)
                } else {
                    source
                };
                let physical = std::fs::symlink_metadata(&source)
                    .map_err(|error| format!("inspect `{}`: {error}", source.display()))?;
                if !physical.is_file() || physical.is_symlink() {
                    return Err(format!(
                        "selected tree source `{}` must be a regular physical file",
                        source.display()
                    ));
                }
                file_count = file_count
                    .checked_add(1)
                    .ok_or("selected tree file count overflow")?;
                total_bytes = total_bytes
                    .checked_add(physical.len())
                    .ok_or("selected tree physical byte count overflow")?;
                if file_count > grant.archive.max_files
                    || total_bytes > grant.archive.max_expanded_bytes
                {
                    return Err("selected tree source exceeds explicit storage bound".to_string());
                }
                HostedTreeEntry::File {
                    path,
                    bytes: read_bounded(
                        &source,
                        grant
                            .archive
                            .max_file_bytes
                            .min(runmat_execution_artifact::target_pack::MAX_COMPONENT_BYTES),
                    )?,
                    executable,
                }
            }
            HostedTreeInput::Directory { path } => HostedTreeEntry::Directory { path },
            HostedTreeInput::Symlink { path, target } => HostedTreeEntry::Symlink { path, target },
        });
    }
    let produced = build_hosted_tree(entries, grant).map_err(|error| error.to_string())?;
    let digest = serde_json::to_vec(&produced.digest)
        .map_err(|error| format!("encode hosted tree digest: {error}"))?;
    std::fs::create_dir(&blob_root)
        .map_err(|error| format!("create `{}`: {error}", blob_root.display()))?;
    for (id, bytes) in produced.blobs {
        write_new(&blob_root.join(id.as_str()), &bytes)?;
    }
    write_new(&metadata_out, &produced.metadata)?;
    write_new(&digest_out, &digest)?;
    Ok(())
}

struct SelectedArguments {
    archive: PathBuf,
    declaration: PathBuf,
    numerical_backend: Option<PathBuf>,
    numerical_evidence: Option<PathBuf>,
    manifest_out: PathBuf,
    component_root: PathBuf,
}

fn run_selected(arguments: impl Iterator<Item = std::ffi::OsString>) -> Result<(), String> {
    use runmat_execution_artifact::target_pack::MAX_COMPONENT_BYTES;
    let arguments = parse_selected_arguments(arguments)?;
    reject_existing(&arguments.manifest_out)?;
    reject_existing(&arguments.component_root)?;
    let declaration = read_bounded(&arguments.declaration, 1024 * 1024)?;
    let declaration: runmat_aot::target_pack::HostedPackDeclaration =
        serde_json::from_slice(&declaration)
            .map_err(|error| format!("decode independent target-build declaration: {error}"))?;
    let archive = read_bounded(&arguments.archive, MAX_COMPONENT_BYTES)?;
    let numerical_backend = arguments
        .numerical_backend
        .as_ref()
        .map(|path| read_bounded(path, MAX_COMPONENT_BYTES))
        .transpose()?;
    let numerical_evidence = arguments
        .numerical_evidence
        .as_ref()
        .map(|path| read_bounded(path, MAX_COMPONENT_BYTES))
        .transpose()?;
    let produced = runmat_aot::target_pack::build_hosted_target_pack(
        archive,
        numerical_backend,
        numerical_evidence,
        declaration,
    )
    .map_err(|error| error.to_string())?;
    let manifest = produced
        .manifest
        .canonical_bytes()
        .map_err(|error| error.to_string())?;
    std::fs::create_dir(&arguments.component_root).map_err(|error| {
        format!(
            "create component root `{}`: {error}",
            arguments.component_root.display()
        )
    })?;
    for (id, bytes) in produced.components {
        let path = arguments.component_root.join(id.as_str());
        write_new(&path, &bytes)?;
    }
    write_new(&arguments.manifest_out, &manifest)?;
    Ok(())
}

fn parse_selected_arguments(
    mut arguments: impl Iterator<Item = std::ffi::OsString>,
) -> Result<SelectedArguments, String> {
    let mut archive = None;
    let mut declaration = None;
    let mut numerical_backend = None;
    let mut numerical_evidence = None;
    let mut manifest_out = None;
    let mut component_root = None;
    while let Some(argument) = arguments.next() {
        let argument = argument
            .into_string()
            .map_err(|_| "selected pack arguments must be UTF-8".to_string())?;
        let value = arguments
            .next()
            .ok_or_else(|| format!("{argument} requires a value"))?;
        let path = PathBuf::from(value);
        let slot = match argument.as_str() {
            "--archive" => &mut archive,
            "--declaration" => &mut declaration,
            "--numerical-backend" => &mut numerical_backend,
            "--numerical-evidence" => &mut numerical_evidence,
            "--manifest-out" => &mut manifest_out,
            "--component-root" => &mut component_root,
            _ => return Err(format!("unknown selected pack argument `{argument}`")),
        };
        if slot.replace(path).is_some() {
            return Err(format!("duplicate selected pack argument `{argument}`"));
        }
    }
    Ok(SelectedArguments {
        archive: archive.ok_or("--archive is required")?,
        declaration: declaration.ok_or("--declaration is required")?,
        numerical_backend,
        numerical_evidence,
        manifest_out: manifest_out.ok_or("--manifest-out is required")?,
        component_root: component_root.ok_or("--component-root is required")?,
    })
}

fn read_bounded(path: &std::path::Path, maximum: u64) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let file =
        std::fs::File::open(path).map_err(|error| format!("open `{}`: {error}", path.display()))?;
    if file
        .metadata()
        .map_err(|error| format!("inspect `{}`: {error}", path.display()))?
        .len()
        > maximum
    {
        return Err(format!(
            "`{}` exceeds its selected pack byte bound",
            path.display()
        ));
    }
    let mut bytes = Vec::new();
    file.take(maximum.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read `{}`: {error}", path.display()))?;
    if bytes.len() as u64 > maximum {
        return Err(format!(
            "`{}` changed beyond its selected pack byte bound",
            path.display()
        ));
    }
    Ok(bytes)
}

fn parse_arguments(
    mut arguments: impl Iterator<Item = std::ffi::OsString>,
) -> Result<Arguments, String> {
    let mut archive = None;
    let mut payload_out = None;
    let mut manifest_out = None;
    let mut notices_out = None;
    let mut native_link_tokens = Vec::new();
    while let Some(argument) = arguments.next() {
        let argument = argument
            .into_string()
            .map_err(|_| "arguments must be valid UTF-8".to_string())?;
        let value = match argument.as_str() {
            "--archive"
            | "--payload-out"
            | "--manifest-out"
            | "--notices-out"
            | "--native-link-token" => arguments
                .next()
                .ok_or_else(|| format!("{argument} requires a value"))?,
            _ => return Err(format!("unknown argument `{argument}`")),
        };
        match argument.as_str() {
            "--archive" => archive = Some(PathBuf::from(value)),
            "--payload-out" => payload_out = Some(PathBuf::from(value)),
            "--manifest-out" => manifest_out = Some(PathBuf::from(value)),
            "--notices-out" => notices_out = Some(PathBuf::from(value)),
            "--native-link-token" => native_link_tokens.push(
                value
                    .into_string()
                    .map_err(|_| "native-link tokens must be valid UTF-8".to_string())?,
            ),
            _ => unreachable!(),
        }
    }
    Ok(Arguments {
        archive: archive.ok_or_else(|| "--archive is required".to_string())?,
        payload_out: payload_out.ok_or_else(|| "--payload-out is required".to_string())?,
        manifest_out: manifest_out.ok_or_else(|| "--manifest-out is required".to_string())?,
        notices_out: notices_out.ok_or_else(|| "--notices-out is required".to_string())?,
        native_link_tokens,
    })
}

fn canonical_manifest_json(manifest: &RuntimeArchiveManifest) -> Result<String, String> {
    let mut output = serde_json::to_string_pretty(manifest)
        .map_err(|error| format!("encode runtime archive manifest: {error}"))?;
    output.push('\n');
    Ok(output)
}

fn reject_existing(path: &std::path::Path) -> Result<(), String> {
    if path.exists() {
        Err(format!("refusing to overwrite `{}`", path.display()))
    } else if !path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .is_none_or(std::path::Path::is_dir)
    {
        Err(format!(
            "output directory for `{}` does not exist",
            path.display()
        ))
    } else {
        Ok(())
    }
}

fn write_new(path: &std::path::Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("create `{}`: {error}", path.display()))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| format!("write `{}`: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standalone_archive_requires_explicit_notice_output() {
        let options = [
            "--archive",
            "runtime.a",
            "--payload-out",
            "runtime.payload",
            "--manifest-out",
            "runtime.json",
        ];
        assert!(parse_arguments(
            options
                .iter()
                .map(|argument| std::ffi::OsString::from(*argument))
        )
        .is_err());
        let complete = options
            .into_iter()
            .chain(["--notices-out", "THIRD_PARTY_NOTICES.txt"]);
        let parsed = parse_arguments(complete.map(std::ffi::OsString::from)).unwrap();
        assert_eq!(parsed.notices_out, PathBuf::from("THIRD_PARTY_NOTICES.txt"));
    }
}
