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
    native_link_tokens: Vec<String>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("runmat-aot-pack: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let arguments = parse_arguments(std::env::args_os().skip(1))?;
    reject_existing(&arguments.payload_out)?;
    reject_existing(&arguments.manifest_out)?;
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
    let product = build_runtime_archive(
        &bytes,
        &environment,
        native_link_tokens,
        RuntimeArchiveEncoding::Zstd,
        RuntimeArchiveCapabilities::standalone_host(),
    )
    .map_err(|error| error.to_string())?;
    let manifest = canonical_manifest_json(&product.manifest)?;
    write_new(&arguments.payload_out, product.payload())?;
    if let Err(error) = write_new(&arguments.manifest_out, manifest.as_bytes()) {
        let _ = std::fs::remove_file(&arguments.payload_out);
        return Err(error);
    }
    Ok(())
}

fn parse_arguments(
    mut arguments: impl Iterator<Item = std::ffi::OsString>,
) -> Result<Arguments, String> {
    let mut archive = None;
    let mut payload_out = None;
    let mut manifest_out = None;
    let mut native_link_tokens = Vec::new();
    while let Some(argument) = arguments.next() {
        let argument = argument
            .into_string()
            .map_err(|_| "arguments must be valid UTF-8".to_string())?;
        let value = match argument.as_str() {
            "--archive" | "--payload-out" | "--manifest-out" | "--native-link-token" => arguments
                .next()
                .ok_or_else(|| format!("{argument} requires a value"))?,
            _ => return Err(format!("unknown argument `{argument}`")),
        };
        match argument.as_str() {
            "--archive" => archive = Some(PathBuf::from(value)),
            "--payload-out" => payload_out = Some(PathBuf::from(value)),
            "--manifest-out" => manifest_out = Some(PathBuf::from(value)),
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
