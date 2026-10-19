use std::path::PathBuf;

use runmat_native_codegen::aot::RelocatableNativeObject;

use crate::{archive::RuntimeArchive, AotError, AotResult};

use super::{build_link_plan, discover_linker};

#[derive(Clone, Debug)]
pub struct LinkRequest {
    pub output: PathBuf,
    pub linker: Option<PathBuf>,
    pub keep_temps: bool,
    pub overwrite: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedProgram {
    pub output: PathBuf,
    pub retained_temps: Option<PathBuf>,
    pub product: runmat_execution::host::NativeProductMode,
    pub output_digest: runmat_execution::Digest,
    pub selected_build: Option<super::NativeLinkBuildIdentity>,
    pub inputs: super::NativeProgramLinkIdentity,
}

pub(super) enum LinkProcessEnvironment {
    AmbientHost,
    QualifiedTree,
}

pub fn link_standalone(
    object: &RelocatableNativeObject,
    runtime: &RuntimeArchive,
    request: &LinkRequest,
) -> AotResult<LinkedProgram> {
    // This entrypoint discovers an ambient host linker. Selected products need
    // the verified retained-input/toolchain adapter, never a metadata downgrade.
    if object.manifest.product.selected().is_some() || runtime.manifest.product.selected().is_some()
    {
        return Err(AotError::contract(
            "aot.link.selected_admission",
            "selected products require verified selected-link admission",
        ));
    }
    let parent = request
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    if !parent.is_dir() {
        return Err(AotError::contract(
            "aot.link.output",
            format!("output directory `{}` does not exist", parent.display()),
        ));
    }
    let temporary = tempfile::Builder::new()
        .prefix(".runmat-aot-")
        .tempdir_in(parent)
        .map_err(|error| AotError::io("create temporary link directory", parent, error))?;
    let object_path = temporary.path().join(if cfg!(target_os = "windows") {
        "program.obj"
    } else {
        "program.o"
    });
    let archive_path = temporary.path().join(if cfg!(target_os = "windows") {
        "runmat-runtime.lib"
    } else {
        "librunmat-runtime.a"
    });
    let linked_path = temporary.path().join(if cfg!(target_os = "windows") {
        "program.exe"
    } else {
        "program"
    });
    std::fs::write(&object_path, &object.bytes)
        .map_err(|error| AotError::io("write user object", &object_path, error))?;
    let archive = runtime.decode()?;
    std::fs::write(&archive_path, archive)
        .map_err(|error| AotError::io("write runtime archive", &archive_path, error))?;
    let driver = discover_linker(request.linker.as_deref())?;
    let plan = build_link_plan(
        object,
        runtime,
        driver,
        &object_path,
        &archive_path,
        &linked_path,
    )?;
    invoke_prepared(
        plan,
        temporary,
        request,
        LinkProcessEnvironment::AmbientHost,
    )
}

/// Both admission routes converge here only after their own exact inputs and
/// tool authority are established. The private lease survives process exit,
/// linked-image validation and recoverable publication (including failures).
pub(super) fn invoke_prepared(
    plan: super::LinkPlan,
    temporary: tempfile::TempDir,
    request: &LinkRequest,
    environment: LinkProcessEnvironment,
) -> AotResult<LinkedProgram> {
    match (&environment, plan.product.selected(), &plan.selected_build) {
        (LinkProcessEnvironment::AmbientHost, None, None) => {}
        (LinkProcessEnvironment::QualifiedTree, Some(binding), Some(build))
            if binding
                .fingerprint()
                .map_err(|error| AotError::contract("aot.link.product", error.to_string()))?
                == build.product => {}
        _ => {
            return Err(AotError::contract(
                "aot.link.product",
                "link process mode differs from selected build admission",
            ))
        }
    }
    let response_path = temporary.path().join("link.rsp");
    let linked_path = plan.output.clone();
    let response = super::response::encode(&plan.response_tokens, plan.driver.family)?;
    std::fs::write(&response_path, response)
        .map_err(|error| AotError::io("write linker response file", &response_path, error))?;
    let mut command = plan.driver.command();
    if matches!(environment, LinkProcessEnvironment::QualifiedTree) {
        // The discovered tool environment (for example the MSVC LIB/PATH set)
        // is part of the linker authority and survives the isolation reset.
        command
            .env_clear()
            .envs(plan.driver.environment.iter().cloned())
            .current_dir(temporary.path());
    }
    let result = command
        .arg(format!("@{}", response_path.display()))
        .output()
        .map_err(|error| AotError::io("invoke native linker", &plan.driver.path, error))?;
    if !result.status.success() {
        let diagnostic = linker_diagnostic(&result.stdout, &result.stderr);
        return Err(AotError::Linker {
            driver: plan.driver.path,
            status: result.status.to_string(),
            diagnostic: bounded_diagnostic(&diagnostic),
        });
    }
    if !std::fs::symlink_metadata(&linked_path)
        .map(|metadata| metadata.is_file())
        .unwrap_or(false)
    {
        return Err(AotError::contract(
            "aot.link.output",
            "native linker succeeded without producing the requested executable",
        ));
    }
    let mut linked_file = std::fs::File::open(&linked_path)
        .map_err(|error| AotError::io("open completed linked image", &linked_path, error))?;
    if let Some(binding) = plan.product.selected() {
        #[cfg(unix)]
        super::native_input::require_unix_executable_mode(
            &linked_file
                .metadata()
                .map_err(|error| AotError::io("inspect linked image mode", &linked_path, error))?,
            binding.target.target.operating_system,
            super::native_input::NativeImagePurpose::FinalExecutable,
        )?;
        // Non-Unix build hosts do not certify a Unix deployment mode. Target
        // installation remains an independent deployment obligation.
        let reader = linked_file
            .try_clone()
            .map_err(|error| AotError::io("inspect completed linked image", &linked_path, error))?;
        let cache = object::read::ReadCache::new(reader);
        super::native_input::validate_linked_program(&cache, &binding.target.target)?;
    }
    use std::io::{Read, Seek};
    linked_file
        .rewind()
        .map_err(|error| AotError::io("rewind completed linked image", &linked_path, error))?;
    let mut digest = runmat_execution::DigestStream::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let count = linked_file
            .read(&mut buffer)
            .map_err(|error| AotError::io("hash completed linked image", &linked_path, error))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let output_digest = digest.finish();
    drop(linked_file);
    let retained_temps = super::publication::publish(
        temporary,
        &linked_path,
        &request.output,
        request.overwrite,
        request.keep_temps,
    )?;
    Ok(LinkedProgram {
        output: request.output.clone(),
        retained_temps,
        product: plan.product,
        output_digest,
        selected_build: plan.selected_build,
        inputs: plan.inputs,
    })
}

fn linker_diagnostic(stdout: &[u8], stderr: &[u8]) -> String {
    let stdout = String::from_utf8_lossy(stdout);
    let stderr = String::from_utf8_lossy(stderr);
    match (stdout.trim(), stderr.trim()) {
        ("", stderr) => stderr.to_string(),
        (stdout, "") => stdout.to_string(),
        (stdout, stderr) => format!("{stdout}\n{stderr}"),
    }
}

fn bounded_diagnostic(diagnostic: &str) -> String {
    const MAX_BYTES: usize = 64 * 1024;
    if diagnostic.len() <= MAX_BYTES {
        diagnostic.trim().to_string()
    } else {
        let mut end = MAX_BYTES;
        while !diagnostic.is_char_boundary(end) {
            end -= 1;
        }
        format!(
            "{}\n[linker diagnostic truncated]",
            diagnostic[..end].trim()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::linker_diagnostic;

    #[test]
    fn linker_diagnostic_preserves_stdout_and_stderr() {
        assert_eq!(linker_diagnostic(b"MSVC stdout\n", b""), "MSVC stdout");
        assert_eq!(
            linker_diagnostic(b"stdout\n", b"stderr\n"),
            "stdout\nstderr"
        );
    }
}
