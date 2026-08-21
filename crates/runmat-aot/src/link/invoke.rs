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
}

pub fn link_standalone(
    object: &RelocatableNativeObject,
    runtime: &RuntimeArchive,
    request: &LinkRequest,
) -> AotResult<LinkedProgram> {
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
    let response_path = temporary.path().join("link.rsp");
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
    let response = super::response::encode(&plan.response_tokens, plan.driver.family)?;
    std::fs::write(&response_path, response)
        .map_err(|error| AotError::io("write linker response file", &response_path, error))?;
    let result = std::process::Command::new(&plan.driver.path)
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
    if !linked_path.is_file() {
        return Err(AotError::contract(
            "aot.link.output",
            "native linker succeeded without producing the requested executable",
        ));
    }
    let prior_output = temporary.path().join("prior-output");
    let had_prior_output = request.output.exists();
    if had_prior_output && !request.overwrite {
        return Err(AotError::contract(
            "aot.link.output_exists",
            format!(
                "output `{}` already exists; enable overwrite explicitly to replace it",
                request.output.display()
            ),
        ));
    }
    if had_prior_output {
        std::fs::rename(&request.output, &prior_output)
            .map_err(|error| AotError::io("preserve existing output", &request.output, error))?;
    }
    if let Err(error) = std::fs::rename(&linked_path, &request.output) {
        if had_prior_output {
            let _ = std::fs::rename(&prior_output, &request.output);
        }
        return Err(AotError::io(
            "publish output executable",
            &request.output,
            error,
        ));
    }
    let retained_temps = if request.keep_temps {
        Some(temporary.keep())
    } else {
        None
    };
    Ok(LinkedProgram {
        output: request.output.clone(),
        retained_temps,
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
