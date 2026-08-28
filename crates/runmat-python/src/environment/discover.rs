use std::collections::BTreeSet;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

use super::{PythonDiscoveryRequest, PythonInstallation, PythonVersion};
use crate::PythonError;

const DISCOVERY_CODE: &str = r#"
import json
import os
import sys
import sysconfig

major, minor, patch = sys.version_info[:3]
prefix = os.path.realpath(sys.prefix)
base_prefix = os.path.realpath(getattr(sys, "base_prefix", sys.prefix))
executable = os.path.realpath(sys.executable)

candidates = []
library = sysconfig.get_config_var("LDLIBRARY") or sysconfig.get_config_var("INSTSONAME")
library_dir = sysconfig.get_config_var("LIBDIR")
if library and os.path.isabs(library):
    candidates.append(library)
if library and library_dir:
    candidates.append(os.path.join(library_dir, library))

framework = sysconfig.get_config_var("PYTHONFRAMEWORK")
framework_prefix = sysconfig.get_config_var("PYTHONFRAMEWORKPREFIX")
if framework and framework_prefix:
    candidates.append(os.path.join(framework_prefix, framework + ".framework", "Versions", f"{major}.{minor}", framework))

if os.name == "nt":
    dll = f"python{major}{minor}.dll"
    candidates.extend([
        os.path.join(base_prefix, dll),
        os.path.join(prefix, dll),
        os.path.join(os.path.dirname(executable), dll),
    ])

resolved = ""
for candidate in candidates:
    candidate = os.path.realpath(candidate)
    if os.path.isfile(candidate):
        resolved = candidate
        break

print(json.dumps({
    "version": {"major": major, "minor": minor, "patch": patch},
    "executable": executable,
    "library": resolved,
    "home": base_prefix,
    "prefix": prefix,
}, sort_keys=True))
"#;

#[derive(Debug, Deserialize)]
struct DiscoveryOutput {
    version: PythonVersion,
    executable: PathBuf,
    library: PathBuf,
    home: PathBuf,
    prefix: PathBuf,
}

pub fn discover_python(
    request: &PythonDiscoveryRequest,
) -> Result<PythonInstallation, PythonError> {
    validate_request(request)?;
    let candidates = executable_candidates(request);
    let mut failures = Vec::new();
    for candidate in candidates {
        match inspect_candidate(&candidate) {
            Ok(installation) if version_matches(request, installation.version) => {
                return Ok(installation);
            }
            Ok(installation) => failures.push(format!(
                "{} reports Python {}",
                candidate.display(),
                installation.version
            )),
            Err(error) => failures.push(format!("{}: {}", candidate.display(), error.message)),
        }
    }
    let requested = request
        .version
        .map(|(major, minor)| format!(" {major}.{minor}"))
        .unwrap_or_default();
    Err(PythonError::host(
        "PythonDiscoveryError",
        format!(
            "could not discover a supported CPython{requested} interpreter{}",
            if failures.is_empty() {
                String::new()
            } else {
                format!(": {}", failures.join("; "))
            }
        ),
    ))
}

fn validate_request(request: &PythonDiscoveryRequest) -> Result<(), PythonError> {
    if request.minimum_version.0 != 3 {
        return Err(PythonError::host(
            "PythonConfigurationError",
            "RunMat's Python adapter supports CPython 3.x",
        ));
    }
    if let Some(maximum) = request.maximum_version {
        if maximum < request.minimum_version {
            return Err(PythonError::host(
                "PythonConfigurationError",
                "maximum Python version precedes the minimum version",
            ));
        }
    }
    if let Some(version) = request.version {
        if version < request.minimum_version
            || request
                .maximum_version
                .is_some_and(|maximum| version > maximum)
        {
            return Err(PythonError::host(
                "PythonConfigurationError",
                format!(
                    "requested Python {}.{} is outside the configured version range",
                    version.0, version.1
                ),
            ));
        }
    }
    Ok(())
}

fn executable_candidates(request: &PythonDiscoveryRequest) -> Vec<PathBuf> {
    if let Some(executable) = &request.executable {
        return vec![executable.clone()];
    }
    let mut names = Vec::<OsString>::new();
    if let Some((major, minor)) = request.version {
        names.push(format!("python{major}.{minor}").into());
        #[cfg(windows)]
        names.push(format!("python{major}{minor}.exe").into());
    }
    names.push("python3".into());
    #[cfg(windows)]
    names.push("python.exe".into());

    let mut resolved = BTreeSet::new();
    for name in names {
        let path = PathBuf::from(&name);
        if path.components().count() > 1 {
            resolved.insert(path);
            continue;
        }
        if let Some(paths) = std::env::var_os("PATH") {
            for directory in std::env::split_paths(&paths) {
                let candidate = directory.join(&name);
                if candidate.is_file() {
                    resolved.insert(candidate);
                }
            }
        }
    }
    resolved.into_iter().collect()
}

fn inspect_candidate(executable: &Path) -> Result<PythonInstallation, PythonError> {
    let output = Command::new(executable)
        .args(["-I", "-c", DISCOVERY_CODE])
        .output()
        .map_err(|error| {
            PythonError::host(
                "PythonDiscoveryError",
                format!("could not launch interpreter: {error}"),
            )
        })?;
    if !output.status.success() {
        return Err(PythonError::host(
            "PythonDiscoveryError",
            format!(
                "interpreter probe exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        ));
    }
    let discovered: DiscoveryOutput = serde_json::from_slice(&output.stdout).map_err(|error| {
        PythonError::host(
            "PythonDiscoveryError",
            format!("interpreter returned invalid discovery metadata: {error}"),
        )
    })?;
    if discovered.version.major != 3 {
        return Err(PythonError::host(
            "PythonDiscoveryError",
            format!("{} is not a CPython 3 interpreter", executable.display()),
        ));
    }
    if discovered.library.as_os_str().is_empty() || !discovered.library.is_file() {
        return Err(PythonError::host(
            "PythonDiscoveryError",
            format!(
                "{} did not report a loadable CPython library",
                discovered.executable.display()
            ),
        ));
    }
    Ok(PythonInstallation {
        version: discovered.version,
        executable: discovered.executable,
        library: discovered.library,
        home: discovered.home,
        prefix: discovered.prefix,
    })
}

fn version_matches(request: &PythonDiscoveryRequest, version: PythonVersion) -> bool {
    let pair = (version.major, version.minor);
    request.version.is_none_or(|requested| requested == pair)
        && pair >= request.minimum_version
        && request
            .maximum_version
            .is_none_or(|maximum| pair <= maximum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_an_impossible_version_range_without_launching_a_process() {
        let error = discover_python(&PythonDiscoveryRequest {
            minimum_version: (3, 12),
            maximum_version: Some((3, 11)),
            ..PythonDiscoveryRequest::default()
        })
        .unwrap_err();
        assert_eq!(error.type_name, "PythonConfigurationError");
    }

    #[test]
    fn explicit_current_interpreter_resolves_its_matching_library_when_available() {
        let Ok(default_installation) = discover_python(&PythonDiscoveryRequest::default()) else {
            return;
        };
        let installation = discover_python(&PythonDiscoveryRequest {
            executable: Some(default_installation.executable),
            ..PythonDiscoveryRequest::default()
        })
        .expect("installed CPython should report a loadable library");
        assert_eq!(installation.version.major, 3);
        assert!(installation.executable.is_file());
        assert!(installation.library.is_file());
        assert!(installation.home.is_dir());
    }
}
