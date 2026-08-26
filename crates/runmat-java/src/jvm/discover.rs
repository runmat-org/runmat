use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::{JvmError, JvmVersion};

#[derive(Debug, Clone, Default)]
pub struct JavaDiscoveryRequest {
    pub explicit_home: Option<PathBuf>,
    pub environment_home: Option<PathBuf>,
    pub java_executable: Option<PathBuf>,
    pub search_system: bool,
}

impl JavaDiscoveryRequest {
    pub fn from_process() -> Self {
        Self {
            explicit_home: None,
            environment_home: std::env::var_os("JAVA_HOME").map(PathBuf::from),
            java_executable: None,
            search_system: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JvmInstallation {
    pub home: PathBuf,
    pub library: PathBuf,
    pub java_executable: PathBuf,
    pub version: JvmVersion,
}

pub fn discover_jvm(request: &JavaDiscoveryRequest) -> Result<JvmInstallation, JvmError> {
    let mut candidates = Vec::new();
    if let Some(home) = &request.explicit_home {
        candidates.push(home.clone());
    }
    if let Some(home) = &request.environment_home {
        candidates.push(home.clone());
    }
    if let Some(executable) = &request.java_executable {
        if let Some(home) = query_java_home(executable) {
            candidates.push(home);
        }
    }
    if request.search_system {
        if let Some(executable) = find_executable("java") {
            if let Some(home) = query_java_home(&executable) {
                candidates.push(home);
            }
        }
        candidates.extend(platform_homes());
    }

    let mut visited = BTreeSet::new();
    let mut failures = Vec::new();
    for candidate in candidates {
        let candidate = normalize_home(candidate);
        if !visited.insert(candidate.clone()) {
            continue;
        }
        match inspect_home(&candidate) {
            Ok(installation) => return Ok(installation),
            Err(error) => failures.push(error.to_string()),
        }
    }
    let detail = if failures.is_empty() {
        String::new()
    } else {
        format!(": {}", failures.join("; "))
    };
    Err(JvmError::NotFound { detail })
}

fn inspect_home(home: &Path) -> Result<JvmInstallation, JvmError> {
    let library = jvm_library_candidates(home)
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| JvmError::InvalidInstallation {
            path: home.display().to_string(),
            reason: "JVM shared library is missing".into(),
        })?;
    let java_executable = java_executable(home);
    if !java_executable.is_file() {
        return Err(JvmError::InvalidInstallation {
            path: home.display().to_string(),
            reason: "Java executable is missing".into(),
        });
    }
    let version = read_release_version(home)
        .or_else(|| query_java_version(&java_executable))
        .ok_or_else(|| JvmError::InvalidInstallation {
            path: home.display().to_string(),
            reason: "Java version could not be determined".into(),
        })?;
    Ok(JvmInstallation {
        home: home.to_path_buf(),
        library,
        java_executable,
        version,
    })
}

fn normalize_home(home: PathBuf) -> PathBuf {
    let nested = home.join("Contents/Home");
    if nested.is_dir() {
        nested
    } else if home.file_name().is_some_and(|name| name == "jre") {
        home.parent().unwrap_or(&home).to_path_buf()
    } else {
        home
    }
}

fn jvm_library_candidates(home: &Path) -> Vec<PathBuf> {
    let names: &[&str] = if cfg!(target_os = "windows") {
        &["bin/server/jvm.dll", "jre/bin/server/jvm.dll"]
    } else if cfg!(target_os = "macos") {
        &["lib/server/libjvm.dylib", "jre/lib/server/libjvm.dylib"]
    } else {
        &[
            "lib/server/libjvm.so",
            "jre/lib/server/libjvm.so",
            "jre/lib/amd64/server/libjvm.so",
            "lib/amd64/server/libjvm.so",
            "jre/lib/aarch64/server/libjvm.so",
            "lib/aarch64/server/libjvm.so",
        ]
    };
    names.iter().map(|name| home.join(name)).collect()
}

fn java_executable(home: &Path) -> PathBuf {
    home.join("bin").join(if cfg!(target_os = "windows") {
        "java.exe"
    } else {
        "java"
    })
}

fn read_release_version(home: &Path) -> Option<JvmVersion> {
    let release = fs::read_to_string(home.join("release")).ok()?;
    let raw = release.lines().find_map(|line| {
        line.strip_prefix("JAVA_VERSION=")
            .map(|value| value.trim().trim_matches('"').to_string())
    })?;
    JvmVersion::parse(raw).ok()
}

fn query_java_home(executable: &Path) -> Option<PathBuf> {
    let output = Command::new(executable)
        .args(["-XshowSettings:properties", "-version"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&output.stderr)
        .lines()
        .chain(String::from_utf8_lossy(&output.stdout).lines())
        .find_map(|line| {
            line.trim()
                .strip_prefix("java.home =")
                .map(|value| PathBuf::from(value.trim()))
        })
}

fn query_java_version(executable: &Path) -> Option<JvmVersion> {
    let output = Command::new(executable).arg("-version").output().ok()?;
    let text = String::from_utf8_lossy(&output.stderr);
    let quoted = text.lines().next()?.split('"').nth(1)?;
    JvmVersion::parse(quoted).ok()
}

fn find_executable(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .flat_map(|directory| {
            if cfg!(target_os = "windows") {
                vec![directory.join(format!("{name}.exe")), directory.join(name)]
            } else {
                vec![directory.join(name)]
            }
        })
        .find(|candidate| candidate.is_file())
}

fn platform_homes() -> Vec<PathBuf> {
    let mut homes = Vec::new();
    if cfg!(target_os = "macos") {
        if let Ok(output) = Command::new("/usr/libexec/java_home").output() {
            if output.status.success() {
                homes.push(PathBuf::from(
                    String::from_utf8_lossy(&output.stdout).trim(),
                ));
            }
        }
        homes.extend(children_of(Path::new("/Library/Java/JavaVirtualMachines")));
    } else if cfg!(target_os = "windows") {
        for variable in ["ProgramFiles", "ProgramFiles(x86)"] {
            if let Some(root) = std::env::var_os(variable) {
                homes.extend(children_of(&PathBuf::from(root).join("Java")));
            }
        }
    } else {
        homes.extend(children_of(Path::new("/usr/lib/jvm")));
    }
    homes.sort();
    homes.reverse();
    homes
}

fn children_of(root: &Path) -> Vec<PathBuf> {
    fs::read_dir(root)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_home_is_deterministic() {
        let directory = tempfile::tempdir().unwrap();
        let home = directory.path();
        fs::create_dir_all(home.join("bin")).unwrap();
        let java = java_executable(home);
        fs::write(&java, []).unwrap();
        let library = jvm_library_candidates(home).remove(0);
        fs::create_dir_all(library.parent().unwrap()).unwrap();
        fs::write(&library, []).unwrap();
        fs::write(home.join("release"), "JAVA_VERSION=\"21.0.4\"\n").unwrap();

        let installation = discover_jvm(&JavaDiscoveryRequest {
            explicit_home: Some(home.to_path_buf()),
            search_system: false,
            ..JavaDiscoveryRequest::default()
        })
        .unwrap();
        assert_eq!(installation.home, home);
        assert_eq!(installation.version.major, 21);
    }

    #[test]
    fn missing_home_reports_an_actionable_failure() {
        let directory = tempfile::tempdir().unwrap();
        let error = discover_jvm(&JavaDiscoveryRequest {
            explicit_home: Some(directory.path().to_path_buf()),
            search_system: false,
            ..JavaDiscoveryRequest::default()
        })
        .unwrap_err();
        assert!(error.to_string().contains("JVM shared library is missing"));
    }
}
