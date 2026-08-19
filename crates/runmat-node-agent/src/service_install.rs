use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Serialize;

use crate::{AgentError, AgentFileConfig, AgentResult};

const SERVICE_NAME: &str = "runmat-node-agent";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ServicePlan {
    pub platform: &'static str,
    pub service_name: &'static str,
    pub files: Vec<ServiceFile>,
    pub commands: Vec<ServiceCommand>,
    pub remove_files: Vec<PathBuf>,
    pub post_remove_commands: Vec<ServiceCommand>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ServiceFile {
    pub path: PathBuf,
    pub content: String,
    pub unix_mode: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ServiceCommand {
    pub program: String,
    pub arguments: Vec<String>,
    pub ignore_failure: bool,
}

pub fn service_state_directory() -> AgentResult<PathBuf> {
    #[cfg(target_os = "linux")]
    {
        Ok(PathBuf::from("/var/lib/runmat/node-agent"))
    }
    #[cfg(target_os = "macos")]
    {
        Ok(PathBuf::from(
            "/Library/Application Support/RunMat/node-agent",
        ))
    }
    #[cfg(windows)]
    {
        std::env::var_os("ProgramData")
            .map(PathBuf::from)
            .map(|path| path.join("RunMat").join("node-agent").join("state"))
            .ok_or_else(|| AgentError::Configuration("ProgramData is unavailable".into()))
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        Err(AgentError::Configuration(
            "node service installation is unsupported on this platform".into(),
        ))
    }
}

pub fn install_plan(config: &AgentFileConfig, agent_executable: &Path) -> AgentResult<ServicePlan> {
    config.clone().into_runtime()?;
    validate_executable(agent_executable)?;
    let config_content = String::from_utf8(config.encode_pretty()?)
        .map_err(|error| AgentError::Configuration(error.to_string()))?;

    #[cfg(target_os = "linux")]
    {
        linux_install_plan(config_content, agent_executable)
    }
    #[cfg(target_os = "macos")]
    {
        macos_install_plan(config_content, agent_executable)
    }
    #[cfg(windows)]
    {
        windows_install_plan(config_content, config, agent_executable)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        let _ = (config_content, agent_executable);
        Err(AgentError::Configuration(
            "node service installation is unsupported on this platform".into(),
        ))
    }
}

pub fn uninstall_plan() -> AgentResult<ServicePlan> {
    #[cfg(target_os = "linux")]
    {
        Ok(ServicePlan {
            platform: "systemd",
            service_name: SERVICE_NAME,
            files: Vec::new(),
            commands: vec![command(
                "systemctl",
                ["disable", "--now", "runmat-node-agent.service"],
                true,
            )],
            remove_files: vec![
                PathBuf::from("/etc/systemd/system/runmat-node-agent.service"),
                PathBuf::from("/etc/runmat/node-agent.json"),
            ],
            post_remove_commands: vec![command("systemctl", ["daemon-reload"], false)],
        })
    }
    #[cfg(target_os = "macos")]
    {
        Ok(ServicePlan {
            platform: "launchd",
            service_name: SERVICE_NAME,
            files: Vec::new(),
            commands: vec![command(
                "launchctl",
                ["bootout", "system/com.runmat.node-agent"],
                true,
            )],
            remove_files: vec![
                PathBuf::from("/Library/LaunchDaemons/com.runmat.node-agent.plist"),
                PathBuf::from("/Library/Application Support/RunMat/node-agent.json"),
            ],
            post_remove_commands: Vec::new(),
        })
    }
    #[cfg(windows)]
    {
        let config_path = windows_config_path()?;
        Ok(ServicePlan {
            platform: "windows-service",
            service_name: SERVICE_NAME,
            files: Vec::new(),
            commands: vec![
                command("sc.exe", ["stop", "RunMatNodeAgent"], true),
                command("sc.exe", ["delete", "RunMatNodeAgent"], true),
            ],
            remove_files: vec![config_path],
            post_remove_commands: Vec::new(),
        })
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        Err(AgentError::Configuration(
            "node service installation is unsupported on this platform".into(),
        ))
    }
}

pub fn apply_install(plan: &ServicePlan, state_directory: &Path) -> AgentResult<()> {
    create_state_directory(state_directory)?;
    for file in &plan.files {
        write_atomic(file)?;
    }
    execute_commands(&plan.commands)
}

pub fn apply_uninstall(plan: &ServicePlan) -> AgentResult<()> {
    execute_commands(&plan.commands)?;
    for path in &plan.remove_files {
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    execute_commands(&plan.post_remove_commands)
}

fn validate_executable(path: &Path) -> AgentResult<()> {
    if !path.is_absolute() || !path.is_file() {
        return Err(AgentError::Configuration(
            "RunMat executable must be an existing absolute file".into(),
        ));
    }
    Ok(())
}

fn write_atomic(file: &ServiceFile) -> AgentResult<()> {
    let parent = file
        .path
        .parent()
        .ok_or_else(|| AgentError::Configuration("service file has no parent".into()))?;
    std::fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(".runmat-service-{}.tmp", rand::random::<u64>()));
    let mut options = std::fs::OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(file.unix_mode.unwrap_or(0o644));
    }
    let mut output = options.open(&temporary)?;
    use std::io::Write as _;
    if let Err(error) = output
        .write_all(file.content.as_bytes())
        .and_then(|()| output.sync_all())
    {
        let _ = std::fs::remove_file(&temporary);
        return Err(error.into());
    }
    drop(output);
    replace_file(&temporary, &file.path)?;
    Ok(())
}

#[cfg(not(windows))]
fn replace_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::rename(source, destination)
}

#[cfg(windows)]
fn replace_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt as _;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let source = source
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let destination = destination
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let replaced = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if replaced == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn create_state_directory(path: &Path) -> AgentResult<()> {
    std::fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn execute_commands(commands: &[ServiceCommand]) -> AgentResult<()> {
    for planned in commands {
        let status = Command::new(&planned.program)
            .args(&planned.arguments)
            .status()?;
        if !status.success() && !planned.ignore_failure {
            return Err(AgentError::Configuration(format!(
                "{} failed with {status}",
                planned.program
            )));
        }
    }
    Ok(())
}

fn command<const N: usize>(
    program: &str,
    arguments: [&str; N],
    ignore_failure: bool,
) -> ServiceCommand {
    ServiceCommand {
        program: program.into(),
        arguments: arguments.into_iter().map(str::to_string).collect(),
        ignore_failure,
    }
}

#[cfg(target_os = "linux")]
fn linux_install_plan(config_content: String, agent_executable: &Path) -> AgentResult<ServicePlan> {
    let config_path = PathBuf::from("/etc/runmat/node-agent.json");
    let executable = systemd_quote(agent_executable)?;
    let config = systemd_quote(&config_path)?;
    let unit = format!(
        "[Unit]\nDescription=RunMat execution node agent\nAfter=network-online.target\nWants=network-online.target\n\n[Service]\nType=simple\nExecStart={executable} cluster join --node-config {config} run\nRestart=on-failure\nRestartSec=5s\nNoNewPrivileges=true\nPrivateTmp=true\nProtectHome=true\nProtectSystem=strict\nReadWritePaths=/var/lib/runmat/node-agent\nLockPersonality=true\nRestrictSUIDSGID=true\n\n[Install]\nWantedBy=multi-user.target\n"
    );
    Ok(ServicePlan {
        platform: "systemd",
        service_name: SERVICE_NAME,
        files: vec![
            ServiceFile {
                path: config_path,
                content: config_content,
                unix_mode: Some(0o644),
            },
            ServiceFile {
                path: PathBuf::from("/etc/systemd/system/runmat-node-agent.service"),
                content: unit,
                unix_mode: Some(0o644),
            },
        ],
        commands: vec![
            command("systemctl", ["daemon-reload"], false),
            command(
                "systemctl",
                ["enable", "--now", "runmat-node-agent.service"],
                false,
            ),
        ],
        remove_files: Vec::new(),
        post_remove_commands: Vec::new(),
    })
}

#[cfg(target_os = "linux")]
fn systemd_quote(path: &Path) -> AgentResult<String> {
    let text = path_text(path)?;
    Ok(format!(
        "\"{}\"",
        text.replace('\\', "\\\\").replace('"', "\\\"")
    ))
}

#[cfg(target_os = "macos")]
fn macos_install_plan(config_content: String, agent_executable: &Path) -> AgentResult<ServicePlan> {
    let config_path = PathBuf::from("/Library/Application Support/RunMat/node-agent.json");
    let executable = xml_escape(&path_text(agent_executable)?);
    let config = xml_escape(&path_text(&config_path)?);
    let plist = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\">\n<dict>\n  <key>Label</key><string>com.runmat.node-agent</string>\n  <key>ProgramArguments</key>\n  <array><string>{executable}</string><string>cluster</string><string>join</string><string>--node-config</string><string>{config}</string><string>run</string></array>\n  <key>RunAtLoad</key><true/>\n  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>\n  <key>ProcessType</key><string>Background</string>\n</dict>\n</plist>\n"
    );
    Ok(ServicePlan {
        platform: "launchd",
        service_name: SERVICE_NAME,
        files: vec![
            ServiceFile {
                path: config_path,
                content: config_content,
                unix_mode: Some(0o644),
            },
            ServiceFile {
                path: PathBuf::from("/Library/LaunchDaemons/com.runmat.node-agent.plist"),
                content: plist,
                unix_mode: Some(0o644),
            },
        ],
        commands: vec![
            command(
                "launchctl",
                ["bootout", "system/com.runmat.node-agent"],
                true,
            ),
            command(
                "launchctl",
                [
                    "bootstrap",
                    "system",
                    "/Library/LaunchDaemons/com.runmat.node-agent.plist",
                ],
                false,
            ),
            command(
                "launchctl",
                ["enable", "system/com.runmat.node-agent"],
                false,
            ),
        ],
        remove_files: Vec::new(),
        post_remove_commands: Vec::new(),
    })
}

#[cfg(target_os = "macos")]
fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(windows)]
fn windows_install_plan(
    config_content: String,
    config: &AgentFileConfig,
    agent_executable: &Path,
) -> AgentResult<ServicePlan> {
    let config_path = windows_config_path()?;
    let binary_path = format!(
        "\"{}\" cluster join --node-config \"{}\" windows-service-run",
        path_text(agent_executable)?,
        path_text(&config_path)?
    );
    let state = path_text(&config.state_directory)?;
    Ok(ServicePlan {
        platform: "windows-service",
        service_name: SERVICE_NAME,
        files: vec![ServiceFile {
            path: config_path,
            content: config_content,
            unix_mode: None,
        }],
        commands: vec![
            command("sc.exe", ["stop", "RunMatNodeAgent"], true),
            command("sc.exe", ["delete", "RunMatNodeAgent"], true),
            ServiceCommand {
                program: "sc.exe".into(),
                arguments: vec![
                    "create".into(),
                    "RunMatNodeAgent".into(),
                    "binPath=".into(),
                    binary_path,
                    "start=".into(),
                    "auto".into(),
                    "DisplayName=".into(),
                    "RunMat Execution Node Agent".into(),
                ],
                ignore_failure: false,
            },
            ServiceCommand {
                program: "icacls.exe".into(),
                arguments: vec![
                    state,
                    "/inheritance:r".into(),
                    "/grant:r".into(),
                    "SYSTEM:(OI)(CI)F".into(),
                    "Administrators:(OI)(CI)F".into(),
                ],
                ignore_failure: false,
            },
            command("sc.exe", ["start", "RunMatNodeAgent"], false),
        ],
        remove_files: Vec::new(),
        post_remove_commands: Vec::new(),
    })
}

#[cfg(windows)]
fn windows_config_path() -> AgentResult<PathBuf> {
    std::env::var_os("ProgramData")
        .map(PathBuf::from)
        .map(|path| path.join("RunMat").join("node-agent.json"))
        .ok_or_else(|| AgentError::Configuration("ProgramData is unavailable".into()))
}

fn path_text(path: &Path) -> AgentResult<String> {
    let text = path
        .to_str()
        .ok_or_else(|| AgentError::Configuration("service path is not valid Unicode".into()))?;
    if text.is_empty() || text.chars().any(char::is_control) {
        return Err(AgentError::Configuration(
            "service path is empty or contains control characters".into(),
        ));
    }
    Ok(text.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_plan_contains_only_validated_absolute_paths() {
        let directory = tempfile::tempdir().unwrap();
        let executable = directory.path().join("runmat");
        std::fs::write(&executable, b"binary").unwrap();
        let runmat = directory.path().join("runmat");
        std::fs::write(&runmat, b"binary").unwrap();
        let config = AgentFileConfig {
            server_url: "https://api.runmat.com".into(),
            runmat_executable: runmat,
            state_directory: service_state_directory().unwrap(),
            heartbeat_interval_seconds: 15,
            heartbeat_ttl_seconds: 60,
            drain_timeout_seconds: 30,
            maximum_allocations: 1,
            trust_tier: runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
        };
        let plan = install_plan(&config, &executable).unwrap();
        assert!(!plan.files.is_empty());
        assert!(!plan.commands.is_empty());
        assert!(plan.files.iter().all(|file| file.path.is_absolute()));
        let rendered = serde_json::to_string(&plan).unwrap();
        assert!(rendered.contains(SERVICE_NAME));
        assert!(rendered.contains("cluster"));
        assert!(rendered.contains("join"));
        assert!(rendered.contains("runmat"));
        assert!(!rendered.contains("runmat-node-agent --"));
        assert!(!rendered.contains("credential"));
    }
}
