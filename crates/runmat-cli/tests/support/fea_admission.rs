//! Hermetic production-CLI authentication and the current three-route lease contract.
//! This is a local test Server, not an admission bypass in the executable.
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    process::Command,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
};

const TOKEN: &str = "fixture-local-fea-key";
const ORG: &str = "org_00000000000000000000000000000001";

pub struct AdmissionServer {
    stop: Arc<AtomicBool>,
    task: Option<thread::JoinHandle<()>>,
}
struct Lease {
    run_id: String,
    allocation: u64,
    held: u64,
    cpu: u64,
    wall: u64,
    charged: Option<u64>,
}

impl AdmissionServer {
    /// Configure a command only when its public entry point actually runs FEA.
    /// Checks, plans, result readers and plain scripts retain signed-out fixtures.
    pub fn for_command(command: &mut Command, root: &Path) -> Option<Self> {
        if !executes_fea(command) {
            return None;
        }
        Some(Self::configure(command, root))
    }

    pub fn configure(command: &mut Command, root: &Path) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let task = thread::spawn(move || {
            let mut leases = HashMap::<String, Lease>::new();
            while !stopping.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((mut socket, _)) => {
                        // On macOS an accepted socket inherits the listener's nonblocking
                        // mode; reads must block (up to the timeout) until the request lands.
                        socket.set_nonblocking(false).unwrap();
                        socket
                            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                            .unwrap();
                        socket
                            .set_write_timeout(Some(std::time::Duration::from_secs(5)))
                            .unwrap();
                        let (status, body) = handle(&mut socket, &mut leases);
                        let body = serde_json::to_vec(&body).unwrap();
                        let header = format!("HTTP/1.1 {status}\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n", body.len());
                        let _ = socket.write_all(header.as_bytes());
                        let _ = socket.write_all(&body);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(std::time::Duration::from_millis(5))
                    }
                    Err(error) => panic!("test lease listener failed: {error}"),
                }
            }
        });
        let server = Self {
            stop,
            task: Some(task),
        };
        // Exercise actual file-credential login, with an isolated default org.
        let mut login = Command::new(env!("CARGO_BIN_EXE_runmat"));
        super::local_offline::configure(&mut login, root);
        let output = login
            .current_dir(root)
            .args([
                "login",
                "--server",
                &endpoint,
                "--api-key",
                TOKEN,
                "--org",
                ORG,
                "--credential-store",
                "file",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "fixture login failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        // Use the saved authenticated session/default org, never inherited keys.
        command
            .env_remove("RUNMAT_API_KEY")
            .env_remove("RUNMAT_ORG_ID")
            .env_remove("RUNMAT_SERVER_URL")
            .env("RUNMAT_CLI_CONFIG_DIR", root.join("signed-out-cli"));
        server
    }
}
impl Drop for AdmissionServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(task) = self.task.take() {
            let result = task.join();
            if !thread::panicking() {
                result.unwrap();
            }
        }
    }
}

fn executes_fea(command: &Command) -> bool {
    let args = command
        .get_args()
        .map(|arg| arg.to_string_lossy())
        .collect::<Vec<_>>();
    if args
        .windows(2)
        .any(|pair| pair[0] == "fea" && matches!(pair[1].as_ref(), "run" | "snapshot"))
        || args.first().is_some_and(|arg| arg == "mesh")
    {
        return true;
    }
    if args
        .iter()
        .any(|arg| matches!(arg.as_ref(), "check" | "plan" | "compile"))
    {
        return false;
    }
    let directory = command.get_current_dir().unwrap_or_else(|| Path::new("."));
    args.iter().filter(|arg| !arg.starts_with('-')).any(|arg| {
        let path = directory.join(arg.as_ref());
        if path.extension().is_some_and(|ext| ext == "fea") {
            return true;
        }
        path.extension().is_some_and(|ext| ext == "m")
            && std::fs::read_to_string(path).is_ok_and(|source| {
                source.contains("fea.run(")
                    || source.contains("fea.run_")
                    || source.contains("meshing.generate(")
            })
    })
}

fn error(status: &'static str, code: &str) -> (&'static str, Value) {
    (status, json!({"error":{"code":code}}))
}
fn handle(socket: &mut TcpStream, leases: &mut HashMap<String, Lease>) -> (&'static str, Value) {
    let Some((path, body)) = request(socket) else {
        return error("400 Bad Request", "fea.invalid_request");
    };
    let expiry = || (chrono::Utc::now() + chrono::Duration::minutes(2)).to_rfc3339();
    if path == format!("/v1/orgs/{ORG}/fea/leases") {
        let Some(run_id) = body["runId"].as_str() else {
            return error("422 Unprocessable Entity", "fea.invalid_request");
        };
        let Some(allocation) = body["allocatedCpuMillicores"]
            .as_u64()
            .filter(|n| (1..=4_096_000).contains(n))
        else {
            return error("422 Unprocessable Entity", "fea.invalid_request");
        };
        let id = leases
            .iter()
            .find(|(_, lease)| lease.run_id == run_id)
            .map(|(id, _)| id.clone())
            .unwrap_or_else(|| {
                format!("fealease_00000000-0000-0000-0000-{:012x}", leases.len() + 1)
            });
        assert!(leases.len() < 128);
        let lease = leases.entry(id.clone()).or_insert_with(|| Lease {
            run_id: run_id.into(),
            allocation,
            held: allocation * 120,
            cpu: 0,
            wall: 0,
            charged: None,
        });
        assert_eq!(
            lease.allocation, allocation,
            "idempotent start changed allocation"
        );
        return (
            "200 OK",
            json!({"leaseId":id,"heldCoreMillis":lease.held,"expiresAt":expiry(),"heartbeatSeconds":30}),
        );
    }
    let Some(tail) = path.strip_prefix("/v1/fea/leases/") else {
        return error("404 Not Found", "fea.lease_not_found");
    };
    let Some((id, operation)) = tail.split_once('/') else {
        return error("404 Not Found", "fea.lease_not_found");
    };
    let Some(lease) = leases.get_mut(id) else {
        return error("404 Not Found", "fea.lease_not_found");
    };
    let (Some(cpu), Some(wall)) = (body["cpuCoreMillis"].as_u64(), body["wallMillis"].as_u64())
    else {
        return error("422 Unprocessable Entity", "fea.invalid_request");
    };
    assert!(
        cpu >= lease.cpu && wall >= lease.wall,
        "usage must be cumulative"
    );
    assert_eq!(
        u128::from(cpu),
        (u128::from(lease.allocation) * u128::from(wall)).div_ceil(1000),
        "usage must report allocated CPU times original wall time"
    );
    lease.cpu = cpu;
    lease.wall = wall;
    match operation {
        "usage" if lease.charged.is_none() => {
            lease.held = lease.held.max(cpu + lease.allocation * 120);
            (
                "200 OK",
                json!({"heldCoreMillis":lease.held,"expiresAt":expiry()}),
            )
        }
        "settle"
            if matches!(
                body["outcome"].as_str(),
                Some("completed" | "cancelled" | "failed")
            ) =>
        {
            let charged = *lease.charged.get_or_insert(cpu.min(lease.held));
            ("200 OK", json!({"chargedCoreMillis":charged}))
        }
        _ => error("404 Not Found", "fea.lease_not_found"),
    }
}
fn request(socket: &mut TcpStream) -> Option<(String, Value)> {
    let mut bytes = Vec::new();
    let end = loop {
        let mut chunk = [0; 1024];
        let count = socket.read(&mut chunk).ok()?;
        if count == 0 {
            return None;
        }
        bytes.extend_from_slice(&chunk[..count]);
        if bytes.len() > 16 * 1024 {
            return None;
        }
        if let Some(end) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
            break end + 4;
        }
    };
    let header = std::str::from_utf8(&bytes[..end]).ok()?;
    let first = header.lines().next()?;
    let path = first
        .strip_prefix("POST ")?
        .strip_suffix(" HTTP/1.1")?
        .to_owned();
    assert!(
        header
            .lines()
            .any(|line| line.eq_ignore_ascii_case(&format!("authorization: Bearer {TOKEN}"))),
        "fixture request must use isolated saved credentials"
    );
    let length: usize = header
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(|value| value.trim().parse().ok())
        })
        .flatten()?;
    if length > 8192 {
        return None;
    }
    while bytes.len() < end + length {
        let mut chunk = [0; 1024];
        let count = socket.read(&mut chunk).ok()?;
        if count == 0 {
            return None;
        }
        bytes.extend_from_slice(&chunk[..count]);
    }
    Some((
        path,
        serde_json::from_slice(&bytes[end..end + length]).ok()?,
    ))
}
