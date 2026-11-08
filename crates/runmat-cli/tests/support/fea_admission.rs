//! Hermetic production-CLI authentication and the current three-route lease contract.
//! This is a local test Server, not an admission bypass in the executable.
use serde_json::Value;
#[path = "fea_admission/operations.rs"]
mod operations;
use operations::Lease;
use std::{
    collections::HashMap,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    process::Command,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

const TOKEN: &str = "fixture-local-fea-key";
const ORG: &str = "org_00000000000000000000000000000001";

pub struct AdmissionServer {
    stop: Arc<AtomicBool>,
    task: Option<thread::JoinHandle<()>>,
    journal: Arc<Mutex<Journal>>,
}

/// Close the port after `after` lease requests, as a restarting Server does, and listen
/// again on the same port once `down` has passed.
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct Outage {
    pub after: usize,
    pub down: Duration,
}

/// What the test Server saw: each lease request's route and status, every receipt, and how
/// long the port was actually closed.
#[derive(Default)]
#[allow(dead_code)]
pub struct Journal {
    pub requests: Vec<(String, &'static str)>,
    pub receipts: Vec<Value>,
    pub closed_for: Option<Duration>,
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
        Self::configure_with(command, root, None)
    }

    #[allow(dead_code)]
    pub fn journal(&self) -> std::sync::MutexGuard<'_, Journal> {
        self.journal.lock().unwrap()
    }

    pub fn configure_with(command: &mut Command, root: &Path, outage: Option<Outage>) -> Self {
        let mut listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let endpoint = format!("http://{address}");
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let journal = Arc::new(Mutex::new(Journal::default()));
        let recorded = journal.clone();
        let task = thread::spawn(move || {
            let mut leases = HashMap::<String, Lease>::new();
            let mut outage = outage;
            while !stopping.load(Ordering::Acquire) {
                if let Some(planned) =
                    outage.filter(|o| {
                    let journal = recorded.lock().unwrap();
                    let leases = journal.requests.iter();
                    leases.filter(|(path, _)| path.contains("/simulation/")).count() >= o.after
                })
                {
                    outage = None;
                    drop(listener);
                    let closed = Instant::now();
                    thread::sleep(planned.down);
                    // The closed connections can hold the port briefly; keep trying.
                    listener = loop {
                        match TcpListener::bind(address) {
                            Ok(listener) => break listener,
                            Err(_) if closed.elapsed() < planned.down * 4 => {
                                thread::sleep(Duration::from_millis(250))
                            }
                            Err(error) => panic!("test lease listener did not return: {error}"),
                        }
                    };
                    listener.set_nonblocking(true).unwrap();
                    recorded.lock().unwrap().closed_for = Some(closed.elapsed());
                }
                match listener.accept() {
                    Ok((mut socket, _)) => {
                        // Accepted sockets can inherit the listener's nonblocking mode.
                        // Request I/O uses the bounded timeouts below instead.
                        socket.set_nonblocking(false).unwrap();
                        socket
                            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                            .unwrap();
                        socket
                            .set_write_timeout(Some(std::time::Duration::from_secs(5)))
                            .unwrap();
                        let (path, status, body) = handle(&mut socket, &mut leases);
                        if let Some(path) = path {
                            let mut journal = recorded.lock().unwrap();
                            if path.ends_with("/settle") && status.starts_with("200") {
                                journal.receipts.push(body.clone());
                            }
                            journal.requests.push((path, status));
                        }
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
            journal,
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

fn handle(
    socket: &mut TcpStream,
    leases: &mut HashMap<String, Lease>,
) -> (Option<String>, &'static str, Value) {
    let Some((path, body)) = request(socket) else {
        let (status, body) = operations::invalid();
        return (None, status, body);
    };
    let (status, body) = operations::handle(&path, body, leases);
    (Some(path), status, body)
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
