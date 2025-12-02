use serde_json::Value;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use tempfile::TempDir;

fn get_binary_path() -> PathBuf {
    let mut path = std::env::current_exe().unwrap();
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.push("runmat");
    path
}

fn write_script() -> (TempDir, PathBuf) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("script.m");
    // pause gives telemetry thread time to flush before process exit
    fs::write(&path, "pause(0.5);\ndisp('telemetry');").unwrap();
    (dir, path)
}

#[test]
fn telemetry_http_events_fire_for_script_execution() {
    let (_dir, script) = write_script();
    let (endpoint, rx) = start_http_probe(2);

    let output = Command::new(get_binary_path())
        .arg(script)
        .env("RUNMAT_ACCEL_ENABLE", "0")
        .env("RUNMAT_ACCEL_PROVIDER", "inprocess")
        .env("RUNMAT_TELEMETRY", "1")
        .env("RUNMAT_TELEMETRY_KEY", "test-key")
        .env("RUNMAT_TELEMETRY_SYNC", "1")
        .env("RUNMAT_TELEMETRY_UDP_ENDPOINT", "off")
        .env("RUNMAT_TELEMETRY_ENDPOINT", &endpoint)
        .output()
        .expect("runmat execution");
    assert!(
        output.status.success(),
        "runmat failed: {}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let first = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("session payload");
    let second = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("value payload");

    assert!(contains_event(&first, "runtime_session_start"));
    assert!(contains_event(&second, "runtime_value"));
}

#[test]
fn telemetry_respects_opt_out_env() {
    let (_dir, script) = write_script();
    let (endpoint, rx) = start_http_probe(1);

    let output = Command::new(get_binary_path())
        .arg(script)
        .env("RUNMAT_ACCEL_ENABLE", "0")
        .env("RUNMAT_ACCEL_PROVIDER", "inprocess")
        .env("RUNMAT_TELEMETRY_SYNC", "1")
        .env("RUNMAT_TELEMETRY", "0")
        .env("RUNMAT_TELEMETRY_ENDPOINT", &endpoint)
        .env("RUNMAT_TELEMETRY_UDP_ENDPOINT", "off")
        .output()
        .expect("runmat execution");
    assert!(
        output.status.success(),
        "runmat failed: {}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(
        rx.recv_timeout(Duration::from_secs(2)).is_err(),
        "expected no telemetry payloads when disabled"
    );
}

fn start_http_probe(expected: usize) -> (String, mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for stream in listener.incoming().flatten().take(expected) {
            if let Some(body) = read_http_request(stream) {
                let _ = tx.send(body);
            }
        }
    });
    (format!("http://127.0.0.1:{}/ingest", addr.port()), rx)
}

fn read_http_request(stream: TcpStream) -> Option<String> {
    let mut reader = BufReader::new(stream.try_clone().ok()?);
    let mut content_length = 0usize;
    let mut line = String::new();
    loop {
        line.clear();
        reader.read_line(&mut line).ok()?;
        if line == "\r\n" {
            break;
        }
        let lower = line.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("content-length:") {
            content_length = rest.trim().parse().ok()?;
        }
    }
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body).ok()?;
    let mut writer = stream;
    let _ = writer.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK");
    Some(String::from_utf8_lossy(&body).to_string())
}

fn contains_event(body: &str, expected: &str) -> bool {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| value.get("event_label").cloned())
        .and_then(|label| label.as_str().map(|s| s == expected))
        .unwrap_or(false)
}
