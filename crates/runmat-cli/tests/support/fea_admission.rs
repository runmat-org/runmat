//! One real signed-lease HTTP fixture for this owning integration-test process.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ed25519_dalek::{Signer, SigningKey};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    process::Command,
    sync::OnceLock,
    time::Duration,
};

struct Fixture {
    origin: String,
}
static FIXTURE: OnceLock<Fixture> = OnceLock::new();
const CREDENTIAL: &str = "fixture:cli-check-only";
const ORG: &str = "00000000-0000-0000-0000-000000000042";

pub fn configure(command: &mut Command, root: &Path) {
    let fixture = FIXTURE.get_or_init(|| {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        std::thread::Builder::new().name("cli-admission-fixture".into()).spawn(move || {
            let key = SigningKey::from_bytes(&[23; 32]);
            let mut leases = HashMap::<String, (Value, Value)>::new();
            for connection in listener.incoming() {
                let mut stream = connection.expect("accept CLI fixture request");
                stream.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
                let (status, body) = serve(&mut stream, &key, &mut leases);
                let bytes = serde_json::to_vec(&body).unwrap();
                write!(stream, "HTTP/1.1 {status} fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", bytes.len()).unwrap();
                stream.write_all(&bytes).unwrap();
            }
        }).unwrap();
        Fixture { origin }
    });
    // Every spawned CLI (including a builtin's child CLI) uses only this test's
    // file-backed config and fake credential. Never inspect an operator keychain.
    for (key, _) in
        std::env::vars_os().filter(|(key, _)| key.to_string_lossy().starts_with("RUNMAT_"))
    {
        command.env_remove(key);
    }
    let auth = root.join("cli-auth");
    fs::create_dir_all(&auth).unwrap();
    fs::write(
        auth.join("remote.json"),
        serde_json::to_vec(&json!({
            "server_url": fixture.origin, "org_id": ORG, "credential_store": "file",
        }))
        .unwrap(),
    )
    .unwrap();
    command
        .env("RUNMAT_CLI_CONFIG_DIR", auth)
        .env("RUNMAT_API_KEY", CREDENTIAL)
        .env("RUNMAT_SERVER_URL", &fixture.origin)
        .env("RUNMAT_ORG_ID", ORG)
        .env("RUNMAT_PACKAGE_CACHE_DIR", root.join("package-cache"))
        .env("RUNMAT_EXECUTION_STATE_DIR", root.join("execution-state"))
        .env("RUNMAT_FEA_ARTIFACT_ROOT", root.join("artifacts"))
        .env(
            "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
            root.join("study-artifacts"),
        );
}

fn serve(
    stream: &mut TcpStream,
    key: &SigningKey,
    leases: &mut HashMap<String, (Value, Value)>,
) -> (u16, Value) {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let first = line.clone();
    let words: Vec<_> = first.split_whitespace().collect();
    let (method, path) = (words[0], words[1]);
    let mut length = 0;
    let mut authorized = false;
    loop {
        line.clear();
        reader.read_line(&mut line).unwrap();
        if line == "\r\n" {
            break;
        }
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.eq_ignore_ascii_case("content-length") {
            length = value.trim().parse().unwrap();
        }
        if name.eq_ignore_ascii_case("authorization") {
            authorized = value.trim() == format!("Bearer {CREDENTIAL}");
        }
    }
    if method == "GET" && path == "/v1/execution/core-hour-lease-keys" {
        return (
            200,
            json!({"keys":[{"keyId":"cli-test","algorithm":"Ed25519","publicKey":URL_SAFE_NO_PAD.encode(key.verifying_key().as_bytes())}]}),
        );
    }
    if !authorized {
        return (401, json!({"error":{"code":"auth.invalid_token"}}));
    }
    let mut bytes = vec![0; length];
    reader.read_exact(&mut bytes).unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    let parts: Vec<_> = path.trim_start_matches('/').split('/').collect();
    if method != "POST"
        || parts.len() < 5
        || parts[..2] != ["v1", "orgs"]
        || parts[3..5] != ["execution", "core-hour-leases"]
    {
        return (404, json!({"error":{"code":"fixture.unexpected_route"}}));
    }
    let org = parts[2];
    if parts.len() == 5 {
        assert_eq!(body["clientSurface"], "cli");
        let run = body["idempotencyKey"].as_str().unwrap();
        if let Some((previous, lease)) = leases.get(run) {
            return if previous == &body && lease["orgId"] == org {
                (200, lease.clone())
            } else {
                (
                    409,
                    json!({"error":{"code":"core_hour_lease.idempotency_conflict"}}),
                )
            };
        }
        let allocation = body["allocatedCpuMillicores"].as_u64().unwrap();
        assert!((1..=4_096_000).contains(&allocation));
        assert!(matches!(
            body["kind"].as_str(),
            Some("meshing" | "solve" | "meshAndSolve")
        ));
        let estimate = body["estimatedWallSeconds"].as_u64().unwrap_or(900);
        let issued = chrono::Utc::now();
        let expiry = issued + chrono::Duration::seconds(estimate as i64 + 86_400);
        let lease_id = format!("chlease_{}", uuid::Uuid::new_v4().simple());
        let reservation = (allocation * estimate * 5).div_ceil(18);
        let claims = json!({"audience":"runmat.fea.core_hours","leaseId":lease_id,"orgId":org,
            "userId":"usr_cli_fixture","idempotencyKey":run,"allocatedCpuMillicores":allocation,
            "coreMicroHours":reservation,"issuedAt":issued.timestamp(),"expiresAt":expiry.timestamp()});
        let signed = format!(
            "ed25519v1.cli-test.{}",
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims).unwrap())
        );
        let token = format!(
            "{signed}.{}",
            URL_SAFE_NO_PAD.encode(key.sign(signed.as_bytes()).to_bytes())
        );
        let lease = json!({"leaseId":lease_id,"orgId":org,"userId":"usr_cli_fixture","idempotencyKey":run,
            "status":"active","allocatedCpuMillicores":allocation,"reservedCoreMicroHours":reservation,
            "usedCoreMicroHours":0,"chargedCoreMicroHours":null,"overageCoreMicroHours":0,"outcome":null,
            "issuedAt":issued.to_rfc3339(),"expiresAt":expiry.to_rfc3339(),"refillsAt":null,"token":token});
        leases.insert(run.into(), (body, lease.clone()));
        return (201, lease);
    }
    if parts.len() == 7 {
        let Some((_, lease)) = leases
            .values_mut()
            .find(|(_, lease)| lease["leaseId"] == parts[5] && lease["orgId"] == org)
        else {
            return (404, json!({"error":{"code":"core_hour_lease.not_found"}}));
        };
        if lease["status"] == "active" {
            let used = (body["cpuCoreMillis"].as_u64().unwrap() * 5).div_ceil(18);
            lease["usedCoreMicroHours"] =
                json!(used.max(lease["usedCoreMicroHours"].as_u64().unwrap()));
            if parts[6] == "settle" {
                lease["outcome"] = body["outcome"].clone();
                lease["status"] = json!(if body["outcome"] == "failed" {
                    "released"
                } else {
                    "settled"
                });
                lease["chargedCoreMicroHours"] = if body["outcome"] == "failed" {
                    json!(0)
                } else {
                    lease["usedCoreMicroHours"].clone()
                };
            } else {
                assert_eq!(parts[6], "heartbeat");
            }
        }
        let mut response = lease.clone();
        response.as_object_mut().unwrap().remove("token");
        return (200, response);
    }
    (404, json!({"error":{"code":"fixture.unexpected_route"}}))
}
