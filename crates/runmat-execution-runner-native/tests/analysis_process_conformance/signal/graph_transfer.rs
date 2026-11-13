//! Serve real sealed Program chunks to the existing authenticated QUIC staging
//! route. No raw executable or fabricated worker receipt crosses a shortcut.
use super::*;
use runmat_execution_artifact::{encryption::*, GraphExecutionRequest};
use runmat_execution_runner_native::RemoteGraphTransfer;
use runmat_execution_transport_native::control::DriverArtifactDownload;
use std::collections::BTreeMap;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub(super) struct Transfer {
    pub admitted: Option<RemoteGraphTransfer>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    server: Option<tokio::task::JoinHandle<()>>,
}
impl Drop for Transfer {
    fn drop(&mut self) {
        if let Some(server) = self.server.take() {
            server.abort();
        }
    }
}
impl Transfer {
    pub async fn finish(mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(server) = self.server.take() {
            server.await.unwrap();
        }
    }
}

pub(super) async fn prepare(request: &GraphExecutionRequest, key: &RunKeyMaterial) -> Transfer {
    let maximum = 64 * 1024 * 1024;
    // RemoteProgramObject::Graph is untagged: these are its exact JSON bytes.
    let bytes = runmat_canonical_codec::encode_json_bounded(request, maximum).unwrap();
    let digest = Digest::sha256(&bytes);
    let length = bytes.len() as u64;
    let count = bytes.len().div_ceil(MAX_PROGRAM_CHUNK_PLAINTEXT_BYTES);
    let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .unwrap();
    let address = listener.local_addr().unwrap();
    let mut objects = BTreeMap::new();
    let mut entries = Vec::new();
    let mut downloads = Vec::new();
    for (index, chunk) in bytes.chunks(MAX_PROGRAM_CHUNK_PLAINTEXT_BYTES).enumerate() {
        let authority = ProgramChunkAuthority {
            run_identity: "signal-conformance".into(),
            program_digest: digest,
            program_length: length,
            chunk_index: index as u64,
            chunk_count: count as u64,
            chunk_length: u32::try_from(chunk.len()).unwrap(),
            chunk_digest: Digest::sha256(chunk),
            key_epoch: 1,
        };
        let mut salt = [0x51; 32];
        salt[24..].copy_from_slice(&(index as u64).to_le_bytes());
        let encoded =
            encode_encrypted_run_object(&seal_program_chunk(key, salt, &authority, chunk).unwrap())
                .unwrap();
        let encoded_digest = Digest::sha256(&encoded);
        let encoded_length = encoded.len() as u64;
        let id = format!("signal-program-{digest}-{index}");
        let path = format!("/{id}");
        entries.push(ProgramChunkObject {
            artifact_id: id.clone(),
            plaintext_digest: authority.chunk_digest,
            plaintext_length: authority.chunk_length,
            encoded_object_digest: encoded_digest,
            encoded_object_length: encoded_length,
        });
        downloads.push(DriverArtifactDownload {
            artifact_id: id,
            kind: "program-chunk".into(),
            ciphertext_digest: encoded_digest.to_string(),
            ciphertext_size_bytes: encoded_length,
            media_type: "application/cbor".into(),
            method: "GET".into(),
            url: format!("http://{address}{path}"),
            headers: BTreeMap::new(),
        });
        objects.insert(path, encoded);
    }
    drop(bytes);
    let manifest = ProgramChunkManifest {
        schema_version: 1,
        program_digest: digest,
        program_length: length,
        chunks: entries,
    };
    let authority = ProgramManifestAuthority {
        run_identity: "signal-conformance".into(),
        key_epoch: 1,
    };
    let encoded = encode_encrypted_run_object(
        &seal_program_chunk_manifest(key, [0xa7; 32], &authority, &manifest, maximum).unwrap(),
    )
    .unwrap();
    let admitted =
        RemoteGraphTransfer::admit(request, encoded, downloads, maximum, key, &authority).unwrap();
    let (stop, mut stopped) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        loop {
            let (stream, _) =
                tokio::select! {_=&mut stopped=>break,result=listener.accept()=>result.unwrap()};
            tokio::time::timeout(Duration::from_secs(5), serve(stream, &objects))
                .await
                .unwrap();
        }
    });
    Transfer {
        admitted: Some(admitted),
        stop: Some(stop),
        server: Some(server),
    }
}

async fn serve(mut stream: tokio::net::TcpStream, objects: &BTreeMap<String, Vec<u8>>) {
    let mut header = Vec::new();
    let mut buffer = [0u8; 1024];
    while !header.windows(4).any(|value| value == b"\r\n\r\n") {
        let count = stream.read(&mut buffer).await.unwrap();
        if count == 0 || header.len() + count > 16 * 1024 {
            return;
        }
        header.extend_from_slice(&buffer[..count]);
    }
    let text = std::str::from_utf8(&header).unwrap();
    let mut request = text.lines().next().unwrap().split_whitespace();
    assert_eq!(request.next(), Some("GET"));
    let path = request.next().unwrap();
    let object = objects.get(path).expect("signed Program chunk path");
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        object.len()
    );
    stream.write_all(response.as_bytes()).await.unwrap();
    stream.write_all(object).await.unwrap();
    stream.shutdown().await.unwrap();
}
