use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use sha2::{Digest, Sha256};
pub(super) const RUNMAT_PCODE_MAGIC: &str = "RUNMAT-PCODE-V1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcodeAlgorithm {
    R2007b,
    R2022a,
}

impl PcodeAlgorithm {
    fn label(self) -> &'static str {
        match self {
            PcodeAlgorithm::R2007b => "R2007b",
            PcodeAlgorithm::R2022a => "R2022a",
        }
    }

    fn from_label(label: &str) -> Option<Self> {
        match label {
            "R2007b" => Some(PcodeAlgorithm::R2007b),
            "R2022a" => Some(PcodeAlgorithm::R2022a),
            _ => None,
        }
    }

    fn key(self) -> &'static [u8] {
        match self {
            PcodeAlgorithm::R2007b => b"runmat-pcode-r2007b-v1",
            PcodeAlgorithm::R2022a => b"runmat-pcode-r2022a-v1",
        }
    }
}

pub fn encode_pcode_source(source: &str, source_name: &str, algorithm: PcodeAlgorithm) -> String {
    let payload = transform_payload(source.as_bytes(), algorithm);
    let source_name = BASE64.encode(source_name.as_bytes());
    format!(
        "{RUNMAT_PCODE_MAGIC}\nalgorithm:{}\nsource:{}\nsha256:{}\npayload:{}\n",
        algorithm.label(),
        source_name,
        sha256_hex(source.as_bytes()),
        BASE64.encode(payload)
    )
}

pub fn decode_pcode_bytes(bytes: &[u8]) -> Result<String, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "P-code file is not UTF-8".to_string())?;
    let mut lines = text.lines();
    if lines.next() != Some(RUNMAT_PCODE_MAGIC) {
        return Err("not a RunMat-generated P-code file".to_string());
    }

    let mut algorithm = None;
    let mut expected_hash = None;
    let mut payload = None;
    for line in lines {
        if let Some(rest) = line.strip_prefix("algorithm:") {
            algorithm = PcodeAlgorithm::from_label(rest);
        } else if let Some(rest) = line.strip_prefix("sha256:") {
            expected_hash = Some(rest.to_string());
        } else if let Some(rest) = line.strip_prefix("payload:") {
            payload = Some(rest.to_string());
        }
    }

    let algorithm = algorithm.ok_or_else(|| "P-code algorithm is missing".to_string())?;
    let expected_hash = expected_hash.ok_or_else(|| "P-code hash is missing".to_string())?;
    let payload = payload.ok_or_else(|| "P-code payload is missing".to_string())?;
    let obfuscated = BASE64
        .decode(payload.as_bytes())
        .map_err(|err| format!("P-code payload is invalid base64: {err}"))?;
    let decoded = transform_payload(&obfuscated, algorithm);
    if sha256_hex(&decoded) != expected_hash {
        return Err("P-code payload hash mismatch".to_string());
    }
    String::from_utf8(decoded).map_err(|_| "P-code payload is not UTF-8 source".to_string())
}

fn transform_payload(bytes: &[u8], algorithm: PcodeAlgorithm) -> Vec<u8> {
    let key = algorithm.key();
    bytes
        .iter()
        .enumerate()
        .map(|(index, byte)| {
            let salt = match algorithm {
                PcodeAlgorithm::R2007b => index.wrapping_mul(31) as u8,
                PcodeAlgorithm::R2022a => index.wrapping_mul(73).rotate_left(1) as u8,
            };
            byte ^ key[index % key.len()] ^ salt
        })
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}
