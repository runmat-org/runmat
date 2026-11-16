//! Distribution container for the stock compile-time runtime. The trusted CLI
//! pins this container's identity; native compatibility remains the existing
//! RuntimeArchiveManifest contract. Members are read, never extracted as paths.

use super::manifest::MAX_RUNTIME_MANIFEST_BYTES;
use super::{RuntimeArchive, RuntimeArchiveManifest, MAX_RUNTIME_PAYLOAD_BYTES};
use crate::{AotError, AotResult};
use runmat_execution::Digest;
use std::io::{Cursor, Read, Write};

#[cfg(test)]
mod tests;

const SCHEMA_VERSION: u16 = 1;
const MAX_MANIFEST_BYTES: u64 = 2 * 1024 * 1024;
const MAX_NOTICE_BYTES: u64 = 8 * 1024 * 1024;
const MAX_PACK_BYTES: u64 = MAX_RUNTIME_PAYLOAD_BYTES + 16 * 1024 * 1024;
const MEMBERS: [&str; 5] = [
    "runtime-archive.payload",
    "runtime-archive.json",
    "LICENSE.md",
    "FREE-USE.md",
    "THIRD_PARTY_NOTICES.txt",
];

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct RuntimePackManifest {
    pub schema_version: u16,
    pub runtime: RuntimeArchiveManifest,
    pub pack_digest: Digest,
    pub pack_bytes: u64,
}

impl RuntimePackManifest {
    pub fn from_json(bytes: &[u8]) -> AotResult<Self> {
        if u64::try_from(bytes.len()).map_err(invalid)? > MAX_MANIFEST_BYTES {
            return Err(invalid("runtime pack manifest exceeds its size bound"));
        }
        #[derive(serde::Deserialize)]
        struct Header {
            schema_version: u16,
        }
        let header: Header = serde_json::from_slice(bytes).map_err(invalid)?;
        if header.schema_version != SCHEMA_VERSION {
            return Err(invalid("unsupported runtime pack manifest schema"));
        }
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire<'a> {
            schema_version: u16,
            #[serde(borrow)]
            runtime: &'a serde_json::value::RawValue,
            pack_digest: Digest,
            pack_bytes: u64,
        }
        let wire: Wire<'_> = serde_json::from_slice(bytes).map_err(invalid)?;
        let runtime = RuntimeArchiveManifest::from_json(wire.runtime.get().as_bytes())?;
        let result = Self {
            schema_version: wire.schema_version,
            runtime,
            pack_digest: wire.pack_digest,
            pack_bytes: wire.pack_bytes,
        };
        result.validate_for_packaging()?;
        Ok(result)
    }

    fn validate_for_packaging(&self) -> AotResult<()> {
        if self.schema_version != SCHEMA_VERSION
            || self.pack_bytes == 0
            || self.pack_bytes > MAX_PACK_BYTES
        {
            return Err(invalid("invalid runtime pack schema or size"));
        }
        self.runtime.validate_for_packaging()?;
        if self.runtime.product.selected().is_some() {
            return Err(invalid(
                "the stock runtime pack cannot replace selected target-pack admission",
            ));
        }
        Ok(())
    }

    pub fn validate(&self) -> AotResult<()> {
        self.validate_for_packaging()?;
        self.runtime.validate()
    }

    pub fn file_name(&self) -> AotResult<String> {
        self.validate_for_packaging()?;
        Ok(format!(
            "runmat-aot-runtime-{}-{}.zip",
            self.runtime.runmat_version, self.runtime.target_triple
        ))
    }
}

pub struct RuntimePack {
    pub manifest: RuntimePackManifest,
    pub bytes: Vec<u8>,
}

pub struct RuntimePackLicenses<'a> {
    pub license: &'a [u8],
    pub free_use: &'a [u8],
}

pub fn build_runtime_pack(
    runtime: &RuntimeArchive,
    licenses: RuntimePackLicenses<'_>,
) -> AotResult<RuntimePack> {
    runtime.manifest.validate_for_packaging()?;
    if u64::try_from(runtime.payload().len()).map_err(invalid)? != runtime.manifest.payload_bytes
        || Digest::sha256(runtime.payload()) != runtime.manifest.payload_digest
    {
        return Err(invalid("runtime payload differs from its manifest"));
    }
    let metadata = serde_json::to_vec(&runtime.manifest).map_err(invalid)?;
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .unix_permissions(0o644);
    for (name, bytes) in MEMBERS.into_iter().zip([
        runtime.payload(),
        metadata.as_slice(),
        licenses.license,
        licenses.free_use,
        runmat_runtime::third_party_notices::bytes(),
    ]) {
        let limit = match name {
            "runtime-archive.payload" => runtime.manifest.payload_bytes,
            "runtime-archive.json" => u64::try_from(MAX_RUNTIME_MANIFEST_BYTES).map_err(invalid)?,
            _ => MAX_NOTICE_BYTES,
        };
        if u64::try_from(bytes.len()).map_err(invalid)? > limit {
            return Err(invalid(
                "runtime pack metadata or notices exceed their size bound",
            ));
        }
        if name == MEMBERS[2] || name == MEMBERS[3] {
            validate_license(bytes)?;
        }
        writer.start_file(name, options).map_err(invalid)?;
        writer.write_all(bytes).map_err(invalid)?;
    }
    let bytes = writer.finish().map_err(invalid)?.into_inner();
    let manifest = RuntimePackManifest {
        schema_version: SCHEMA_VERSION,
        runtime: runtime.manifest.clone(),
        pack_digest: Digest::sha256(&bytes),
        pack_bytes: u64::try_from(bytes.len()).map_err(invalid)?,
    };
    manifest.validate_for_packaging()?;
    Ok(RuntimePack { manifest, bytes })
}

pub fn decode_runtime_pack(
    expected: &RuntimePackManifest,
    bytes: &[u8],
) -> AotResult<RuntimeArchive> {
    expected.validate()?;
    if u64::try_from(bytes.len()).map_err(invalid)? != expected.pack_bytes
        || Digest::sha256(bytes) != expected.pack_digest
    {
        return Err(invalid(
            "downloaded runtime pack differs from the trusted CLI manifest",
        ));
    }
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).map_err(invalid)?;
    if archive.len() != MEMBERS.len() {
        return Err(invalid("runtime pack has an unexpected member inventory"));
    }
    let metadata = read_member(
        &mut archive,
        MEMBERS[1],
        u64::try_from(MAX_RUNTIME_MANIFEST_BYTES).map_err(invalid)?,
    )?;
    let runtime = RuntimeArchiveManifest::from_json(&metadata)?;
    if runtime != expected.runtime {
        return Err(invalid(
            "runtime pack contains a different native runtime manifest",
        ));
    }
    // First-party license bytes are part of the already verified whole-pack
    // digest. They are supplied by the release producer rather than read from
    // a workspace path when a crates.io consumer builds this crate.
    for name in [MEMBERS[2], MEMBERS[3]] {
        validate_license(&read_member(&mut archive, name, MAX_NOTICE_BYTES)?)?;
    }
    if read_member(&mut archive, MEMBERS[4], MAX_NOTICE_BYTES)?
        != runmat_runtime::third_party_notices::bytes()
    {
        return Err(invalid("runtime pack notices differ from this CLI"));
    }
    RuntimeArchive::new(
        runtime,
        read_member(&mut archive, MEMBERS[0], expected.runtime.payload_bytes)?,
    )
}

fn validate_license(bytes: &[u8]) -> AotResult<()> {
    if bytes.is_empty() || std::str::from_utf8(bytes).is_err() {
        return Err(invalid("runtime pack license must contain UTF-8 text"));
    }
    Ok(())
}

fn read_member(
    archive: &mut zip::ZipArchive<Cursor<&[u8]>>,
    name: &str,
    limit: u64,
) -> AotResult<Vec<u8>> {
    let file = archive.by_name(name).map_err(invalid)?;
    if file.is_dir()
        || file.size() > limit
        || file
            .unix_mode()
            .is_some_and(|mode| mode & 0o170000 == 0o120000)
    {
        return Err(invalid("runtime pack member has an invalid kind or size"));
    }
    let mut bytes = Vec::new();
    file.take(
        limit
            .checked_add(1)
            .ok_or_else(|| invalid("runtime pack size overflow"))?,
    )
    .read_to_end(&mut bytes)
    .map_err(invalid)?;
    if u64::try_from(bytes.len()).map_err(invalid)? > limit {
        return Err(invalid("runtime pack member exceeds its admitted size"));
    }
    Ok(bytes)
}

fn invalid(error: impl std::fmt::Display) -> AotError {
    AotError::contract("aot.distribution", error.to_string())
}
