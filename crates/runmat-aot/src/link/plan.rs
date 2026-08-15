use std::path::{Path, PathBuf};

use runmat_native_codegen::aot::RelocatableNativeObject;

use crate::{archive::RuntimeArchive, AotError, AotResult};

use super::{LinkerDriver, LinkerFamily};

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct LinkPlan {
    pub target_triple: String,
    pub driver: LinkerDriver,
    pub object: PathBuf,
    pub runtime_archive: PathBuf,
    pub output: PathBuf,
    pub response_tokens: Vec<String>,
}

pub fn build_link_plan(
    object_product: &RelocatableNativeObject,
    runtime: &RuntimeArchive,
    driver: LinkerDriver,
    object: &Path,
    archive: &Path,
    output: &Path,
) -> AotResult<LinkPlan> {
    object_product
        .validate()
        .map_err(|error| AotError::contract("aot.link.object", error.to_string()))?;
    runtime.manifest.validate()?;
    if object_product.manifest.target != runtime.manifest.native_target {
        return Err(AotError::contract(
            "aot.link.target",
            "user object and runtime archive target bindings do not match",
        ));
    }
    if object_product.manifest.runtime_fingerprint != runtime.manifest.runtime_fingerprint
        || object_product.manifest.catalog_fingerprint != runtime.manifest.catalog_fingerprint
    {
        return Err(AotError::contract(
            "aot.link.environment",
            "user object and runtime archive runtime/catalog identities do not match",
        ));
    }
    let mut response_tokens = Vec::new();
    match driver.family {
        LinkerFamily::UnixCc => {
            response_tokens.push(object.display().to_string());
            if cfg!(target_os = "macos") {
                response_tokens.push(format!("-Wl,-force_load,{}", archive.display()));
            } else {
                response_tokens.push("-Wl,--whole-archive".into());
                response_tokens.push(archive.display().to_string());
                response_tokens.push("-Wl,--no-whole-archive".into());
            }
            response_tokens.push("-o".into());
            response_tokens.push(output.display().to_string());
        }
        LinkerFamily::Msvc => {
            response_tokens.push(object.display().to_string());
            response_tokens.push(format!("/WHOLEARCHIVE:{}", archive.display()));
            response_tokens.push(format!("/OUT:{}", output.display()));
        }
    }
    response_tokens.extend(runtime.manifest.native_link_tokens.iter().cloned());
    if response_tokens.iter().any(|token| {
        token.is_empty()
            || token.len() > 4096
            || token.bytes().any(|byte| matches!(byte, 0 | b'\r' | b'\n'))
    }) {
        return Err(AotError::contract(
            "aot.link.response",
            "link response contains an invalid or oversized token",
        ));
    }
    Ok(LinkPlan {
        target_triple: runtime.manifest.target_triple.clone(),
        driver,
        object: object.to_path_buf(),
        runtime_archive: archive.to_path_buf(),
        output: output.to_path_buf(),
        response_tokens,
    })
}
