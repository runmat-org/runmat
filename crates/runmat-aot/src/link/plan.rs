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
    let closed_world = object_product.manifest.runtime_binding_mode
        == runmat_native_codegen::aot::AotRuntimeBindingMode::ClosedWorld;
    if closed_world && !runtime.manifest.capabilities.closed_world_linking {
        return Err(AotError::contract(
            "aot.link.closed_world",
            "runtime archive does not support closed-world extraction",
        ));
    }
    let mut response_tokens = Vec::new();
    match driver.family {
        LinkerFamily::UnixCc => {
            response_tokens.push(object.display().to_string());
            if closed_world {
                response_tokens.push(archive.display().to_string());
                response_tokens.push(if cfg!(target_os = "macos") {
                    "-Wl,-dead_strip".into()
                } else {
                    "-Wl,--gc-sections".into()
                });
            } else if cfg!(target_os = "macos") {
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
            if closed_world {
                response_tokens.push(archive.display().to_string());
                response_tokens.push("/OPT:REF".into());
            } else {
                response_tokens.push(format!("/WHOLEARCHIVE:{}", archive.display()));
            }
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

#[cfg(test)]
mod tests {
    use runmat_execution::{Digest, ProgramEnvironment};
    use runmat_native_codegen::{
        aot::{
            AotRuntimeBindingMode, NativeObjectFormat, NativeObjectFunction, NativeObjectManifest,
            NativeOptimization, RelocatableNativeObject, AOT_ENTRY_SYMBOL,
            NATIVE_OBJECT_SCHEMA_VERSION,
        },
        NativeTarget,
    };
    use runmat_types::ProgramFunctionId;

    use crate::{
        archive::{build_runtime_archive, RuntimeArchiveCapabilities, RuntimeArchiveEncoding},
        link::{LinkerDriver, LinkerFamily},
    };

    use super::build_link_plan;

    fn products(
        mode: AotRuntimeBindingMode,
    ) -> (RelocatableNativeObject, crate::archive::RuntimeArchive) {
        let runtime = Digest::sha256(b"runtime");
        let catalog = Digest::sha256(b"catalog");
        let environment = ProgramEnvironment::new(1, 1, runtime, catalog, "matlab").unwrap();
        let archive = build_runtime_archive(
            b"!<arch>\nfixture",
            &environment,
            Vec::new(),
            RuntimeArchiveEncoding::Raw,
            RuntimeArchiveCapabilities::standalone_host(),
        )
        .unwrap();
        let bytes = b"object".to_vec();
        let target = NativeTarget::current();
        let function = ProgramFunctionId(1);
        let object = RelocatableNativeObject {
            manifest: NativeObjectManifest {
                schema_version: NATIVE_OBJECT_SCHEMA_VERSION,
                object_format: NativeObjectFormat::for_target(&target).unwrap(),
                target,
                executable_cache_key: Digest::sha256(b"executable"),
                native_cache_key: Digest::sha256(b"native"),
                runtime_fingerprint: runtime,
                catalog_fingerprint: catalog,
                optimization: NativeOptimization::Speed,
                runtime_binding_mode: mode,
                object_digest: Digest::sha256(&bytes),
                object_bytes: bytes.len() as u64,
                entrypoint: function,
                functions: vec![NativeObjectFunction {
                    function,
                    symbol: AOT_ENTRY_SYMBOL.into(),
                }],
                data: Vec::new(),
            },
            bytes,
        };
        (object, archive)
    }

    #[test]
    fn closed_world_uses_selective_archive_extraction_on_unix_and_msvc() {
        let (object, runtime) = products(AotRuntimeBindingMode::ClosedWorld);
        for family in [LinkerFamily::UnixCc, LinkerFamily::Msvc] {
            let plan = build_link_plan(
                &object,
                &runtime,
                LinkerDriver {
                    path: "linker".into(),
                    family,
                },
                "program.o".as_ref(),
                "runtime.a".as_ref(),
                "program".as_ref(),
            )
            .unwrap();
            assert!(plan
                .response_tokens
                .iter()
                .any(|token| token == "runtime.a"));
            assert!(!plan
                .response_tokens
                .iter()
                .any(|token| token.contains("whole-archive") || token.contains("WHOLEARCHIVE")));
        }
    }

    #[test]
    fn dynamic_runtime_preserves_whole_archive_discovery() {
        let (object, runtime) = products(AotRuntimeBindingMode::Dynamic);
        let plan = build_link_plan(
            &object,
            &runtime,
            LinkerDriver {
                path: "linker".into(),
                family: LinkerFamily::Msvc,
            },
            "program.obj".as_ref(),
            "runtime.lib".as_ref(),
            "program.exe".as_ref(),
        )
        .unwrap();
        assert!(plan
            .response_tokens
            .iter()
            .any(|token| token == "/WHOLEARCHIVE:runtime.lib"));
    }
}
