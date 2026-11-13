//! Selected admission at the generated linked-image boundary.
//! Decoded program bytes are never independently sufficient for this path.

use runmat_native_codegen::{aot::AotProgramManifest, aot::LinkedNativeProduct, NativeAssembly};

pub(super) struct LinkedSelectedAdmission {
    pub product: LinkedNativeProduct,
    pub services: Vec<runmat_execution::host::VerifiedNativeServiceIssuer>,
}

pub(super) fn admit(
    program: &AotProgramManifest,
    assembly: &NativeAssembly,
    verified_link: bool,
    source_service_factory: *const std::ffi::c_void,
    network_service_factory: *const std::ffi::c_void,
    external_factories: crate::input::LinkedExternalFactories,
) -> Result<Option<LinkedSelectedAdmission>, String> {
    let Some(binding) = program.product.selected() else {
        if !source_service_factory.is_null()
            || !network_service_factory.is_null()
            || !external_factories.ui.is_null()
            || !external_factories.callback.is_null()
            || !external_factories.dynamic.is_null()
        {
            return Err("unselected program passed a linked service factory".into());
        }
        return Ok(None);
    };
    if !verified_link {
        return Err("selected program requires installed selected-runtime authority".into());
    }
    let probe = runmat_runtime::native::native_abi_probe();
    if probe.binding().map_err(|error| error.to_string())? != binding.abi
        || *assembly.program.runtime_fingerprint()
            != runmat_runtime::native::linked_runtime_fingerprint()
        || *assembly.program.catalog_fingerprint()
            != runmat_runtime::native::linked_catalog_fingerprint()
    {
        return Err("selected program differs from linked Runtime ABI or catalog".into());
    }
    // SAFETY: only the unsafe generated-launcher entry can set verified_link.
    // The qualified linker admitted exact selected object/runtime bytes before
    // producing this image; no owned input/manifest decoder can issue it.
    let linked = unsafe { LinkedNativeProduct::from_verified_link(binding.clone()) }
        .map_err(|error| format!("selected linked product is invalid: {error}"))?;
    linked
        .validate_assembly(assembly)
        .map_err(|error| format!("selected Native IR is invalid: {error}"))?;
    let has_source = binding.services.iter().any(|service| {
        service.kind == runmat_execution::host::NativeDeploymentServiceKind::SourceEvaluation
    });
    if !has_source && !source_service_factory.is_null() {
        return Err(
            "selected program without SourceEvaluation passed a source-service factory".into(),
        );
    }
    let has_network = binding.services.iter().any(|service| {
        service.kind
            == runmat_execution::host::NativeDeploymentServiceKind::Effect(
                runmat_types::EffectKind::Network,
            )
    });
    if !has_network && !network_service_factory.is_null() {
        return Err("selected program without Network passed a network-service factory".into());
    }
    for (kind, factory) in [
        (
            runmat_execution::host::NativeDeploymentServiceKind::Effect(
                runmat_types::EffectKind::UserInterface,
            ),
            external_factories.ui,
        ),
        (
            runmat_execution::host::NativeDeploymentServiceKind::Effect(
                runmat_types::EffectKind::HostCallback,
            ),
            external_factories.callback,
        ),
        (
            runmat_execution::host::NativeDeploymentServiceKind::HostedDynamicDispatch,
            external_factories.dynamic,
        ),
    ] {
        let reached = binding.services.iter().any(|service| service.kind == kind);
        if reached == factory.is_null() {
            return Err(format!(
                "selected linked factory inventory differs for {kind:?}"
            ));
        }
    }
    if !external_factories.dynamic.is_null()
        && program.runtime_binding_mode
            != runmat_native_codegen::aot::AotRuntimeBindingMode::Dynamic
    {
        return Err("hosted dynamic bindings cannot replace a closed-world table".into());
    }
    let mut services = Vec::with_capacity(binding.services.len());
    for service in &binding.services {
        let issuer = match service.kind {
            runmat_execution::host::NativeDeploymentServiceKind::Effect(
                runmat_types::EffectKind::Clock,
            ) => crate::clock_service::admit(&linked)?,
            runmat_execution::host::NativeDeploymentServiceKind::SourceEvaluation => {
                #[cfg(feature = "source-evaluation")]
                {
                    crate::source_service::admit(&linked, source_service_factory)?
                }
                #[cfg(not(feature = "source-evaluation"))]
                {
                    return Err("selected source-evaluation component is not linked".into());
                }
            }
            runmat_execution::host::NativeDeploymentServiceKind::Effect(
                runmat_types::EffectKind::Network,
            ) => crate::network_service::admit(&linked, network_service_factory)?,
            runmat_execution::host::NativeDeploymentServiceKind::Effect(
                runmat_types::EffectKind::UserInterface,
            ) => crate::external_service::admit(&linked, service.kind, external_factories.ui)?,
            runmat_execution::host::NativeDeploymentServiceKind::Effect(
                runmat_types::EffectKind::HostCallback,
            ) => {
                crate::external_service::admit(&linked, service.kind, external_factories.callback)?
            }
            runmat_execution::host::NativeDeploymentServiceKind::HostedDynamicDispatch => {
                crate::external_service::admit(&linked, service.kind, external_factories.dynamic)?
            }
            runmat_execution::host::NativeDeploymentServiceKind::Effect(
                runmat_types::EffectKind::WorkspaceRead
                | runmat_types::EffectKind::WorkspaceWrite
                | runmat_types::EffectKind::EnvironmentRead
                | runmat_types::EffectKind::EnvironmentWrite
                | runmat_types::EffectKind::FilesystemRead
                | runmat_types::EffectKind::FilesystemWrite
                | runmat_types::EffectKind::Randomness,
            ) => crate::local_service::admit(&linked, service.kind)?,
            runmat_execution::host::NativeDeploymentServiceKind::Effect(
                runmat_types::EffectKind::MaySuspend
                | runmat_types::EffectKind::MayThrow
                | runmat_types::EffectKind::Unknown,
            ) => {
                return Err(format!(
                    "selected program requires unsupported linked runtime service {:?}",
                    service.kind
                ))
            }
        };
        services.push(issuer);
    }
    Ok(Some(LinkedSelectedAdmission {
        product: linked,
        services,
    }))
}
