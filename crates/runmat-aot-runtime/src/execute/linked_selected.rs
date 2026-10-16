//! Selected admission at the generated linked-image boundary.
//! Decoded program bytes are never independently sufficient for this path.

use runmat_native_codegen::{aot::AotProgramManifest, aot::LinkedNativeProduct, NativeAssembly};

pub(super) struct LinkedSelectedAdmission {
    pub product: LinkedNativeProduct,
    pub source_evaluation: Option<runmat_execution::host::VerifiedNativeServiceIssuer>,
}

pub(super) fn admit(
    program: &AotProgramManifest,
    assembly: &NativeAssembly,
    verified_link: bool,
    source_service_factory: *const std::ffi::c_void,
) -> Result<Option<LinkedSelectedAdmission>, String> {
    let Some(binding) = program.product.selected() else {
        if !source_service_factory.is_null() {
            return Err("unselected program passed a linked source-service factory".into());
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
    let source_evaluation = if binding.required_services.is_empty() {
        if !source_service_factory.is_null() {
            return Err("pure selected program passed a source-service factory".into());
        }
        None
    } else if binding.required_services.len() == 1
        && binding
            .required_services
            .contains(&runmat_execution::host::NativeDeploymentServiceKind::SourceEvaluation)
    {
        #[cfg(feature = "source-evaluation")]
        {
            Some(crate::source_service::admit(
                &linked,
                source_service_factory,
            )?)
        }
        #[cfg(not(feature = "source-evaluation"))]
        {
            return Err("selected source-evaluation component is not linked".into());
        }
    } else {
        return Err("selected program requires unsupported linked runtime services".into());
    };
    Ok(Some(LinkedSelectedAdmission {
        product: linked,
        source_evaluation,
    }))
}
