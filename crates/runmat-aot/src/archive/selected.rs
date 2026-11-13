//! Host-side packing of exact authenticated target-built runtime bytes. No
//! target code is executed, and no ambient native-link tokens are inherited.
use runmat_execution::ProgramEnvironment;
use runmat_native_codegen::aot::SelectedNativeProduct;

use super::{RuntimeArchive, RuntimeArchiveCapabilities, RuntimeArchiveEncoding};
use crate::{AotError, AotResult};

pub fn build_selected_runtime_archive(
    selected: &SelectedNativeProduct<'_>,
    environment: &ProgramEnvironment,
    encoding: RuntimeArchiveEncoding,
    mut capabilities: RuntimeArchiveCapabilities,
) -> AotResult<RuntimeArchive> {
    let contents = selected.contents();
    let manifest = contents.manifest();
    if environment.runtime_fingerprint != manifest.runtime
        || environment.catalog_fingerprint != manifest.catalog
    {
        return Err(AotError::contract(
            "aot.archive.environment",
            "selected pack differs from program environment",
        ));
    }
    let bytes = contents
        .component_bytes(&manifest.runtime_component)
        .ok_or_else(|| {
            AotError::contract(
                "aot.archive.component",
                "verified runtime bytes are missing",
            )
        })?;
    let probe = super::read_native_abi_probe(bytes)?;
    if &probe != contents.abi_probe() {
        return Err(AotError::contract(
            "aot.archive.probe",
            "target-built runtime probe differs from retained probe",
        ));
    }
    let physical = super::read_native_accelerator_profile(bytes)?;
    if capabilities.accelerators.has_backend() && capabilities.accelerators != physical {
        return Err(AotError::contract(
            "aot.archive.accelerator_profile",
            "selected capability claim differs from target-built backend inventory",
        ));
    }
    capabilities.accelerators = physical;
    super::product::build_archive(
        bytes,
        environment,
        Vec::new(),
        encoding,
        capabilities,
        selected.target().clone(),
        selected.product().clone(),
    )
}
