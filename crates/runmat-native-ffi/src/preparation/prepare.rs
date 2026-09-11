use thiserror::Error;

use crate::{
    prepare_header_with_declarations_report, HeaderPreparation, HeaderPreparationError,
    NativeInterfaceArtifactError, NativeInterfaceArtifactManifest,
};

use super::{NativeInterfacePreparation, PreparedNativeInterface};

#[derive(Debug, Error)]
pub enum NativeInterfacePreparationError {
    #[error(transparent)]
    Header(#[from] HeaderPreparationError),
    #[error(transparent)]
    Artifact(#[from] NativeInterfaceArtifactError),
}

pub fn prepare_native_interface(
    request: &NativeInterfacePreparation,
) -> Result<PreparedNativeInterface, NativeInterfacePreparationError> {
    let prepared = prepare_header_with_declarations_report(
        &HeaderPreparation {
            header: request.primary_header.clone(),
            library_name: request.library_name.clone(),
            library_path: request.library_path.display().to_string(),
            target_triple: target_lexicon::HOST.to_string(),
            clang: request.compiler_frontend.clone(),
            include_directories: request.include_directories.clone(),
            definitions: request.definitions.clone(),
        },
        &request.additional_headers,
    )?;
    let manifest = NativeInterfaceArtifactManifest::from_library_path(
        request.interface_name.clone(),
        prepared.metadata,
        &request.library_path,
    )?;
    Ok(PreparedNativeInterface {
        manifest,
        warnings: prepared.warnings,
    })
}
