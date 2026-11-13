//! Two selected links from one frozen paired object binding. This is a
//! completed-pair handoff, not an atomic multi-file publication transaction.

use std::path::{Path, PathBuf};

use runmat_execution_artifact::target_test::{NativePairedBuildBinding, NativePairedProductHeader};
use runmat_native_codegen::aot::SelectedNativeProduct;

use super::persisted::CompletedImageCommitment;
use super::{
    link_selected_standalone, ExternalNativeLinkAdmission, HostedNativeBuildTrees, LinkRequest,
    LinkedProgram, NativeProgramLinkIdentity, PersistedSelectedImage,
};
use crate::archive::RuntimeArchive;
use crate::object::PairedNativeObjects;
use crate::{AotError, AotResult};

#[derive(Clone, Copy)]
pub struct PairedLinkRequests<'a> {
    pub application: &'a LinkRequest,
    pub tests: &'a LinkRequest,
}

pub(crate) struct PersistedPairReopenRequest<'a> {
    pub application_header: NativePairedProductHeader,
    pub test_header: NativePairedProductHeader,
    pub application_receipt: &'a PersistedSelectedImage,
    pub test_receipt: &'a PersistedSelectedImage,
    pub application_path: PathBuf,
    pub test_path: PathBuf,
    pub application_commitment: CompletedImageCommitment,
    pub test_commitment: CompletedImageCommitment,
}

/// Issued only when both qualified selected links have completed and their
/// private completed-image receipts agree with the same paired build binding.
/// Paths may later move; consumers must hash the opened bytes against each
/// receipt before execution or installation.
pub struct LinkedNativePair {
    binding: NativePairedBuildBinding,
    application_header: NativePairedProductHeader,
    test_header: NativePairedProductHeader,
    application: LinkedProgram,
    tests: LinkedProgram,
}

impl LinkedNativePair {
    pub fn binding(&self) -> &NativePairedBuildBinding {
        &self.binding
    }

    pub fn application_header(&self) -> &NativePairedProductHeader {
        &self.application_header
    }

    pub fn test_header(&self) -> &NativePairedProductHeader {
        &self.test_header
    }

    pub fn application(&self) -> &LinkedProgram {
        &self.application
    }

    pub fn tests(&self) -> &LinkedProgram {
        &self.tests
    }
}

pub fn link_selected_native_pair(
    selected: &SelectedNativeProduct<'_>,
    pair: &PairedNativeObjects,
    runtime: &RuntimeArchive,
    admission: &ExternalNativeLinkAdmission,
    hosted: HostedNativeBuildTrees<'_>,
    requests: PairedLinkRequests<'_>,
) -> AotResult<LinkedNativePair> {
    // All identity/path checks precede staging, linking, or publication.
    pair.validate()?;
    super::plan::validate_pair(pair.application(), runtime)?;
    super::plan::validate_pair(pair.tests(), runtime)?;
    if &runtime.manifest.product != selected.product()
        || &pair.binding().selected_product
            != selected.product().selected().expect("selected constructor")
        || requests.application.linker.is_some()
        || requests.tests.linker.is_some()
    {
        return Err(invalid(
            "paired objects, runtime, or requests differ from verified selection",
        ));
    }
    let app_path = admitted_output(requests.application, true)?;
    let test_path = admitted_output(requests.tests, true)?;
    if app_path.canonical == test_path.canonical || same_existing_file(&app_path, &test_path)? {
        return Err(invalid(
            "paired application and test outputs alias the same path",
        ));
    }

    // Use the resolved parent paths for both requests. This removes the
    // caller's original parent alias; it is not a directory-handle race proof.
    let mut application_request = requests.application.clone();
    application_request.output = app_path.canonical;
    let mut test_request = requests.tests.clone();
    test_request.output = test_path.canonical.clone();

    let application = link_selected_standalone(
        selected,
        pair.application(),
        runtime,
        admission,
        hosted,
        &application_request,
    )?;
    // On a case-insensitive filesystem, two distinct absent spellings can
    // become aliases only after the application image is published. Check the
    // actual destination again before the second linker has any effects.
    let current_test_path = admitted_output(&test_request, true)?;
    if current_test_path.canonical != test_path.canonical
        || same_existing_file(
            &admitted_output(&application_request, false)?,
            &current_test_path,
        )?
    {
        return Err(invalid(
            "paired test output aliases the published application image",
        ));
    }
    // If this second link fails, the first publication remains a standalone
    // image. No LinkedNativePair is issued or atomic pair success claimed.
    let tests = link_selected_standalone(
        selected,
        pair.tests(),
        runtime,
        admission,
        hosted,
        &test_request,
    )?;
    check_completed(
        &application,
        pair.application(),
        runtime,
        selected,
        admission,
    )?;
    check_completed(&tests, pair.tests(), runtime, selected, admission)?;
    Ok(LinkedNativePair {
        binding: pair.binding().clone(),
        application_header: pair.application_header().clone(),
        test_header: pair.test_header().clone(),
        application,
        tests,
    })
}

/// Reopen a durable pair only while the independently verified selected pack,
/// original native objects, runtime archive and host admission are available.
/// The persisted records alone never authorize execution.
pub(crate) fn reopen_selected_native_pair(
    selected: &SelectedNativeProduct<'_>,
    pair: &PairedNativeObjects,
    runtime: &RuntimeArchive,
    admission: &ExternalNativeLinkAdmission,
    request: PersistedPairReopenRequest<'_>,
) -> AotResult<LinkedNativePair> {
    let PersistedPairReopenRequest {
        application_header,
        test_header,
        application_receipt,
        test_receipt,
        application_path,
        test_path,
        application_commitment,
        test_commitment,
    } = request;
    pair.validate()?;
    super::plan::validate_pair(pair.application(), runtime)?;
    super::plan::validate_pair(pair.tests(), runtime)?;
    if application_header != *pair.application_header()
        || test_header != *pair.test_header()
        || application_path == test_path
    {
        return Err(invalid(
            "persisted paired headers or physical image paths differ from original objects",
        ));
    }
    let application = LinkedProgram::reopen_selected(
        application_path,
        application_receipt,
        selected,
        admission,
        &NativeProgramLinkIdentity::from_products(pair.application(), runtime),
        pair.application().manifest.runtime_binding_mode,
        application_commitment,
    )?;
    let tests = LinkedProgram::reopen_selected(
        test_path,
        test_receipt,
        selected,
        admission,
        &NativeProgramLinkIdentity::from_products(pair.tests(), runtime),
        pair.tests().manifest.runtime_binding_mode,
        test_commitment,
    )?;
    check_completed(
        &application,
        pair.application(),
        runtime,
        selected,
        admission,
    )?;
    check_completed(&tests, pair.tests(), runtime, selected, admission)?;
    Ok(LinkedNativePair {
        binding: pair.binding().clone(),
        application_header,
        test_header,
        application,
        tests,
    })
}

fn check_completed(
    linked: &LinkedProgram,
    object: &runmat_native_codegen::aot::RelocatableNativeObject,
    runtime: &RuntimeArchive,
    selected: &SelectedNativeProduct<'_>,
    admission: &ExternalNativeLinkAdmission,
) -> AotResult<()> {
    let completed = linked
        .verified_selected_image()?
        .ok_or_else(|| invalid("paired link lacks completed selected image proof"))?;
    if completed.product() != selected.product().selected().expect("selected constructor")
        || completed.selected_build() != &admission.identity()
        || completed.inputs() != &NativeProgramLinkIdentity::from_products(object, runtime)
    {
        return Err(invalid(
            "completed paired image differs from its exact selected inputs",
        ));
    }
    Ok(())
}

struct AdmittedOutput {
    canonical: PathBuf,
    exists: bool,
}

fn admitted_output(request: &LinkRequest, reject_existing: bool) -> AotResult<AdmittedOutput> {
    let parent = request
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = request
        .output
        .file_name()
        .ok_or_else(|| invalid("paired output has no file name"))?;
    let parent = std::fs::canonicalize(parent)
        .map_err(|error| AotError::io("resolve paired output parent", parent, error))?;
    if !parent.is_dir() {
        return Err(invalid("paired output parent is not a directory"));
    }
    let path = parent.join(name);
    let exists = match std::fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(invalid("paired output may not be a symbolic link"));
        }
        Ok(metadata) if !metadata.is_file() => {
            return Err(invalid(
                "paired output already exists and is not a regular file",
            ));
        }
        Ok(_) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => return Err(AotError::io("inspect paired output", &path, error)),
    };
    if reject_existing && exists && !request.overwrite {
        return Err(invalid(
            "paired output already exists and overwrite was not requested",
        ));
    }
    Ok(AdmittedOutput {
        canonical: path,
        exists,
    })
}

fn same_existing_file(left: &AdmittedOutput, right: &AdmittedOutput) -> AotResult<bool> {
    if !left.exists || !right.exists {
        return Ok(false);
    }
    same_file::is_same_file(&left.canonical, &right.canonical).map_err(|error| {
        AotError::io(
            "compare paired output file identity",
            &right.canonical,
            error,
        )
    })
}

fn invalid(message: impl Into<String>) -> AotError {
    AotError::contract("aot.link.paired", message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preflight_resolves_parent_aliases_without_publishing() {
        let root = tempfile::tempdir().unwrap();
        let request = |output| LinkRequest {
            output,
            linker: None,
            keep_temps: false,
            overwrite: false,
        };
        let direct = admitted_output(&request(root.path().join("app")), true).unwrap();
        let distinct_case = admitted_output(&request(root.path().join("APP")), true).unwrap();
        assert_ne!(direct.canonical, distinct_case.canonical);
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(root.path(), root.path().join("alias")).unwrap();
            let aliased = admitted_output(&request(root.path().join("alias/app")), true).unwrap();
            assert_eq!(direct.canonical, aliased.canonical);
        }
        assert!(!root.path().join("app").exists());

        std::fs::write(root.path().join("first"), b"image").unwrap();
        std::fs::hard_link(root.path().join("first"), root.path().join("alias-image")).unwrap();
        assert!(admitted_output(&request(root.path().join("first")), true).is_err());
        let mut first = request(root.path().join("first"));
        first.overwrite = true;
        let mut alias = request(root.path().join("alias-image"));
        alias.overwrite = true;
        assert!(same_existing_file(
            &admitted_output(&first, true).unwrap(),
            &admitted_output(&alias, true).unwrap(),
        )
        .unwrap());
    }
}
