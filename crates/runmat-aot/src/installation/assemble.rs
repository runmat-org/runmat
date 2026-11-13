use super::{
    invalid, loader, ExternalNativeInstallationAdmission, NativeInstallationLayout,
    NativeLoaderImage, NATIVE_INSTALLATION_SCHEMA_VERSION,
};
use crate::link::{LinkedProgram, NativeLinkBuildIdentity, NativeProgramLinkIdentity};
use crate::tree::{
    physical, resolve_build_tree, HostedNativeTree, NativeFile, NativeStorageGrant,
    ResolvedNativeTree,
};
use crate::{AotError, AotResult};
use runmat_execution::{
    host::{NativeSelectedProductBinding, VerifiedNativeServiceIssuer},
    Digest,
};
use runmat_execution_artifact::target_pack::VerifiedTargetPackContents;
use runmat_package::{ContentDigest, NormalizedRelativePath};
use runmat_package_cache::{TreeEntry, TreeManifest};

#[derive(Debug, serde::Serialize)]
pub struct NativeInstallationManifest {
    pub schema_version: u16,
    pub product: NativeSelectedProductBinding,
    pub target: runmat_native_codegen::NativeTarget,
    pub image: Digest,
    pub inputs: NativeProgramLinkIdentity,
    pub build: NativeLinkBuildIdentity,
    pub layout: NativeInstallationLayout,
    /// Recorded declaration only; deserializing it cannot mint permission.
    pub storage: NativeStorageGrant,
    pub loader_images: Vec<NativeLoaderImage>,
    pub deployment_qualification: Digest,
    pub payload: TreeManifest,
}

/// Borrows the authenticated pack/blob bytes and retains the actual image
/// File. No full binary JSON/base64 encoding or copied installation payload.
pub struct NativeInstallationBundle<'a> {
    pub(super) tree: ResolvedNativeTree<'a>,
    pub(super) manifest: NativeInstallationManifest,
    pub(super) bytes: Vec<u8>,
    pub(super) identity: Digest,
    pub(super) source_evaluation_issuer: Option<VerifiedNativeServiceIssuer>,
}
impl NativeInstallationBundle<'_> {
    pub fn manifest(&self) -> &NativeInstallationManifest {
        &self.manifest
    }
    pub fn canonical_manifest_bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn identity(&self) -> Digest {
        self.identity
    }
}

pub fn prepare_native_installation<'a>(
    contents: &'a VerifiedTargetPackContents<'_>,
    linked: &LinkedProgram,
    hosted_sysroot: Option<HostedNativeTree<'a>>,
    expected: &ExternalNativeInstallationAdmission,
) -> AotResult<NativeInstallationBundle<'a>> {
    let product = contents.selected_product_binding().map_err(invalid)?;
    let source_evaluation_issuer = product
        .required_services
        .contains(&runmat_execution::host::NativeDeploymentServiceKind::SourceEvaluation)
        .then(|| {
            contents.verified_service_issuer(
                runmat_execution::host::NativeDeploymentServiceKind::SourceEvaluation,
            )
        })
        .transpose()
        .map_err(invalid)?;
    let product_id = product.fingerprint().map_err(invalid)?;
    if linked.product.selected() != Some(&product)
        || linked.output_digest != expected.image
        || linked.inputs != expected.inputs
        || linked.selected_build.as_ref() != Some(&expected.build)
        || expected.product != product_id
        || expected.build.product != product_id
        || [expected.image, expected.deployment_qualification]
            .iter()
            .any(|value| value.bytes().iter().all(|byte| *byte == 0))
    {
        return Err(invalid(
            "deployment expectations differ from selected completed product",
        ));
    }
    expected
        .layout
        .validate(contents.selection().runtime_components())?;
    // Bound the caller-owned variable metadata before retaining any clones.
    // The final complete record is counted again by the same writer below.
    crate::tree::admit_json(
        &(&expected.layout, &expected.loader_images),
        expected.storage.metadata_bytes,
    )?;
    let sysroot = resolve_build_tree(
        contents,
        &contents.manifest().build_inputs.sysroot,
        hosted_sysroot,
        expected.build.sysroot,
        &expected.storage,
    )?;
    let source = std::fs::symlink_metadata(&linked.output)
        .map_err(|error| AotError::io("inspect installation source", &linked.output, error))?;
    if !source.is_file() {
        return Err(invalid("installation source is not a physical file"));
    }
    if source.len() > expected.storage.archive.max_file_bytes {
        return Err(AotError::ResourceUnavailable {
            message: "native image exceeds its independent caller file bound".into(),
        });
    }
    // Charge all physical occurrences, not unique source blobs. No payload
    // copy or filesystem write occurs before this independent grant check.
    let mut total = sysroot
        .manifest
        .total_bytes
        .checked_add(source.len())
        .ok_or_else(|| invalid("installation extent overflow"))?;
    for placement in &expected.layout.components {
        total = total
            .checked_add(
                contents
                    .component_bytes(&placement.component)
                    .ok_or_else(|| invalid("selected runtime bytes disappeared"))?
                    .len() as u64,
            )
            .ok_or_else(|| invalid("installation extent overflow"))?;
    }
    let added = expected.layout.components.len() as u64 + 1;
    expected.storage.admit_total(
        expected
            .storage
            .admit(&sysroot.manifest)?
            .checked_add(added + 1)
            .ok_or_else(|| invalid("installation count overflow"))?,
        sysroot
            .manifest
            .file_count
            .checked_add(added)
            .ok_or_else(|| invalid("installation count overflow"))?,
        total,
    )?;
    let image = std::fs::File::open(&linked.output)
        .map_err(|error| AotError::io("retain installation image", &linked.output, error))?;
    if physical::hash_open_file(&image, source.len(), &linked.output)? != expected.image {
        return Err(invalid("completed image changed before installation"));
    }
    {
        let reader = image.try_clone().map_err(|error| {
            AotError::io("inspect retained installation image", &linked.output, error)
        })?;
        let cache = object::read::ReadCache::new(reader);
        crate::link::native_input::validate_linked_program(&cache, &product.target.target)?;
    }
    let image_digest = ContentDigest::from_sha256(*expected.image.bytes());
    let mut entries = vec![
        TreeEntry::file(
            expected.layout.executable.clone(),
            image_digest.clone(),
            source.len(),
            true,
        ),
        TreeEntry::directory(expected.layout.sysroot.clone()),
    ];
    for mut entry in sysroot.manifest.entries {
        entry.path = prefixed(&expected.layout.sysroot, &entry.path)?;
        entry.link_target = entry
            .link_target
            .as_ref()
            .map(|target| prefixed(&expected.layout.sysroot, target))
            .transpose()?;
        entries.push(entry);
    }
    let mut files = sysroot.files;
    files.entry(image_digest).or_insert(NativeFile::Open(image));
    for placement in &expected.layout.components {
        let bytes = contents
            .component_bytes(&placement.component)
            .ok_or_else(|| invalid("selected runtime bytes disappeared"))?;
        let descriptor = contents
            .manifest()
            .components
            .iter()
            .find(|entry| entry.id == placement.component)
            .ok_or_else(|| invalid("selected runtime descriptor disappeared"))?;
        if matches!(
            descriptor.kind,
            runmat_execution_artifact::target_pack::TargetPackComponentKind::NativeObject
                | runmat_execution_artifact::target_pack::TargetPackComponentKind::StaticArchive
        ) {
            crate::link::native_input::validate(bytes, descriptor.kind, &product.target.target)?;
        }
        let digest = ContentDigest::from_sha256(*descriptor.object.digest.bytes());
        entries.push(TreeEntry::file(
            placement.path.clone(),
            digest.clone(),
            bytes.len() as u64,
            false,
        ));
        files.entry(digest).or_insert(NativeFile::Bytes(bytes));
    }
    let payload = TreeManifest::new(entries).map_err(invalid)?;
    expected.storage.admit(&payload)?;
    let tree = ResolvedNativeTree {
        manifest: payload.clone(),
        files,
    };
    validate_loader_inventory(
        &tree,
        &product,
        &expected.layout,
        &expected.build,
        &expected.loader_images,
    )?;
    let target = runmat_native_codegen::NativeTarget::from_probe(
        contents.abi_probe().clone(),
        product.target.cpu_baseline,
    )
    .map_err(invalid)?;
    let manifest = NativeInstallationManifest {
        schema_version: NATIVE_INSTALLATION_SCHEMA_VERSION,
        product,
        target,
        image: expected.image,
        inputs: expected.inputs.clone(),
        build: expected.build.clone(),
        layout: expected.layout.clone(),
        storage: expected.storage,
        loader_images: expected.loader_images.clone(),
        deployment_qualification: expected.deployment_qualification,
        payload,
    };
    let bytes = crate::tree::canonical_json(&manifest, expected.storage.metadata_bytes)?;
    let mut digest = runmat_execution::DigestStream::new();
    digest.update(b"runmat.native.installation.v2\0");
    digest.update(&bytes);
    Ok(NativeInstallationBundle {
        tree,
        manifest,
        bytes,
        identity: digest.finish(),
        source_evaluation_issuer,
    })
}

fn prefixed(
    root: &NormalizedRelativePath,
    path: &NormalizedRelativePath,
) -> AotResult<NormalizedRelativePath> {
    NormalizedRelativePath::new(format!("{}/{}", root.as_str(), path.as_str())).map_err(invalid)
}

fn validate_loader_inventory(
    tree: &ResolvedNativeTree<'_>,
    product: &NativeSelectedProductBinding,
    layout: &NativeInstallationLayout,
    build: &NativeLinkBuildIdentity,
    expected: &[NativeLoaderImage],
) -> AotResult<()> {
    if expected.is_empty()
        || expected.len() > 65_536
        || expected.windows(2).any(|pair| pair[0].path >= pair[1].path)
    {
        return Err(invalid(
            "expected loader inventory must be bounded, sorted and unique",
        ));
    }
    let mut required = std::collections::BTreeSet::from([layout.executable.as_str().to_owned()]);
    for (dependency, path) in &build.hosted_dependencies {
        let installed = format!("{}/{}", layout.sysroot.as_str(), path.as_str());
        let bytes = tree
            .file(&installed)
            .ok_or_else(|| invalid("hosted dependency is not retained"))?;
        crate::link::native_input::validate_qualified_hosted_dependency(
            bytes,
            &product.target.target,
            dependency,
            path,
        )?;
        use object::Object;
        if object::File::parse(bytes).is_ok_and(|image| {
            matches!(
                image.kind(),
                object::ObjectKind::Dynamic | object::ObjectKind::Executable
            )
        }) {
            required.insert(installed);
        }
    }
    if required
        .iter()
        .any(|path| !expected.iter().any(|image| image.path.as_str() == path))
    {
        return Err(invalid("loader obligations omit a reached image"));
    }
    for image in expected {
        let references = match tree
            .file_source(image.path.as_str())
            .ok_or_else(|| invalid("loader image is not a retained file"))?
        {
            NativeFile::Bytes(bytes) => {
                if image.path != layout.executable {
                    crate::link::native_input::validate_hosted_dependency(
                        bytes,
                        &product.target.target,
                    )?;
                }
                loader::observe(*bytes)?
            }
            NativeFile::Open(file) => {
                let reader = file.try_clone().map_err(|error| {
                    AotError::io("inspect retained loader source", image.path.as_str(), error)
                })?;
                let cache = object::read::ReadCache::new(reader);
                loader::observe(&cache)?
            }
        };
        if references != image.references {
            return Err(invalid(
                "actual loader references differ from deployment expectations",
            ));
        }
    }
    Ok(())
}
