//! External expected tool qualification and physical build-host admission.
//! None of these rights can be decoded from a pack's self-description.
use runmat_execution::host::{
    NativeArchitecture, NativeEndianness, NativeEnvironment, NativeOperatingSystem,
};
use runmat_execution::Digest;
use runmat_package::NormalizedRelativePath;
use std::collections::BTreeMap;

use crate::tree::{NativeStorageGrant, ResolvedNativeTree};
use crate::{AotError, AotResult};

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeLinkExecutionHost {
    pub architecture: NativeArchitecture,
    pub operating_system: NativeOperatingSystem,
    pub environment: NativeEnvironment,
    pub endianness: NativeEndianness,
    pub pointer_width: u16,
}
impl NativeLinkExecutionHost {
    pub fn current() -> AotResult<Self> {
        let current = runmat_native_codegen::NativeTarget::current();
        let (_, probe) = current.native_parts().map_err(invalid)?;
        Ok(Self {
            architecture: probe.architecture,
            operating_system: probe.operating_system,
            environment: probe.environment,
            endianness: probe.endianness,
            pointer_width: probe.pointer_width,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeLinkAdapter {
    /// The companion linker is also an exact executable in the same admitted
    /// tree; clang may not discover a linker through PATH or a host SDK.
    ClangDriver {
        linker: NormalizedRelativePath,
    },
    MsvcLinker,
}

/// Locator identity, not a service or semantic-operation registry. The
/// framework locator addresses its link image inside the complete framework
/// tree; staging never flattens its resources or rewrites its install name.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
#[serde(
    tag = "kind",
    content = "name",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum NativeHostedDependency {
    Library(String),
    Framework(String),
}

/// Recorded identity, NOT an admission/qualification right. This can be
/// retained with a completed output digest without accepting it as execution.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeLinkBuildIdentity {
    pub product: Digest,
    pub execution_host: NativeLinkExecutionHost,
    pub adapter: NativeLinkAdapter,
    pub entrypoint: NormalizedRelativePath,
    pub toolchain: Digest,
    pub sysroot: Digest,
    pub qualification: Digest,
    pub storage: NativeStorageGrant,
    /// A sequence, not a JSON map with complex enum keys.
    pub hosted_dependencies: Vec<(NativeHostedDependency, NormalizedRelativePath)>,
}

/// Trusted caller input obtained from the actual tool qualification owner,
/// including its build-host loader/dependency contract. A checksum retained in
/// the pack is NOT a qualification. Matching this record is structural link
/// admission; only the later completed process can witness actual linking.
pub struct ExternalNativeLinkAdmission {
    pub product: Digest,
    pub execution_host: NativeLinkExecutionHost,
    pub adapter: NativeLinkAdapter,
    pub entrypoint: NormalizedRelativePath,
    pub toolchain: Digest,
    pub sysroot: Digest,
    pub qualification: Digest,
    /// Independent caller materialization permission, not a manifest claim.
    pub storage: NativeStorageGrant,
    /// Exact reached hosted dependency locators. Runtime availability/install
    /// semantics remain a separately qualified hosted deployment obligation.
    pub hosted_dependencies: BTreeMap<NativeHostedDependency, NormalizedRelativePath>,
}

impl ExternalNativeLinkAdmission {
    pub(super) fn identity(&self) -> NativeLinkBuildIdentity {
        NativeLinkBuildIdentity {
            product: self.product,
            execution_host: self.execution_host.clone(),
            adapter: self.adapter.clone(),
            entrypoint: self.entrypoint.clone(),
            toolchain: self.toolchain,
            sysroot: self.sysroot,
            qualification: self.qualification,
            storage: self.storage,
            hosted_dependencies: self
                .hosted_dependencies
                .iter()
                .map(|(key, path)| (key.clone(), path.clone()))
                .collect(),
        }
    }
    pub(super) fn validate(&self, product: Digest, tree: &ResolvedNativeTree<'_>) -> AotResult<()> {
        if self.product != product
            || self.execution_host != NativeLinkExecutionHost::current()?
            || [
                self.product,
                self.toolchain,
                self.sysroot,
                self.qualification,
            ]
            .iter()
            .any(|digest| digest.bytes().iter().all(|byte| *byte == 0))
        {
            return Err(invalid(
                "trusted link admission differs from product or actual build host",
            ));
        }
        self.validate_executable(tree, &self.entrypoint)?;
        if let NativeLinkAdapter::ClangDriver { linker } = &self.adapter {
            self.validate_executable(tree, linker)?;
        }
        Ok(())
    }

    fn validate_executable(
        &self,
        tree: &ResolvedNativeTree<'_>,
        entrypoint: &NormalizedRelativePath,
    ) -> AotResult<()> {
        let bytes = tree.executable(entrypoint.as_str()).ok_or_else(|| {
            invalid("admitted tool entrypoint is not an executable regular tree member")
        })?;
        validate_host_image(bytes, &self.execution_host)
    }
}

fn validate_host_image(bytes: &[u8], host: &NativeLinkExecutionHost) -> AotResult<()> {
    use object::Object;
    let architecture = match host.architecture {
        NativeArchitecture::X86_64 => object::Architecture::X86_64,
        NativeArchitecture::Aarch64 => object::Architecture::Aarch64,
    };
    let bytes = match object::FileKind::parse(bytes).map_err(invalid)? {
        object::FileKind::MachOFat32 => {
            let fat = object::read::macho::MachOFatFile32::parse(bytes).map_err(invalid)?;
            host_slice(fat.arches(), bytes, architecture)?
        }
        object::FileKind::MachOFat64 => {
            let fat = object::read::macho::MachOFatFile64::parse(bytes).map_err(invalid)?;
            host_slice(fat.arches(), bytes, architecture)?
        }
        _ => bytes,
    };
    let image = object::File::parse(bytes).map_err(invalid)?;
    let format = match host.operating_system {
        NativeOperatingSystem::Linux => object::BinaryFormat::Elf,
        NativeOperatingSystem::Macos => object::BinaryFormat::MachO,
        NativeOperatingSystem::Windows => object::BinaryFormat::Pe,
    };
    if image.architecture() != architecture
        || image.sub_architecture().is_some()
        || image.format() != format
        || image.is_64() != (host.pointer_width == 64)
        || image.is_little_endian() != (host.endianness == NativeEndianness::Little)
    {
        return Err(invalid(
            "qualified tool image is not executable by the admitted build host",
        ));
    }
    super::native_input::require_image_purpose(
        &image,
        super::native_input::NativeImagePurpose::FinalExecutable,
    )
}

fn host_slice<'a, A: object::read::macho::FatArch>(
    arches: &[A],
    bytes: &'a [u8],
    architecture: object::Architecture,
) -> AotResult<&'a [u8]> {
    let mut matches = arches
        .iter()
        .filter(|arch| arch.architecture() == architecture);
    let arch = matches
        .next()
        .ok_or_else(|| invalid("universal tool image lacks build-host slice"))?;
    if matches.next().is_some() {
        return Err(invalid("universal tool image duplicates build-host slice"));
    }
    arch.data(bytes).map_err(invalid)
}

fn invalid(error: impl std::fmt::Display) -> AotError {
    AotError::contract("aot.link.toolchain", error.to_string())
}
