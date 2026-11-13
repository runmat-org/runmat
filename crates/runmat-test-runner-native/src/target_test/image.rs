use std::io::Read as _;

use runmat_aot::link::LinkedProgram;
use runmat_execution::Digest;
use runmat_execution_artifact::target_test::{NativePairedProductHeader, NativePairedProductRole};
use runmat_execution_artifact::{PrivateArtifactLease, PrivateArtifactRoot};
use runmat_process_host::child::{ProcessRetirementLease, ProcessSpawnEvidence};
use runmat_process_host::HostCommand;

use super::NativeTargetTestRunLimits;
use crate::{NativeRunnerError, NativeRunnerResult};

/// The pending owner covers cancellation during OS spawn, before the normal
/// contained-child owner can take responsibility for these exact bytes.
pub(super) struct PendingImage {
    lease: Option<PrivateArtifactLease>,
    pub command: HostCommand,
    pub spawn: ProcessSpawnEvidence,
}

impl PendingImage {
    pub fn prepare(
        linked: &LinkedProgram,
        header: &NativePairedProductHeader,
        limits: NativeTargetTestRunLimits,
    ) -> NativeRunnerResult<Self> {
        Self::prepare_role(linked, header, NativePairedProductRole::Test, limits)
    }

    pub(super) fn prepare_role(
        linked: &LinkedProgram,
        header: &NativePairedProductHeader,
        role: NativePairedProductRole,
        limits: NativeTargetTestRunLimits,
    ) -> NativeRunnerResult<Self> {
        if limits.max_image_bytes == 0
            || limits.max_stdout_bytes == 0
            || limits.max_stderr_bytes == 0
        {
            return Err(configuration("native test host limits must be nonzero"));
        }
        header.validate().map_err(configuration)?;
        let image = linked
            .verified_selected_image()
            .map_err(configuration)?
            .ok_or_else(|| configuration("native target tests require a selected-image receipt"))?;
        let receipt = image.canonical_plan();
        if header.role != role
            || linked.product.selected() != Some(&header.binding.selected_product)
            || receipt.record().product != header.binding.selected_product
            || receipt.record().inputs != linked.inputs
        {
            return Err(configuration(
                "test header differs from selected link product",
            ));
        }
        let file = std::fs::File::open(&linked.output)?;
        let length = file.metadata()?.len();
        if length == 0 || length > limits.max_image_bytes {
            return Err(configuration(
                "native test image exceeds the artifact bound",
            ));
        }
        // Bound the read even if an ambient producer replaces/grows its output.
        // Execution uses only our private copy after checking the linker digest.
        let mut bytes = Vec::new();
        file.take(length.saturating_add(1))
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 != length || Digest::sha256(&bytes) != image.output_digest() {
            return Err(configuration("native test image differs from linked bytes"));
        }
        let mut root =
            PrivateArtifactRoot::new("runmat-native-target-test-").map_err(configuration)?;
        let path = root
            .write_verified(if cfg!(windows) { "test.exe" } else { "test" }, &bytes)
            .map_err(configuration)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(path.as_path(), std::fs::Permissions::from_mode(0o700))?;
        }
        let mut command = HostCommand::new(path.as_path());
        command.environment_policy = runmat_process_host::environment::EnvironmentPolicy::Allow(
            runmat_process_host::environment::EnvironmentAllowlist::platform_runtime(),
        );
        command.max_stderr_bytes = limits.max_stderr_bytes;
        // No ambient configuration or hidden-mode arguments select a test.
        // Runtime deployment services come from the admitted linked product.
        command.working_directory = Some(root.path().to_path_buf());
        Ok(Self {
            lease: Some(root.lease()),
            command,
            spawn: ProcessSpawnEvidence::default(),
        })
    }

    pub fn take_lease(&mut self) -> ImageLease {
        ImageLease(self.lease.take().expect("one child owns the image lease"))
    }
}

impl Drop for PendingImage {
    fn drop(&mut self) {
        if self.spawn.is_unverified() {
            if let Some(lease) = &self.lease {
                if let Err(error) =
                    lease.persist_unverified(None, "native target-test spawn unverified")
                {
                    eprintln!("{error}");
                }
            }
        }
    }
}

pub(super) struct ImageLease(PrivateArtifactLease);

impl ProcessRetirementLease for ImageLease {
    fn persist_unverified(&mut self, process_id: Option<u32>, reason: &str) -> Result<(), String> {
        self.0.persist_unverified(process_id, reason)
    }
}

fn configuration(message: impl std::fmt::Display) -> NativeRunnerError {
    NativeRunnerError::Configuration(message.to_string())
}
