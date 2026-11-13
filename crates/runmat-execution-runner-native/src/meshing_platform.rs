//! Worker-derived meshing platform identity used by local and remote evidence.

use runmat_execution::Digest;
use runmat_meshing_core::{PlatformBuildIdentity, StableDigest};

pub fn native_meshing_capability_cohort() -> String {
    format!("native-{}-{}", std::env::consts::ARCH, std::env::consts::OS)
}

pub(crate) fn native_meshing_platform(
    capability_cohort: Option<&str>,
    exact_kernel_abi: Option<String>,
) -> PlatformBuildIdentity {
    PlatformBuildIdentity {
        capability_cohort: capability_cohort
            .map(str::to_owned)
            .unwrap_or_else(native_meshing_capability_cohort),
        target_triple: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
        build_digest: StableDigest::from_bytes(
            *Digest::sha256(
                format!("runmat-meshing-build:{}", env!("CARGO_PKG_VERSION")).as_bytes(),
            )
            .bytes(),
        ),
        exact_kernel_abi,
    }
}
