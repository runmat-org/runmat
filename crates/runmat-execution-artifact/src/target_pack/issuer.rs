//! Issue a neutral service identity from this owner's verified pack bytes.

use super::{ArtifactResult, TargetPackServiceKind, VerifiedTargetPackContents};
use runmat_execution::host::VerifiedNativeServiceIssuer;

impl VerifiedTargetPackContents<'_> {
    pub fn verified_service_issuer(
        &self,
        kind: TargetPackServiceKind,
    ) -> ArtifactResult<VerifiedNativeServiceIssuer> {
        let product = self.selected_product_binding()?;
        // SAFETY: this owner has verified the selected product and its exact
        // component bytes. It does not install a port or grant execution.
        unsafe { VerifiedNativeServiceIssuer::from_verified_product(&product, kind) }
            .map_err(super::contract)
    }
}
