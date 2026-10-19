//! Process-local identity issued by a verified native product host.
//! It authenticates service selection, not execution or capability grants.

use super::{
    NativeDeploymentServiceKind, NativeSelectedProductBinding, NativeSelectedServiceBinding,
};
use crate::{ContractError, Digest};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedNativeServiceIssuer {
    product: Digest,
    binding: NativeSelectedServiceBinding,
}

impl VerifiedNativeServiceIssuer {
    pub fn product(&self) -> Digest {
        self.product
    }

    pub fn binding(&self) -> &NativeSelectedServiceBinding {
        &self.binding
    }

    /// Issue an identity after the owning host has verified the selected
    /// product and exact service component. Installing a live implementation
    /// and enforcing the independent caller grant remain host obligations.
    ///
    /// # Safety
    /// The caller must have authenticated the selected product and service
    /// implementation bytes. A decoded manifest alone is insufficient. A
    /// linked-image host must also match the actual service factory symbol.
    pub unsafe fn from_verified_product(
        product: &NativeSelectedProductBinding,
        kind: NativeDeploymentServiceKind,
    ) -> Result<Self, ContractError> {
        product.validate()?;
        let invalid = |message| ContractError::invalid("native.service_issuer", message);
        // Product validation proves `services` is exactly the union of
        // reached-effect services and explicitly required non-effect services.
        // The latter set alone excludes legitimate reached effects.
        let binding = product
            .services
            .iter()
            .find(|service| service.kind == kind)
            .cloned()
            .ok_or_else(|| invalid("selected product lacks its service implementation"))?;
        Ok(Self {
            product: product.fingerprint()?,
            binding,
        })
    }
}
