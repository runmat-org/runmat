//! Fminunc's original model policy adapts to the one shared retained owner.
use super::*;
use crate::builtins::math::optim::common::trust_quadratic::{self, model as shared};
use crate::execution::RootedValue;
pub(super) use shared::Products;

pub(super) fn policy() -> trust_quadratic::Policy {
    trust_quadratic::Policy {
        identity: runmat_builtins::BuiltinCatalogIdentity { name: NAME },
        ownership_identifier: "RunMat:fminunc:HessianOwnership",
        value_source: Some(trust_quadratic::ValueSourcePolicy {
            numeric: &FMINUNC_CALLBACK_NUMERIC_EXTENSION,
            resident: &FMINUNC_RESIDENT_EXTENSION,
            product_preflight: None,
        }),
        invalid,
        allocation,
    }
}
pub(super) struct Model(shared::Model);
impl Model {
    pub async fn matrix(
        source: RootedValue,
        n: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        shared::Model::matrix(policy(), source, n, cooperation)
            .await
            .map(Self)
    }
    pub fn multiply(info: RootedValue, function: Value) -> Self {
        Self(shared::Model::multiply(info, function, policy()))
    }
    pub async fn products<'a>(
        &'a mut self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Products<'a>> {
        self.0.products(cooperation).await
    }
    pub fn into_value(self) -> RootedValue {
        self.0.into_value()
    }
}
