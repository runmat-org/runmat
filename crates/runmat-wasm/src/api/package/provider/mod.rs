mod git;
mod registry;
mod server;
pub(super) mod shared;

use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;

pub(super) struct JsPackageSourceProvider {
    pub(super) bindings: JsValue,
    pub(super) lease_owner: String,
    pub(super) temporary_leases: RefCell<Vec<runmat_package_cache::Lease>>,
    next_lease: Cell<u32>,
}

impl JsPackageSourceProvider {
    pub(super) fn new(bindings: JsValue) -> Result<Self, JsValue> {
        let lease_owner = js_sys::Reflect::get(&bindings, &JsValue::from_str("leaseOwner"))?
            .as_string()
            .ok_or_else(|| JsValue::from_str("package provider leaseOwner must be a string"))?;
        runmat_package_cache::LeaseOwner::new(lease_owner.clone()).map_err(JsValue::from_str)?;
        Ok(Self {
            bindings,
            lease_owner,
            temporary_leases: RefCell::new(Vec::new()),
            next_lease: Cell::new(0),
        })
    }

    pub(super) fn temporary_lease_id(&self) -> String {
        let next = self.next_lease.get();
        self.next_lease.set(next.saturating_add(1));
        format!("{}-acquire-{next}", self.lease_owner)
    }
}

impl runmat_package::PackageSourceProvider for JsPackageSourceProvider {
    fn acquire_git<'a>(
        &'a self,
        plan: &'a runmat_package::GitAcquisitionPlan,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<runmat_package::GitPackageMount, String>> + 'a>,
    > {
        Box::pin(git::acquire(self, plan))
    }

    fn acquire_server_project<'a>(
        &'a self,
        plan: &'a runmat_package::ServerProjectAcquisitionPlan,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<runmat_package::ServerProjectPackageMount, String>,
                > + 'a,
        >,
    > {
        Box::pin(server::acquire(self, plan))
    }

    fn acquire_registry<'a>(
        &'a self,
        plan: &'a runmat_package::RegistryAcquisitionPlan,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<runmat_package::RegistryPackageMount, String>>
                + 'a,
        >,
    > {
        Box::pin(registry::acquire(self, plan))
    }

    fn registry_candidates<'a>(
        &'a self,
        plan: &'a runmat_package::RegistryCandidatePlan,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<Vec<runmat_package::RegistryCandidateRecord>, String>,
                > + 'a,
        >,
    > {
        Box::pin(registry::candidates(self, plan))
    }
}
