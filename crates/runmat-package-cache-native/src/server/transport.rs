use futures::future::LocalBoxFuture;
use runmat_package::ServerProjectAcquisitionPlan;
use runmat_package_cache::ServerProjectTreeInventory;

pub trait ServerSnapshotTransport: Send + Sync {
    fn fetch<'a>(
        &'a self,
        plan: &'a ServerProjectAcquisitionPlan,
    ) -> LocalBoxFuture<'a, Result<ServerProjectTreeInventory, String>>;
}
