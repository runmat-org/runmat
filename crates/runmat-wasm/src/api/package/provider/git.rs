use super::{shared, JsPackageSourceProvider};
use runmat_package_cache::{CacheBackend, CacheError, CommitOutcome};

pub(super) async fn acquire(
    provider: &JsPackageSourceProvider,
    plan: &runmat_package::GitAcquisitionPlan,
) -> Result<runmat_package::GitPackageMount, String> {
    let cache = shared::cache(provider)?;
    let mut snapshot = None;
    if let Some(expected) = &plan.expected {
        match shared::lease_expected(provider, &cache, &expected.tree_digest).await {
            Ok(Some(lease)) => {
                match runmat_package_cache::load_git_snapshot(&cache, expected.clone()).await {
                    Ok(cached) => {
                        provider.temporary_leases.borrow_mut().push(lease);
                        snapshot = Some(cached);
                    }
                    Err(CacheError::Miss(_)) => {
                        let _ = runmat_package_cache::release_lease(&cache, &lease, 16).await;
                    }
                    Err(error) => return Err(error.to_string()),
                }
            }
            Ok(None) => unreachable!("expected Git tree is a lease root"),
            Err(CacheError::Miss(_)) => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    if snapshot.is_none() {
        if !plan.allow_network {
            return Err(format!(
                "package cache miss for Git source `{}` while network access is disabled",
                plan.repository
            ));
        }
        let plan_value = serde_wasm_bindgen::to_value(plan)
            .map_err(|error| format!("Git acquisition plan serialization failed: {error}"))?;
        let inventory_value =
            shared::call_provider(&provider.bindings, "fetchGitInventory", &plan_value).await?;
        let inventory: runmat_package_cache::GitTreeInventory =
            serde_wasm_bindgen::from_value(inventory_value)
                .map_err(|error| format!("Git inventory parse failed: {error}"))?;
        let acquired = inventory
            .into_snapshot(
                plan.repository.as_str(),
                plan.subdir.as_str(),
                runmat_package_cache::ArchiveLimits::default(),
            )
            .map_err(|error| error.to_string())?;
        runmat_package::validate_git_acquisition(plan, &acquired.source)
            .map_err(|error| error.to_string())?;
        loop {
            let current = cache.snapshot().await.map_err(|error| error.to_string())?;
            let transaction = runmat_package_cache::cache_git_snapshot(
                current.revision,
                current.state,
                &acquired,
                js_sys::Date::now().max(0.0) as u64,
            )
            .map_err(|error| error.to_string())?;
            match cache
                .commit(transaction)
                .await
                .map_err(|error| error.to_string())?
            {
                CommitOutcome::Committed(_) => break,
                CommitOutcome::Conflict { .. } => continue,
            }
        }
        snapshot = Some(acquired);
    }
    let snapshot = snapshot.expect("cache hit or fetched snapshot");
    runmat_package::validate_git_acquisition(plan, &snapshot.source)
        .map_err(|error| error.to_string())?;
    shared::retain_tree(provider, &cache, &snapshot.tree.digest).await?;
    let root = shared::mount(provider, &snapshot).await?;
    Ok(runmat_package::GitPackageMount {
        source: snapshot.source,
        root,
    })
}
