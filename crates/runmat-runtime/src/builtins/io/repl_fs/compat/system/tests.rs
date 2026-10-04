use super::*;
use crate::context::{ParallelCapability, RuntimeParallelResources, RuntimeParallelService};
use std::rc::Rc;

#[cfg(unix)]
#[path = "tests/revocation.rs"]
mod revocation;

struct Finite;

impl RuntimeParallelService for Finite {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> RuntimeParallelResources {
        RuntimeParallelResources {
            memory_available_bytes: Some(1),
            ..Default::default()
        }
    }
}

#[test]
fn finite_resource_budget_rejects_unbounded_process_before_spawn() {
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            crate::context::RuntimeServicePorts::default().with_parallel(Rc::new(Finite)),
        );
    let error = futures::executor::block_on(run(Some(&runtime), "printf no-process"))
        .expect_err("unbounded route cannot claim a finite certificate");
    assert_eq!(
        error.identifier.as_deref(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn virtual_provider_cannot_be_used_as_native_process_cwd() {
    let _serial = runmat_filesystem::provider_override_lock();
    let provider = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        "/virtual",
    ));
    let _selected = runmat_filesystem::replace_provider(provider);
    let error = futures::executor::block_on(run(None, "printf no-process"))
        .expect_err("virtual cwd cannot escape to an OS process");
    assert_eq!(
        error.identifier.as_deref(),
        Some("RunMat:system:UnsupportedFilesystem")
    );
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
async fn browser_process_rejects_without_spawning() {
    let error = run(None, "printf no-process")
        .await
        .expect_err("browser has no process host");
    assert_eq!(
        error.identifier.as_deref(),
        Some("RunMat:system:UnsupportedOnWasm")
    );
}
