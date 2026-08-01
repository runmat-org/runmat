use futures::executor::block_on;
use runmat_package_cache::backend::conformance::{verify_backend_contract, MemoryBackend};

#[test]
fn memory_backend_obeys_portable_contract() {
    block_on(verify_backend_contract(MemoryBackend::new)).unwrap();
}
