use futures::executor::block_on;
use runmat_package_cache::backend::conformance::verify_backend_contract;
use runmat_package_cache_native::SqliteCacheBackend;

#[test]
fn sqlite_backend_obeys_portable_contract() {
    block_on(verify_backend_contract(|| {
        SqliteCacheBackend::open_in_memory(None).unwrap()
    }))
    .unwrap();
}
