fn strongest_available_isolation_fixture() {
    use runmat_test_runner::host::{HostCapabilities, IsolationMode};

    let browser = HostCapabilities::new(
        [
            IsolationMode::Worker,
            IsolationMode::Session,
            IsolationMode::None,
        ],
        4,
    )
    .unwrap();
    assert_eq!(
        browser.resolve(IsolationMode::Auto).unwrap(),
        IsolationMode::Worker
    );
    assert!(browser.resolve(IsolationMode::Process).is_err());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn strongest_available_isolation_is_host_honest() {
    strongest_available_isolation_fixture();
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::strongest_available_isolation_fixture;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn strongest_available_isolation_is_host_honest() {
        strongest_available_isolation_fixture();
    }
}
