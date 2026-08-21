use std::collections::BTreeMap;

pub(super) fn capabilities() -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            "runmat.version".to_string(),
            env!("CARGO_PKG_VERSION").to_string(),
        ),
        ("os".to_string(), std::env::consts::OS.to_string()),
        ("arch".to_string(), std::env::consts::ARCH.to_string()),
        ("execution.driver".to_string(), "v1".to_string()),
        ("execution.worker".to_string(), "v1".to_string()),
    ])
}
