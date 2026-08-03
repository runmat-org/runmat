use std::collections::BTreeMap;

pub(super) fn capabilities() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("platform.service".to_string(), "launchd".to_string()),
        (
            "platform.process-isolation".to_string(),
            "rlimit".to_string(),
        ),
        ("process.resource-limits".to_string(), "true".to_string()),
    ])
}
