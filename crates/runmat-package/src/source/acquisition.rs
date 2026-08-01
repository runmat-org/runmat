use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SourceAcquisitionIntent {
    Execute,
    Fetch,
    Update,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SourceAcquisitionPolicy {
    pub locked: bool,
    pub frozen: bool,
    pub offline: bool,
}

#[cfg(test)]
mod tests {
    use super::SourceAcquisitionPolicy;

    #[test]
    fn omitted_wire_policy_fields_use_the_portable_defaults() {
        let policy: SourceAcquisitionPolicy = serde_json::from_str("{}").unwrap();
        assert_eq!(policy, SourceAcquisitionPolicy::default());
        assert!(serde_json::from_str::<SourceAcquisitionPolicy>(
            r#"{"locked":false,"unexpected":true}"#
        )
        .is_err());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SourceLockAction {
    Preserve,
    Write,
    Replace,
}
