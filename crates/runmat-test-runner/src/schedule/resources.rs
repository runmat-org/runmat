use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResourceRequirements {
    pub named: BTreeMap<String, u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceLease {
    pub identity: String,
    pub resources: ResourceRequirements,
}
