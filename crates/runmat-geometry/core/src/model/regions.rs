use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Region {
    pub region_id: String,
    pub name: String,
    pub tag: Option<String>,
}
