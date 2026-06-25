use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssemblyNode {
    pub node_id: String,
    pub label: String,
    pub children: Vec<AssemblyNode>,
}
