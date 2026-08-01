use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::identity::ParameterId;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterDescriptor {
    pub id: ParameterId,
    pub name: String,
    pub normalized_identity: String,
    pub value: Value,
}
