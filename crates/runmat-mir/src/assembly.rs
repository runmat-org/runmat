use crate::MirBody;
use runmat_hir::FunctionId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MirAssembly {
    pub bodies: HashMap<FunctionId, MirBody>,
}
