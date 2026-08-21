use crate::{MirBody, MirFunctionMetadata};
use runmat_hir::FunctionId;
use runmat_types::ClassDeclaration;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MirAssembly {
    pub bodies: BTreeMap<FunctionId, MirBody>,
    pub functions: BTreeMap<FunctionId, MirFunctionMetadata>,
    pub classes: Vec<ClassDeclaration>,
    pub entrypoints: Vec<FunctionId>,
}
