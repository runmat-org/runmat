use runmat_hir::{DefPath, QualifiedName};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProductCacheKey {
    pub product: CacheProduct,
    pub source_hash: String,
    pub manifest_hash: String,
    pub dependency_graph_hash: String,
    pub config_hash: String,
    pub compiler_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheProduct {
    HirModule(QualifiedName),
    MirBody(DefPath),
    FunctionSummary(DefPath),
    ModuleSummary(QualifiedName),
    ClassMetadata(DefPath),
    AnalysisFacts(DefPath),
}
