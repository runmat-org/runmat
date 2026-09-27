//! Immutable declaration snapshots. Capture the complete program before any
//! operation-owned admission awaits; inference and hashing reuse these facts.
use super::{FusionGroupPlan, FusionOp};
use crate::graph::PrimitiveOp;
use anyhow::{anyhow, Result};
use runmat_builtins::{builtin_catalog_entry_by_name, BuiltinCatalogEntry};
use runmat_runtime::builtins::common::spec::{
    canonical_fusion_declaration, CanonicalFusionDeclaration,
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct SemanticBinding {
    pub entry: &'static BuiltinCatalogEntry,
    pub declaration: CanonicalFusionDeclaration,
}
impl SemanticBinding {
    pub fn selected(declaration: CanonicalFusionDeclaration) -> Option<Self> {
        let entry = builtin_catalog_entry_by_name(declaration.binding.builtin.name)?;
        entry
            .bindings
            .iter()
            .any(|binding| entry.binding_identity(binding) == declaration.binding)
            .then_some(Self { entry, declaration })
    }
    pub fn capture(operation: &FusionOp) -> Result<Option<Self>> {
        let entry = match operation {
            FusionOp::Builtin { name, .. } => builtin_catalog_entry_by_name(name)
                .ok_or_else(|| anyhow!("fusion builtin has no canonical type authority"))?,
            FusionOp::Primitive {
                op: PrimitiveOp::ElemPow,
                ..
            } => &runmat_builtins::POWER_CATALOG_ENTRY,
            FusionOp::Primitive { .. } => return Ok(None),
        };
        let [binding] = entry.bindings else {
            return Err(anyhow!("fusion builtin binding is ambiguous"));
        };
        let declaration = canonical_fusion_declaration(entry.binding_identity(binding))
            .ok_or_else(|| anyhow!("missing canonical fusion declaration"))?;
        Ok(Some(Self { entry, declaration }))
    }
}
pub(crate) fn matches(operation: &FusionOp, snapshot: Option<SemanticBinding>) -> bool {
    match (operation, snapshot) {
        (FusionOp::Builtin { name, .. }, Some(binding)) => {
            name == binding.declaration.binding.builtin.name
        }
        (
            FusionOp::Primitive {
                op: PrimitiveOp::ElemPow,
                ..
            },
            Some(binding),
        ) => binding.entry.identity == runmat_builtins::POWER_CATALOG_ENTRY.identity,
        (FusionOp::Primitive { op, .. }, None) => *op != PrimitiveOp::ElemPow,
        _ => false,
    }
}

pub(crate) fn capture(plan: &FusionGroupPlan) -> Result<Vec<Option<SemanticBinding>>> {
    plan.operations
        .iter()
        .map(SemanticBinding::capture)
        .collect()
}
