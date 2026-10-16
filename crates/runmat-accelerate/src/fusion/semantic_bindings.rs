//! Immutable declaration snapshots. Capture the complete program before any
//! operation-owned admission awaits; inference and hashing reuse these facts.
use super::{FusionGroupPlan, FusionOp};
use crate::graph::PrimitiveOp;
use anyhow::Result;
use runmat_builtins::{builtin_catalog_entry_by_name, BuiltinBindingIdentity, BuiltinCatalogEntry};
use runmat_runtime::builtins::common::spec::{
    canonical_fusion_declaration, selected_builtin_binding, CanonicalFusionDeclaration,
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct SemanticBinding {
    pub entry: &'static BuiltinCatalogEntry,
    pub declaration: CanonicalFusionDeclaration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SemanticBindingCaptureError {
    MissingCatalogAuthority,
    AmbiguousBinding,
    MissingFusionDeclaration,
}

impl std::fmt::Display for SemanticBindingCaptureError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::MissingCatalogAuthority => "fusion builtin has no canonical type authority",
            Self::AmbiguousBinding => "fusion builtin binding is ambiguous",
            Self::MissingFusionDeclaration => "missing canonical fusion declaration",
        })
    }
}

impl std::error::Error for SemanticBindingCaptureError {}

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
        Self::capture_typed(operation).map_err(Into::into)
    }

    /// Same canonical lookup as `capture`, without constructing an anyhow
    /// diagnostic before a finite preparation receipt has been admitted.
    pub(crate) fn capture_typed(
        operation: &FusionOp,
    ) -> std::result::Result<Option<Self>, SemanticBindingCaptureError> {
        let entry = match operation {
            FusionOp::Builtin { name, .. } => builtin_catalog_entry_by_name(name)
                .ok_or(SemanticBindingCaptureError::MissingCatalogAuthority)?,
            FusionOp::Primitive {
                op: PrimitiveOp::ElemPow,
                ..
            } => &runmat_builtins::POWER_CATALOG_ENTRY,
            FusionOp::Primitive { .. } => return Ok(None),
        };
        let [binding] = entry.bindings else {
            return Err(SemanticBindingCaptureError::AmbiguousBinding);
        };
        let declaration = canonical_fusion_declaration(entry.binding_identity(binding))
            .ok_or(SemanticBindingCaptureError::MissingFusionDeclaration)?;
        Ok(Some(Self { entry, declaration }))
    }
}

/// A fusion primitive has no named call snapshot, but its ordinary numeric VM
/// instruction still selects a canonical builtin. Resolve that implementation
/// through the same runtime dispatch authority before domain admission awaits.
/// Other primitive instructions have no certified host route yet.
pub(crate) fn selected_host_binding_for_primitive(
    op: PrimitiveOp,
    runtime: Option<&runmat_runtime::context::RuntimeContext>,
) -> Option<BuiltinBindingIdentity> {
    if !matches!(
        op,
        PrimitiveOp::Add
            | PrimitiveOp::Sub
            | PrimitiveOp::ElemMul
            | PrimitiveOp::ElemDiv
            | PrimitiveOp::ElemLeftDiv
            | PrimitiveOp::ElemPow
    ) {
        return None;
    }
    let name = op.operator_kind().overload_name()?;
    selected_canonical_host_binding(name.display_name(), runtime)
}

pub(crate) fn selected_canonical_host_binding(
    name: &str,
    runtime: Option<&runmat_runtime::context::RuntimeContext>,
) -> Option<BuiltinBindingIdentity> {
    let entry = builtin_catalog_entry_by_name(name)?;
    let [binding] = entry.bindings else {
        return None;
    };
    let expected = entry.binding_identity(binding);
    let selected = selected_builtin_binding(entry.identity.name, Some(expected), runtime)?;
    canonical_fusion_declaration(selected.identity)?;
    Some(selected.identity)
}
pub(crate) fn matches(operation: &FusionOp, snapshot: Option<SemanticBinding>) -> bool {
    match (operation, snapshot) {
        // Selection already resolved spelling/case/aliases through the owning
        // registry. The caller maps this snapshot by original operation index;
        // a second textual equality test would create a conflicting authority.
        (FusionOp::Builtin { .. }, Some(_)) => true,
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
    let mut snapshots = Vec::with_capacity(plan.operations.len());
    for operation in &plan.operations {
        snapshots.push(SemanticBinding::capture(operation)?);
    }
    Ok(snapshots)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_capture_preserves_public_binding_and_error_authority() {
        let power = FusionOp::Primitive {
            op: PrimitiveOp::ElemPow,
            inputs: vec![0, 1],
            output: Some(2),
        };
        let typed = SemanticBinding::capture_typed(&power).unwrap().unwrap();
        let public = SemanticBinding::capture(&power).unwrap().unwrap();
        assert_eq!(typed.declaration.binding, public.declaration.binding);

        let missing = FusionOp::Builtin {
            name: "__missing_fusion_binding__".into(),
            origin: crate::graph::FusionCallOrigin::NamedUnresolved,
            inputs: Vec::new(),
            output: None,
        };
        assert_eq!(
            SemanticBinding::capture_typed(&missing).unwrap_err(),
            SemanticBindingCaptureError::MissingCatalogAuthority,
        );
        assert_eq!(
            SemanticBinding::capture(&missing).unwrap_err().to_string(),
            "fusion builtin has no canonical type authority",
        );
    }
}
