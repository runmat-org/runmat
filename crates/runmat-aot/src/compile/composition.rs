//! Runtime composition from the canonical reachability/catalog closure.
use super::CompilationPolicy;
use crate::{archive::RuntimeArchiveCapabilities, AotError, AotResult};
use runmat_mir::analysis::ReachabilityReport;
use runmat_native_codegen::aot::{AotBuiltinBinding, AotRuntimeBindingMode};
use runmat_types::BuiltinId;

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum RuntimeComposition {
    Selected { bindings: Vec<AotBuiltinBinding> },
    Dynamic { reason: DynamicCompositionReason },
}
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum DynamicCompositionReason {
    Requested,
    ArchiveCapability,
    UnknownTargets,
    MissingCatalog { builtin: BuiltinId },
    DynamicBuiltin { builtin: BuiltinId },
    MissingBindings { builtin: BuiltinId },
    TargetConditional { builtin: BuiltinId },
}
impl RuntimeComposition {
    /// Decoded reports do not grant selected-product or native-link authority.
    pub fn select(
        policy: CompilationPolicy,
        capabilities: &RuntimeArchiveCapabilities,
        reachability: &ReachabilityReport,
    ) -> AotResult<Self> {
        policy.validate(capabilities)?;
        if policy == CompilationPolicy::DynamicRuntime {
            return Ok(Self::Dynamic {
                reason: DynamicCompositionReason::Requested,
            });
        }
        if !capabilities.closed_world_linking {
            return Ok(Self::Dynamic {
                reason: DynamicCompositionReason::ArchiveCapability,
            });
        }
        if policy == CompilationPolicy::ClosedWorld {
            return closed_world_bindings(reachability).map(|bindings| Self::Selected { bindings });
        }
        match closure_bindings(reachability) {
            Ok(bindings) => Ok(Self::Selected { bindings }),
            Err(reason) => Ok(Self::Dynamic { reason }),
        }
    }
    pub fn binding_mode(&self) -> AotRuntimeBindingMode {
        match self {
            Self::Selected { .. } => AotRuntimeBindingMode::ClosedWorld,
            Self::Dynamic { .. } => AotRuntimeBindingMode::Dynamic,
        }
    }
    pub fn bindings(&self) -> &[AotBuiltinBinding] {
        match self {
            Self::Selected { bindings } => bindings,
            Self::Dynamic { .. } => &[],
        }
    }
    pub fn is_selected(&self) -> bool {
        matches!(self, Self::Selected { .. })
    }
}
impl DynamicCompositionReason {
    fn strict_error(self) -> AotError {
        let (code, message) = match self {
            Self::UnknownTargets => (
                "aot.compile.reachability_unknown",
                "closed-world compilation cannot retain an unbounded runtime call target".into(),
            ),
            Self::MissingCatalog { builtin } => (
                "aot.compile.closed_world_builtin",
                format!("closed-world compilation requires builtin `{}` to have a canonical catalog runtime binding", builtin.0),
            ),
            Self::DynamicBuiltin { builtin } => (
                "aot.compile.closed_world_builtin_dynamic",
                format!("closed-world compilation cannot prove the complete runtime target set for builtin `{}`", builtin.0),
            ),
            Self::MissingBindings { builtin } => (
                "aot.compile.closed_world_builtin",
                format!("closed-world builtin `{}` has no runtime binding declaration", builtin.0),
            ),
            Self::TargetConditional { builtin } => (
                "aot.compile.closed_world_builtin_target",
                format!("closed-world compilation cannot select target-conditional runtime bindings for builtin `{}`", builtin.0),
            ),
            Self::Requested | Self::ArchiveCapability => unreachable!("policy admission precedes closure proof"),
        };
        AotError::contract(code, message)
    }
}
pub(super) fn closed_world_bindings(
    reachability: &ReachabilityReport,
) -> AotResult<Vec<AotBuiltinBinding>> {
    closure_bindings(reachability).map_err(DynamicCompositionReason::strict_error)
}
fn closure_bindings(
    reachability: &ReachabilityReport,
) -> Result<Vec<AotBuiltinBinding>, DynamicCompositionReason> {
    if reachability.has_unknown_edges {
        return Err(DynamicCompositionReason::UnknownTargets);
    }
    let mut bindings = Vec::new();
    for builtin in reachability
        .nodes
        .iter()
        .filter_map(|node| node.builtin_id())
    {
        let entry =
            runmat_builtins::builtin_catalog_entry_by_name(&builtin.0).ok_or_else(|| {
                DynamicCompositionReason::MissingCatalog {
                    builtin: builtin.clone(),
                }
            })?;
        if !matches!(
            entry.link.reachability,
            runmat_builtins::BuiltinReachability::Always
        ) {
            return Err(DynamicCompositionReason::DynamicBuiltin {
                builtin: builtin.clone(),
            });
        }
        if entry.bindings.is_empty() {
            return Err(DynamicCompositionReason::MissingBindings {
                builtin: builtin.clone(),
            });
        }
        if entry.bindings.iter().any(|binding| {
            binding.availability == runmat_builtins::BuiltinBindingAvailability::TargetConditional
        }) {
            return Err(DynamicCompositionReason::TargetConditional {
                builtin: builtin.clone(),
            });
        }
        bindings.extend(entry.bindings.iter().map(|binding| {
            let identity = entry.binding_identity(binding);
            AotBuiltinBinding {
                name: identity.builtin.name.to_string(),
                variant: identity.variant.to_string(),
                native_symbol: runmat_builtins::native_binding_symbol(
                    identity.builtin.name,
                    identity.variant,
                ),
            }
        }));
    }
    bindings.sort_by(|left, right| (&left.name, &left.variant).cmp(&(&right.name, &right.variant)));
    bindings.dedup_by(|left, right| left.name == right.name && left.variant == right.variant);
    Ok(bindings)
}
