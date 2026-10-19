//! One MIR-owned projection of shared lexical binding demand for both
//! executors. Display names and caller-local frame snapshots are not owners.

use std::collections::{BTreeMap, BTreeSet};

use runmat_types::{BindingId, FunctionSourceRole};
use runmat_types::{FunctionId, WorkspaceEffect};

use crate::{MirAssembly, MirStmtKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MirLexicalEnvironmentProjection {
    /// Bindings owned by this function and captured by a nested descendant.
    pub owned: BTreeSet<BindingId>,
    /// Bindings inherited from an ancestor, retained through this frame.
    pub inherited: BTreeSet<BindingId>,
    /// Ancestor cells carried through an anonymous frame to its nested
    /// descendants. The anonymous frame's own capture remains by value.
    pub transit: BTreeSet<BindingId>,
    /// Captures of canonical session storage delegate to that storage, not a
    /// shared lexical cell or a captured value snapshot.
    pub inherited_storage: Vec<MirInheritedStorageBinding>,
    /// A fresh cell is needed only for bindings owned by this activation;
    /// inherited and transit bindings reuse their existing ancestor cell.
    pub creates_activation: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirInheritedStorageKind {
    Global,
    Persistent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MirInheritedStorageBinding {
    pub binding: BindingId,
    pub owner: FunctionId,
    pub kind: MirInheritedStorageKind,
}

impl MirAssembly {
    pub fn lexical_environment_projection(
        &self,
        function: FunctionId,
    ) -> Result<MirLexicalEnvironmentProjection, String> {
        let metadata = self
            .functions
            .get(&function)
            .ok_or_else(|| "lexical environment function metadata is absent".to_string())?;
        let body = self
            .bodies
            .get(&function)
            .ok_or_else(|| "lexical environment function body is absent".to_string())?;
        // Anonymous captures are snapshots. Only captures demanded by a
        // lexically nested callable participate in shared mutable storage.
        // The fixed-point capture pass has already carried these demands
        // through intervening frames, including anonymous frames that create
        // or call a nested handle.
        // Global and persistent declarations use Runtime workspace authority.
        // A nested capture of the same HIR binding never creates a competing
        // per-activation lexical cell for that storage.
        let mut nonlexical = BTreeMap::new();
        for (owner, declared_body) in &self.bodies {
            for statement in declared_body
                .blocks
                .iter()
                .flat_map(|block| &block.statements)
            {
                let MirStmtKind::WorkspaceEffect { effect, bindings } = &statement.kind else {
                    continue;
                };
                let kind = match effect {
                    WorkspaceEffect::MutatesGlobal => MirInheritedStorageKind::Global,
                    WorkspaceEffect::MutatesPersistent => MirInheritedStorageKind::Persistent,
                    _ => continue,
                };
                for local_id in bindings {
                    let local = declared_body
                        .locals
                        .iter()
                        .find(|local| local.id == *local_id)
                        .ok_or_else(|| "declared workspace binding has no MIR local".to_string())?;
                    let binding = local
                        .binding
                        .ok_or_else(|| "declared workspace local has no HIR binding".to_string())?;
                    if let Some(previous) = nonlexical.insert(binding, (*owner, kind)) {
                        if previous != (*owner, kind) {
                            return Err(
                                "canonical workspace binding has conflicting declarations".into()
                            );
                        }
                    }
                }
            }
        }
        let shared = self
            .functions
            .values()
            .filter(|candidate| candidate.source_role == FunctionSourceRole::Nested)
            .flat_map(|candidate| candidate.captures.iter())
            .map(|capture| capture.binding)
            .filter(|binding| !nonlexical.contains_key(binding))
            .collect::<BTreeSet<_>>();
        let inherited_storage = if metadata.source_role == FunctionSourceRole::Nested {
            metadata
                .captures
                .iter()
                .filter_map(|capture| {
                    nonlexical.get(&capture.binding).map(|(owner, kind)| {
                        MirInheritedStorageBinding {
                            binding: capture.binding,
                            owner: *owner,
                            kind: *kind,
                        }
                    })
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        if inherited_storage.iter().any(|entry| {
            !metadata.captures.iter().any(|capture| {
                capture.binding == entry.binding && capture.from_function == entry.owner
            })
        }) {
            return Err("inherited canonical storage has a mismatched owner".into());
        }
        let inherited_demand = metadata
            .captures
            .iter()
            .filter(|capture| {
                capture.from_function != function && shared.contains(&capture.binding)
            })
            .map(|capture| capture.binding)
            .collect::<BTreeSet<_>>();
        let (inherited, transit) = if metadata.source_role == FunctionSourceRole::Anonymous {
            (BTreeSet::new(), inherited_demand)
        } else {
            (inherited_demand, BTreeSet::new())
        };
        let owned = self
            .functions
            .values()
            .flat_map(|candidate| candidate.captures.iter())
            .filter(|capture| {
                capture.from_function == function && shared.contains(&capture.binding)
            })
            .map(|capture| capture.binding)
            .collect::<BTreeSet<_>>();
        if owned.iter().any(|binding| {
            !body
                .locals
                .iter()
                .any(|local| local.binding == Some(*binding))
        }) {
            return Err("shared lexical binding is absent from its owner frame".into());
        }
        if inherited.iter().chain(transit.iter()).any(|binding| {
            !body
                .locals
                .iter()
                .any(|local| local.binding == Some(*binding))
        }) {
            return Err("inherited lexical binding is absent from its frame".into());
        }
        let creates_activation = !owned.is_empty();
        Ok(MirLexicalEnvironmentProjection {
            owned,
            inherited,
            transit,
            inherited_storage,
            creates_activation,
        })
    }
}

#[cfg(all(test, feature = "compiler"))]
mod tests {
    use super::*;
    use crate::{MirBody, MirFunctionMetadata, MirLocal, MirLocalId, MirLocalKind};
    use runmat_hir::{
        CapturedBinding, FunctionAbi, FunctionKind, FunctionModifiers, FunctionName, Span,
    };
    use runmat_types::ProgramSourceId;

    fn function(
        assembly: &mut MirAssembly,
        id: usize,
        parent: Option<usize>,
        role: FunctionSourceRole,
        captures: Vec<(usize, usize)>,
        locals: Vec<usize>,
    ) {
        let id = FunctionId(id);
        assembly.functions.insert(
            id,
            MirFunctionMetadata {
                source: ProgramSourceId(0),
                name: FunctionName(format!("f{}", id.0)),
                parent: parent.map(FunctionId),
                enclosing_class: None,
                class_method_owner: None,
                kind: if role == FunctionSourceRole::Anonymous {
                    FunctionKind::Anonymous
                } else {
                    FunctionKind::Named
                },
                source_role: role,
                argument_validations: Vec::new(),
                captures: captures
                    .into_iter()
                    .map(|(binding, owner)| CapturedBinding {
                        binding: BindingId(binding),
                        from_function: FunctionId(owner),
                    })
                    .collect(),
                modifiers: FunctionModifiers::default(),
                span: Span::default(),
            },
        );
        assembly.bodies.insert(
            id,
            MirBody {
                function: id,
                abi: FunctionAbi {
                    fixed_inputs: Vec::new(),
                    varargin: None,
                    fixed_outputs: Vec::new(),
                    varargout: None,
                    implicit_nargin: None,
                    implicit_nargout: None,
                },
                locals: locals
                    .into_iter()
                    .enumerate()
                    .map(|(slot, binding)| MirLocal {
                        id: MirLocalId(slot),
                        binding: Some(BindingId(binding)),
                        kind: MirLocalKind::Binding,
                        span: Span::default(),
                    })
                    .collect(),
                blocks: Vec::new(),
                catch_scopes: Default::default(),
            },
        );
    }

    #[test]
    fn anonymous_snapshots_alone_do_not_allocate_a_shared_environment() {
        let mut assembly = MirAssembly::default();
        function(
            &mut assembly,
            0,
            None,
            FunctionSourceRole::Primary,
            vec![],
            vec![10],
        );
        function(
            &mut assembly,
            1,
            Some(0),
            FunctionSourceRole::Anonymous,
            vec![(10, 0)],
            vec![10],
        );
        let owner = assembly
            .lexical_environment_projection(FunctionId(0))
            .unwrap();
        let anonymous = assembly
            .lexical_environment_projection(FunctionId(1))
            .unwrap();
        assert!(owner.owned.is_empty());
        assert!(!owner.creates_activation);
        assert!(anonymous.inherited.is_empty());
        assert!(anonymous.transit.is_empty());
        assert!(!anonymous.creates_activation);
    }

    #[test]
    fn nested_demand_projects_owner_and_intervening_frame() {
        let mut assembly = MirAssembly::default();
        function(
            &mut assembly,
            0,
            None,
            FunctionSourceRole::Primary,
            vec![],
            vec![10],
        );
        function(
            &mut assembly,
            1,
            Some(0),
            FunctionSourceRole::Anonymous,
            vec![(10, 0)],
            vec![10],
        );
        function(
            &mut assembly,
            2,
            Some(1),
            FunctionSourceRole::Nested,
            vec![(10, 0)],
            vec![10],
        );
        let owner = assembly
            .lexical_environment_projection(FunctionId(0))
            .unwrap();
        let intermediary = assembly
            .lexical_environment_projection(FunctionId(1))
            .unwrap();
        assert_eq!(owner.owned, BTreeSet::from([BindingId(10)]));
        assert!(owner.creates_activation);
        assert!(intermediary.inherited.is_empty());
        assert_eq!(intermediary.transit, BTreeSet::from([BindingId(10)]));
        assert!(!intermediary.creates_activation);
    }

    #[test]
    fn declared_global_capture_keeps_workspace_storage_authoritative() {
        let mut assembly = MirAssembly::default();
        function(
            &mut assembly,
            0,
            None,
            FunctionSourceRole::Primary,
            vec![],
            vec![10],
        );
        function(
            &mut assembly,
            1,
            Some(0),
            FunctionSourceRole::Nested,
            vec![(10, 0)],
            vec![10],
        );
        assembly
            .bodies
            .get_mut(&FunctionId(0))
            .unwrap()
            .blocks
            .push(crate::BasicBlock {
                id: crate::BasicBlockId(0),
                statements: vec![crate::MirStmt {
                    kind: MirStmtKind::WorkspaceEffect {
                        effect: WorkspaceEffect::MutatesGlobal,
                        bindings: vec![MirLocalId(0)],
                    },
                    span: Span::default(),
                }],
                terminator: crate::MirTerminator {
                    kind: crate::MirTerminatorKind::Return(Vec::new()),
                    span: Span::default(),
                },
            });
        let owner = assembly
            .lexical_environment_projection(FunctionId(0))
            .unwrap();
        let nested = assembly
            .lexical_environment_projection(FunctionId(1))
            .unwrap();
        assert!(owner.owned.is_empty());
        assert!(!owner.creates_activation);
        assert!(nested.inherited.is_empty());
        assert_eq!(
            nested.inherited_storage,
            vec![MirInheritedStorageBinding {
                binding: BindingId(10),
                owner: FunctionId(0),
                kind: MirInheritedStorageKind::Global,
            }]
        );
    }
}
