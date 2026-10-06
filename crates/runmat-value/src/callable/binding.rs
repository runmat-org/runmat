use runmat_execution::FunctionBindingOrigin;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

/// The executable registry owns the definition; values hold only a counted
/// lease on its opaque identity. No value points back to the registry.
#[derive(Debug, Default)]
pub struct FunctionDefinitionOwner {
    handles: AtomicUsize,
}

impl FunctionDefinitionOwner {
    pub fn issue(owner: &Arc<Self>, function: usize) -> FunctionDefinitionLease {
        owner.handles.fetch_add(1, Ordering::Relaxed);
        FunctionDefinitionLease {
            owner: Arc::clone(owner),
            function,
        }
    }

    pub fn has_handles(&self) -> bool {
        self.handles.load(Ordering::Acquire) != 0
    }
}

#[derive(Debug)]
pub struct FunctionDefinitionLease {
    owner: Arc<FunctionDefinitionOwner>,
    function: usize,
}

impl FunctionDefinitionLease {
    pub fn matches(&self, owner: &Arc<FunctionDefinitionOwner>) -> bool {
        Arc::ptr_eq(&self.owner, owner)
    }

    pub fn function(&self) -> usize {
        self.function
    }
}

impl Clone for FunctionDefinitionLease {
    fn clone(&self) -> Self {
        FunctionDefinitionOwner::issue(&self.owner, self.function)
    }
}

impl Drop for FunctionDefinitionLease {
    fn drop(&mut self) {
        self.owner.handles.fetch_sub(1, Ordering::Release);
    }
}

impl PartialEq for FunctionDefinitionLease {
    fn eq(&self, other: &Self) -> bool {
        self.function == other.function && Arc::ptr_eq(&self.owner, &other.owner)
    }
}

impl Eq for FunctionDefinitionLease {}

/// A bound callable has exactly one authority: a live local issuer or a
/// relocatable immutable program/function origin. Its numeric function ID is
/// only a cache within the currently selected runtime registry.
#[derive(Clone, Debug)]
pub enum BoundFunctionAuthority {
    Local {
        issuer: Weak<()>,
        definition: FunctionDefinitionLease,
        dependencies: Arc<[FunctionDefinitionLease]>,
        indices: Arc<HashMap<usize, Option<usize>>>,
        export: LocalExport,
    },
    Portable {
        origin: FunctionBindingOrigin,
    },
}

/// An export identity is inert for local execution. It can only be converted
/// to portable authority by the value codec after admitting the local owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalExport {
    Ephemeral,
    Exact(FunctionBindingOrigin),
}

/// The local number is only a cache within the selected function registry.
/// Execution authority is either a live local issuer or an immutable origin.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundFunctionBinding {
    pub function: usize,
    pub authority: BoundFunctionAuthority,
}

impl BoundFunctionBinding {
    /// An inspectable legacy/unregistered binding, never execution authority.
    pub fn unavailable(function: usize) -> Self {
        Self {
            function,
            authority: BoundFunctionAuthority::unavailable(function),
        }
    }

    pub fn local(
        function: usize,
        issuer: Weak<()>,
        definition: FunctionDefinitionLease,
        dependencies: Vec<FunctionDefinitionLease>,
        export: LocalExport,
    ) -> Self {
        Self {
            function,
            authority: BoundFunctionAuthority::local(issuer, definition, dependencies, export),
        }
    }

    pub fn portable(function: usize, origin: FunctionBindingOrigin) -> Self {
        Self {
            function,
            authority: BoundFunctionAuthority::portable(origin),
        }
    }
}

impl BoundFunctionAuthority {
    /// Represents an unregistered bound ID without granting execution or
    /// transport authority. This preserves value inspection without turning a
    /// numeric ID into an executable capability.
    pub fn unavailable(function: usize) -> Self {
        static INERT: std::sync::OnceLock<Arc<FunctionDefinitionOwner>> =
            std::sync::OnceLock::new();
        let owner = INERT.get_or_init(|| Arc::new(FunctionDefinitionOwner::default()));
        Self::local(
            Weak::new(),
            FunctionDefinitionOwner::issue(owner, function),
            Vec::new(),
            LocalExport::Ephemeral,
        )
    }

    pub fn local(
        issuer: Weak<()>,
        definition: FunctionDefinitionLease,
        dependencies: Vec<FunctionDefinitionLease>,
        export: LocalExport,
    ) -> Self {
        let mut indices = HashMap::with_capacity(dependencies.len() + 1);
        for (index, lease) in std::iter::once(&definition)
            .chain(dependencies.iter())
            .enumerate()
        {
            indices
                .entry(lease.function())
                .and_modify(|known| *known = None)
                .or_insert(Some(index));
        }
        Self::Local {
            issuer,
            definition,
            dependencies: dependencies.into(),
            indices: Arc::new(indices),
            export,
        }
    }

    pub fn portable(origin: FunctionBindingOrigin) -> Self {
        Self::Portable { origin }
    }

    pub fn admits_local(&self, current: &Weak<()>) -> bool {
        match self {
            Self::Local { issuer, .. } => {
                issuer.strong_count() != 0 && Weak::ptr_eq(issuer, current)
            }
            Self::Portable { .. } => false,
        }
    }

    pub fn portable_origin(&self) -> Option<FunctionBindingOrigin> {
        match self {
            Self::Local { .. } => None,
            Self::Portable { origin } => Some(*origin),
        }
    }

    pub fn local_definition(&self) -> Option<&FunctionDefinitionLease> {
        match self {
            Self::Local { definition, .. } => Some(definition),
            Self::Portable { .. } => None,
        }
    }

    /// Exact graph node, selected by the bound function ID. This index is
    /// constructed once when the value is issued, not on every static call.
    pub fn local_lease(&self, function: usize) -> Option<&FunctionDefinitionLease> {
        let Self::Local {
            definition,
            dependencies,
            indices,
            ..
        } = self
        else {
            return None;
        };
        match (*indices.get(&function)?)? {
            0 => Some(definition),
            index => dependencies.get(index - 1),
        }
    }

    /// Two derived static bindings share one immutable issued dependency
    /// graph. This is not an authority check by itself; the selected lease is
    /// still validated against the executable registry at the call boundary.
    pub fn shares_local_graph(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (
                Self::Local {
                    issuer: left_issuer,
                    definition: left_definition,
                    dependencies: left_dependencies,
                    indices: left_indices,
                    export: left_export,
                },
                Self::Local {
                    issuer: right_issuer,
                    definition: right_definition,
                    dependencies: right_dependencies,
                    indices: right_indices,
                    export: right_export,
                }
            ) if Weak::ptr_eq(left_issuer, right_issuer)
                && left_definition == right_definition
                && Arc::ptr_eq(left_dependencies, right_dependencies)
                && Arc::ptr_eq(left_indices, right_indices)
                && left_export == right_export
        )
    }

    pub fn export_origin(&self, current: &Weak<()>) -> Option<FunctionBindingOrigin> {
        match self {
            Self::Local {
                export: LocalExport::Exact(origin),
                ..
            } if self.admits_local(current) => Some(*origin),
            Self::Portable { origin } => Some(*origin),
            _ => None,
        }
    }
}

impl PartialEq for BoundFunctionAuthority {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Local {
                    issuer: left,
                    definition: left_definition,
                    dependencies: left_dependencies,
                    export: left_export,
                    ..
                },
                Self::Local {
                    issuer: right,
                    definition: right_definition,
                    dependencies: right_dependencies,
                    export: right_export,
                    ..
                },
            ) => {
                Weak::ptr_eq(left, right)
                    && left_definition == right_definition
                    && left_dependencies == right_dependencies
                    && left_export == right_export
            }
            (Self::Portable { origin: left }, Self::Portable { origin: right }) => left == right,
            _ => false,
        }
    }
}

impl Eq for BoundFunctionAuthority {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_authority_expires_and_never_becomes_portable() {
        let owner = Arc::new(());
        let definition = Arc::new(FunctionDefinitionOwner::default());
        let local = BoundFunctionAuthority::local(
            Arc::downgrade(&owner),
            FunctionDefinitionOwner::issue(&definition, 1),
            Vec::new(),
            LocalExport::Ephemeral,
        );
        let other = Arc::new(());
        assert!(local.admits_local(&Arc::downgrade(&owner)));
        assert!(!local.admits_local(&Arc::downgrade(&other)));
        assert_eq!(local.portable_origin(), None);
        drop(owner);
        assert!(!local.admits_local(&Weak::new()));
        assert!(definition.has_handles());
        drop(local);
        assert!(!definition.has_handles());
    }

    #[test]
    fn local_dependency_graph_index_is_shared_by_derived_static_bindings() {
        let issuer = Arc::new(());
        let root = Arc::new(FunctionDefinitionOwner::default());
        let helper = Arc::new(FunctionDefinitionOwner::default());
        let binding = BoundFunctionBinding::local(
            11,
            Arc::downgrade(&issuer),
            FunctionDefinitionOwner::issue(&root, 11),
            vec![FunctionDefinitionOwner::issue(&helper, 12)],
            LocalExport::Ephemeral,
        );
        let derived = BoundFunctionBinding {
            function: 12,
            authority: binding.authority.clone(),
        };
        assert!(derived
            .authority
            .local_lease(12)
            .is_some_and(|lease| lease.matches(&helper)));
        assert!(derived.authority.local_lease(13).is_none());
        let (
            BoundFunctionAuthority::Local { indices: left, .. },
            BoundFunctionAuthority::Local { indices: right, .. },
        ) = (&binding.authority, &derived.authority)
        else {
            panic!("local graph");
        };
        assert!(Arc::ptr_eq(left, right));
    }

    #[test]
    fn conflicting_duplicate_local_lease_id_never_selects_an_arbitrary_body() {
        let issuer = Arc::new(());
        let first = Arc::new(FunctionDefinitionOwner::default());
        let second = Arc::new(FunctionDefinitionOwner::default());
        let binding = BoundFunctionBinding::local(
            11,
            Arc::downgrade(&issuer),
            FunctionDefinitionOwner::issue(&first, 11),
            vec![FunctionDefinitionOwner::issue(&second, 11)],
            LocalExport::Ephemeral,
        );
        assert!(binding.authority.local_lease(11).is_none());
    }
}
