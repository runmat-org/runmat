use runmat_execution::value::GraphStaticReferenceSite;
use runmat_execution::Digest;
use runmat_execution::FunctionBindingOrigin;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use crate::SelectedReceiverDeclarations;

/// The executable registry owns the definition; values hold only a counted
/// lease on its opaque identity. No value points back to the registry.
#[derive(Debug, Default)]
pub struct FunctionDefinitionOwner {
    handles: AtomicUsize,
    selected_receiver_seal: OnceLock<Arc<SelectedReceiverDeclarationSeal>>,
}

/// A constructor-owned slot for pure receiver metadata. The slot is installed
/// before a selected constructor becomes callable; only its selected class
/// product keeps the corresponding fill key. This permits cold ancestry to be
/// completed at invocation without letting a public value factory substitute
/// metadata for an already-issued constructor.
#[derive(Debug)]
pub struct SelectedReceiverDeclarationSeal {
    key: Arc<()>,
    declarations: OnceLock<Arc<SelectedReceiverDeclarations>>,
}

#[derive(Debug)]
pub struct SelectedReceiverDeclarationKey(Arc<()>);

impl SelectedReceiverDeclarationSeal {
    pub fn issue() -> (Arc<Self>, SelectedReceiverDeclarationKey) {
        let key = Arc::new(());
        (
            Arc::new(Self {
                key: Arc::clone(&key),
                declarations: OnceLock::new(),
            }),
            SelectedReceiverDeclarationKey(key),
        )
    }

    pub fn fill(
        &self,
        key: &SelectedReceiverDeclarationKey,
        declarations: Arc<SelectedReceiverDeclarations>,
    ) -> bool {
        if !Arc::ptr_eq(&self.key, &key.0) {
            return false;
        }
        if let Some(admitted) = self.declarations.get() {
            return Arc::ptr_eq(admitted, &declarations);
        }
        self.declarations.set(Arc::clone(&declarations)).is_ok()
            || self
                .declarations
                .get()
                .is_some_and(|admitted| Arc::ptr_eq(admitted, &declarations))
    }

    pub fn declarations(&self) -> Option<&Arc<SelectedReceiverDeclarations>> {
        self.declarations.get()
    }
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

    /// The exact constructor product installs its unfilled seal before a
    /// handle can escape. Reattachment of another selected product, even with
    /// equal class text and code, cannot replace its owner's seal.
    pub fn admit_selected_receiver_seal(&self, seal: Arc<SelectedReceiverDeclarationSeal>) -> bool {
        if let Some(admitted) = self.selected_receiver_seal.get() {
            return Arc::ptr_eq(admitted, &seal);
        }
        self.selected_receiver_seal.set(Arc::clone(&seal)).is_ok()
            || self
                .selected_receiver_seal
                .get()
                .is_some_and(|admitted| Arc::ptr_eq(admitted, &seal))
    }

    pub fn selected_receiver_declarations(&self) -> Option<&Arc<SelectedReceiverDeclarations>> {
        self.selected_receiver_seal.get()?.declarations()
    }

    pub fn owns_selected_receiver_seal(&self, seal: &Arc<SelectedReceiverDeclarationSeal>) -> bool {
        self.selected_receiver_seal
            .get()
            .is_some_and(|admitted| Arc::ptr_eq(admitted, seal))
    }
}

#[derive(Debug)]
pub struct FunctionDefinitionLease {
    owner: Arc<FunctionDefinitionOwner>,
    function: usize,
}

impl FunctionDefinitionLease {
    pub fn owner_arc(&self) -> &Arc<FunctionDefinitionOwner> {
        &self.owner
    }

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

/// A counted local target selected by a sealed call site. The site includes
/// its unit digest, so two revisions may use the same numeric FunctionId
/// without entering the flat numeric definition overlay.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticSiteDefinitionLease {
    pub caller_unit: Digest,
    pub caller: Option<FunctionBindingOrigin>,
    pub site: GraphStaticReferenceSite,
    pub target_unit: Digest,
    pub target: FunctionBindingOrigin,
    pub definition: FunctionDefinitionLease,
    pub dependencies: Arc<[FunctionDefinitionLease]>,
}

/// A fresh counted target binding issued from one exact sealed call site.
/// The caller/target unit and origin fields remain explicit so consumers never
/// infer authority from a colliding numeric function ID or tuple position.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectedStaticSiteBinding {
    pub caller_unit: Digest,
    pub caller: Option<FunctionBindingOrigin>,
    pub site: GraphStaticReferenceSite,
    pub target_binding: BoundFunctionBinding,
    pub target_unit: Digest,
    pub target: FunctionBindingOrigin,
}

/// A bound callable has exactly one authority: a live local issuer or a
/// relocatable immutable program/function origin. Its numeric function ID is
/// only a cache within the currently selected runtime registry.
#[derive(Clone, Debug)]
pub enum BoundFunctionAuthority {
    Local {
        issuer: Weak<()>,
        definition: FunctionDefinitionLease,
        dependencies: Arc<[FunctionDefinitionLease]>,
        static_sites: Arc<[StaticSiteDefinitionLease]>,
        indices: Arc<HashMap<usize, Option<usize>>>,
        export: LocalExport,
    },
    Portable {
        origin: FunctionBindingOrigin,
    },
    /// An invocation-local selected native target. Its typed lease retains
    /// executable code and source authority; it is not a portable export.
    SelectedNative {
        origin: FunctionBindingOrigin,
        lease: std::rc::Rc<dyn SelectedNativeFunctionLease>,
    },
}

/// Neutral lifetime/identity contract for a selected native callable. The
/// Runtime owner supplies the actual invocation implementation; Value never
/// depends on a compiler, executor or runtime context.
pub trait SelectedNativeFunctionLease: std::fmt::Debug + std::any::Any {
    fn compiled_function(&self) -> runmat_types::FunctionId;
    fn origin(&self) -> FunctionBindingOrigin;
    fn is_live(&self) -> bool;
    fn as_any(&self) -> &dyn std::any::Any;
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

    pub fn selected_native(
        function: runmat_types::FunctionId,
        lease: std::rc::Rc<dyn SelectedNativeFunctionLease>,
    ) -> Option<Self> {
        (lease.is_live() && lease.compiled_function() == function).then(|| Self {
            function: function.0,
            authority: BoundFunctionAuthority::SelectedNative {
                origin: lease.origin(),
                lease,
            },
        })
    }
}

impl BoundFunctionAuthority {
    pub fn local_dependencies(&self) -> Option<&[FunctionDefinitionLease]> {
        match self {
            Self::Local { dependencies, .. } => Some(dependencies),
            Self::Portable { .. } | Self::SelectedNative { .. } => None,
        }
    }

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
        Self::local_with_static_sites(issuer, definition, dependencies, Vec::new(), export)
    }

    pub fn local_with_static_sites(
        issuer: Weak<()>,
        definition: FunctionDefinitionLease,
        mut dependencies: Vec<FunctionDefinitionLease>,
        static_sites: Vec<StaticSiteDefinitionLease>,
        export: LocalExport,
    ) -> Self {
        // A selected static graph is a set of exact function/owner leases.
        // Registry traversal order depends on the rooted entrypoint, but it
        // must not change authority equality. The owner pointer is used only
        // as a local tie-breaker for duplicate function IDs; the lookup index
        // below still marks those IDs ambiguous.
        dependencies.sort_unstable_by(|left, right| {
            left.function
                .cmp(&right.function)
                .then_with(|| Arc::as_ptr(&left.owner).cmp(&Arc::as_ptr(&right.owner)))
        });
        let mut indices = HashMap::<usize, Option<usize>>::with_capacity(dependencies.len() + 1);
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
            static_sites: static_sites.into(),
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
            Self::Portable { .. } | Self::SelectedNative { .. } => false,
        }
    }

    pub fn portable_origin(&self) -> Option<FunctionBindingOrigin> {
        match self {
            Self::Local { .. } => None,
            Self::Portable { origin } => Some(*origin),
            Self::SelectedNative { .. } => None,
        }
    }

    pub fn local_definition(&self) -> Option<&FunctionDefinitionLease> {
        match self {
            Self::Local { definition, .. } => Some(definition),
            Self::Portable { .. } | Self::SelectedNative { .. } => None,
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

    pub fn static_sites(&self) -> &[StaticSiteDefinitionLease] {
        match self {
            Self::Local { static_sites, .. } => static_sites,
            Self::Portable { .. } | Self::SelectedNative { .. } => &[],
        }
    }

    /// Derive every exact target once at invocation admission. The shared site
    /// and dependency lease arrays survive an escaped child binding, while
    /// dispatch uses a prebuilt site map rather than rescanning per call.
    pub fn selected_site_bindings(&self) -> Result<Vec<SelectedStaticSiteBinding>, &'static str> {
        let Self::Local {
            issuer,
            static_sites,
            ..
        } = self
        else {
            return Ok(Vec::new());
        };
        if issuer.strong_count() == 0 {
            return Err("selected static site issuer has retired");
        }
        Ok(static_sites
            .iter()
            .map(|selected| {
                let function = selected.definition.function();
                let mut indices =
                    HashMap::<usize, Option<usize>>::with_capacity(selected.dependencies.len() + 1);
                for (index, lease) in std::iter::once(&selected.definition)
                    .chain(selected.dependencies.iter())
                    .enumerate()
                {
                    indices
                        .entry(lease.function())
                        .and_modify(|known| *known = None)
                        .or_insert(Some(index));
                }
                let authority = Self::Local {
                    issuer: issuer.clone(),
                    definition: selected.definition.clone(),
                    dependencies: Arc::clone(&selected.dependencies),
                    static_sites: Arc::clone(static_sites),
                    indices: Arc::new(indices),
                    export: LocalExport::Exact(selected.target),
                };
                SelectedStaticSiteBinding {
                    caller_unit: selected.caller_unit,
                    caller: selected.caller,
                    site: selected.site,
                    target_binding: BoundFunctionBinding {
                        function,
                        authority,
                    },
                    target_unit: selected.target_unit,
                    target: selected.target,
                }
            })
            .collect())
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
                    static_sites: left_static_sites,
                    indices: left_indices,
                    export: left_export,
                },
                Self::Local {
                    issuer: right_issuer,
                    definition: right_definition,
                    dependencies: right_dependencies,
                    static_sites: right_static_sites,
                    indices: right_indices,
                    export: right_export,
                }
            ) if Weak::ptr_eq(left_issuer, right_issuer)
                && left_definition == right_definition
                && Arc::ptr_eq(left_dependencies, right_dependencies)
                && Arc::ptr_eq(left_static_sites, right_static_sites)
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
            Self::SelectedNative { .. } => None,
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
                    static_sites: left_static_sites,
                    export: left_export,
                    ..
                },
                Self::Local {
                    issuer: right,
                    definition: right_definition,
                    dependencies: right_dependencies,
                    static_sites: right_static_sites,
                    export: right_export,
                    ..
                },
            ) => {
                Weak::ptr_eq(left, right)
                    && left_definition == right_definition
                    && left_dependencies == right_dependencies
                    && left_static_sites == right_static_sites
                    && left_export == right_export
            }
            (Self::Portable { origin: left }, Self::Portable { origin: right }) => left == right,
            (
                Self::SelectedNative {
                    origin: left,
                    lease: left_lease,
                },
                Self::SelectedNative {
                    origin: right,
                    lease: right_lease,
                },
            ) => left == right && std::rc::Rc::ptr_eq(left_lease, right_lease),
            _ => false,
        }
    }
}

impl Eq for BoundFunctionAuthority {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    #[derive(Debug)]
    struct TestSelectedNativeLease {
        compiled_function: runmat_types::FunctionId,
        origin: FunctionBindingOrigin,
        live: Rc<Cell<bool>>,
    }

    impl SelectedNativeFunctionLease for TestSelectedNativeLease {
        fn compiled_function(&self) -> runmat_types::FunctionId {
            self.compiled_function
        }
        fn origin(&self) -> FunctionBindingOrigin {
            self.origin
        }

        fn is_live(&self) -> bool {
            self.live.get()
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn selected_native_binding_retains_exact_nonportable_invocation_lease() {
        let origin = FunctionBindingOrigin {
            program: runmat_execution::Digest::sha256(b"selected native test"),
            function: runmat_types::ProgramFunctionId(7),
        };
        let live = Rc::new(Cell::new(true));
        let lease: Rc<dyn SelectedNativeFunctionLease> = Rc::new(TestSelectedNativeLease {
            compiled_function: runmat_types::FunctionId(77),
            origin,
            live: Rc::clone(&live),
        });
        assert!(BoundFunctionBinding::selected_native(
            runmat_types::FunctionId(7),
            Rc::clone(&lease)
        )
        .is_none());
        let binding =
            BoundFunctionBinding::selected_native(runmat_types::FunctionId(77), Rc::clone(&lease))
                .expect("selected target is live and exact");
        assert!(matches!(
            &binding.authority,
            BoundFunctionAuthority::SelectedNative { .. }
        ));
        assert_eq!(binding.authority.portable_origin(), None);
        assert_eq!(binding.function, 77);
        assert!(
            matches!(&binding.authority, BoundFunctionAuthority::SelectedNative { origin: selected, .. } if *selected == origin)
        );
        assert_eq!(binding.authority.export_origin(&Weak::new()), None);
        live.set(false);
        assert!(
            BoundFunctionBinding::selected_native(runmat_types::FunctionId(77), lease).is_none()
        );
    }
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

    #[test]
    fn local_graph_order_is_canonical_but_owner_and_duplicate_ambiguity_remain_exact() {
        let issuer = Arc::new(());
        let definition = Arc::new(FunctionDefinitionOwner::default());
        let first = Arc::new(FunctionDefinitionOwner::default());
        let second = Arc::new(FunctionDefinitionOwner::default());
        let different = Arc::new(FunctionDefinitionOwner::default());
        let make = |dependencies: Vec<(usize, &Arc<FunctionDefinitionOwner>)>| {
            BoundFunctionAuthority::local(
                Arc::downgrade(&issuer),
                FunctionDefinitionOwner::issue(&definition, 1),
                dependencies
                    .into_iter()
                    .map(|(function, owner)| FunctionDefinitionOwner::issue(owner, function))
                    .collect(),
                LocalExport::Ephemeral,
            )
        };
        let selected = make(vec![(3, &second), (2, &first)]);
        assert_eq!(selected, make(vec![(2, &first), (3, &second)]));
        assert_ne!(selected, make(vec![(2, &first), (3, &different)]));

        let duplicate = make(vec![(2, &second), (2, &first)]);
        assert_eq!(duplicate, make(vec![(2, &first), (2, &second)]));
        assert!(duplicate.local_lease(2).is_none());
    }
}
