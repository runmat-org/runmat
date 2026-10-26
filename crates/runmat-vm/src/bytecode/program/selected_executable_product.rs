//! Exact code/source product carried by each selected definition. Its optional
//! local dynamic-source port owns only an invocation-captured capability and
//! a weak session evaluator; it has no strong edge back to a function body or
//! registry, so the last external definition lease can retire it normally.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use runmat_execution::executable::{
    ExecutableStaticImport, SelectedBodySemanticDigest, SelectedUnitContentDigest,
};
use runmat_execution::{Digest, FunctionBindingOrigin};
use runmat_runtime::source_context::SelectedProgramSourceOwner;
use runmat_value::{FunctionDefinitionOwner, StaticSiteDefinitionLease};

use runmat_hir::FunctionId;

use super::FunctionRegistry;

/// Why an exact selected product could not be attached to current definitions.
/// The fingerprint stage retains its original typed cause for source tracing.
#[derive(Debug, thiserror::Error)]
pub enum SelectedExecutableAttachmentError {
    #[error("selected executable body fingerprints are invalid: {0}")]
    Fingerprints(#[source] crate::RetainedCallableStageError),
    #[error("selected executable has no matching current definition {0:?}")]
    MissingDefinition(FunctionId),
    #[error("selected executable body {0:?} has no portable origin")]
    MissingOrigin(FunctionId),
    #[error("selected executable body {0:?} differs from definition/source authority")]
    AuthorityMismatch(FunctionId),
    #[error("selected executable body {0:?} would replace an exact prior product")]
    PriorProductConflict(FunctionId),
}

#[path = "selected_executable_product/code.rs"]
mod code;
#[path = "selected_executable_product/graph.rs"]
mod graph;
pub use code::SelectedExecutableCode;

#[derive(Debug)]
pub struct SelectedExecutableProduct {
    code: Rc<SelectedExecutableCode>,
    source_owner: SelectedProgramSourceOwner,
    foreign_basis: graph::SelectedForeignBasis,
    interop_receipt: Option<std::rc::Rc<crate::SelectedUnitInteropReceipt>>,
    /// Original invocation's finite dynamic-source authority. This local
    /// port is never serialized; a Graph worker needs a separately checked
    /// source handoff, not a reconstructed ambient session capability.
    dynamic_admission:
        Option<crate::program_execution::dynamic_admission::DynamicEvalAdmissionPort>,
}

/// One sealed class-declaration import retained without a registry/body
/// backedge or a counted definition lease. The target product owns its exact
/// V3 bytes and issued source facts; invocation admits those bytes and mints
/// a short-lived bound authority rather than selecting by a later name/ID.
#[derive(Clone, Debug)]
pub struct SelectedStaticProductTarget {
    import: ExecutableStaticImport,
    product: Rc<SelectedExecutableProduct>,
    dependencies: Rc<Vec<SelectedStaticProductDependency>>,
}

/// A captured edge between two immutable selected products. Edges are owned
/// by a separate capture vector, never by either endpoint product, so cycles
/// in the program graph cannot become Rc ownership cycles.
#[derive(Clone, Debug)]
pub struct SelectedStaticProductDependency {
    caller: Rc<SelectedExecutableProduct>,
    import: ExecutableStaticImport,
    target: Rc<SelectedExecutableProduct>,
}

/// Definition-owned immutable closure for one admitted unit. The definition
/// body retains this separate owner, while product nodes never point back to
/// a definition or to this graph; mutually recursive imports cannot pin a
/// registry or form an Rc cycle.
#[derive(Debug)]
pub struct SelectedStaticCapture {
    product: Rc<SelectedExecutableProduct>,
    dependencies: Rc<Vec<SelectedStaticProductDependency>>,
    site_owners: Rc<Vec<SelectedStaticSiteOwner>>,
    linked_native: Rc<Vec<SelectedLinkedNativeSite>>,
}

/// A source- and loaded-code-pinned native site in the separately tagged
/// linked-native section. It carries no fabricated VM definition lease.
#[derive(Clone, Debug)]
pub struct SelectedLinkedNativeSite {
    caller: Rc<SelectedExecutableProduct>,
    import: runmat_execution::executable::LinkedNativeStaticImport,
    target: runmat_runtime::user_functions::SelectedExternalSourceTarget,
}

impl SelectedLinkedNativeSite {
    pub fn new(
        caller: Rc<SelectedExecutableProduct>,
        import: runmat_execution::executable::LinkedNativeStaticImport,
        target: runmat_runtime::user_functions::SelectedExternalSourceTarget,
    ) -> Result<Self, &'static str> {
        if !target.same_issuing_session(caller.source_owner())
            || target.linked_import(import.caller, import.site) != Some(import)
            || !caller.linked_native_imports().contains(&import)
        {
            return Err("linked-native site differs from its sealed code/source target");
        }
        Ok(Self {
            caller,
            import,
            target,
        })
    }

    pub fn caller(&self) -> &Rc<SelectedExecutableProduct> {
        &self.caller
    }

    pub fn import(&self) -> runmat_execution::executable::LinkedNativeStaticImport {
        self.import
    }

    pub fn same_target(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.caller, &other.caller)
            && self.import == other.import
            && self.target.same_loaded_target(&other.target)
    }

    pub fn binding(&self) -> Option<runmat_value::BoundFunctionBinding> {
        self.target.binding()
    }
}

/// Unmetered body-owned target facts. Counted leases are issued only into an
/// escaped bound value, avoiding a persistent cycle between selected bodies.
#[derive(Clone, Debug)]
pub struct SelectedStaticSiteOwner {
    edge: SelectedStaticProductDependency,
    function: FunctionId,
    definition: Arc<FunctionDefinitionOwner>,
    dependencies: Arc<[(usize, Arc<FunctionDefinitionOwner>)]>,
}

impl SelectedStaticCapture {
    pub fn admit(
        product: Rc<SelectedExecutableProduct>,
        dependencies: Vec<SelectedStaticProductDependency>,
    ) -> Result<Rc<Self>, &'static str> {
        Self::admit_with_linked(product, dependencies, Vec::new())
    }

    pub fn admit_with_linked(
        product: Rc<SelectedExecutableProduct>,
        dependencies: Vec<SelectedStaticProductDependency>,
        linked_native: Vec<SelectedLinkedNativeSite>,
    ) -> Result<Rc<Self>, &'static str> {
        validate_static_product_dependencies(&product, &dependencies)?;
        let mut linked_by_caller = HashMap::<usize, HashSet<_>>::new();
        for site in &linked_native {
            if !site.target.same_issuing_session(product.source_owner())
                || site
                    .target
                    .linked_import(site.import.caller, site.import.site)
                    != Some(site.import)
                || !site.caller.linked_native_imports().contains(&site.import)
                || !linked_by_caller
                    .entry(Rc::as_ptr(&site.caller) as usize)
                    .or_default()
                    .insert((site.import.caller, site.import.site))
            {
                return Err("selected linked-native capture has a contradictory site");
            }
        }
        let mut pending = vec![Rc::clone(&product)];
        let mut visited = HashSet::new();
        while let Some(caller) = pending.pop() {
            let key = Rc::as_ptr(&caller) as usize;
            if !visited.insert(key) {
                continue;
            }
            let captured = linked_by_caller.get(&key);
            if caller.linked_native_imports().len() != captured.map_or(0, HashSet::len)
                || caller.linked_native_imports().iter().any(|import| {
                    captured.is_none_or(|sites| !sites.contains(&(import.caller, import.site)))
                })
            {
                return Err("selected linked-native capture omits a sealed site");
            }
            pending.extend(
                dependencies
                    .iter()
                    .filter(|edge| Rc::ptr_eq(edge.caller(), &caller))
                    .map(|edge| Rc::clone(edge.target())),
            );
        }
        if linked_by_caller.keys().any(|key| !visited.contains(key)) {
            return Err("selected linked-native capture omits or repeats a sealed site");
        }
        Ok(Rc::new(Self {
            product,
            dependencies: Rc::new(dependencies),
            site_owners: Rc::new(Vec::new()),
            linked_native: Rc::new(linked_native),
        }))
    }

    pub fn with_site_owners(
        self: Rc<Self>,
        site_owners: Vec<SelectedStaticSiteOwner>,
    ) -> Result<Rc<Self>, &'static str> {
        if site_owners.len() != self.dependencies.len() {
            return Err("selected static owner count differs from sealed edges");
        }
        for (edge, owner) in self.dependencies.iter().zip(&site_owners) {
            if edge.import != owner.edge.import
                || !Rc::ptr_eq(&edge.caller, &owner.edge.caller)
                || !Rc::ptr_eq(&edge.target, &owner.edge.target)
            {
                return Err("selected static owner differs from sealed edge");
            }
        }
        Ok(Rc::new(Self {
            product: Rc::clone(&self.product),
            dependencies: Rc::clone(&self.dependencies),
            site_owners: Rc::new(site_owners),
            linked_native: Rc::clone(&self.linked_native),
        }))
    }

    pub fn site_owners(&self) -> &[SelectedStaticSiteOwner] {
        &self.site_owners
    }

    pub fn issue_site_leases(&self) -> Option<Vec<StaticSiteDefinitionLease>> {
        (self.dependencies.len() == self.site_owners.len()).then(|| {
            self.site_owners
                .iter()
                .map(SelectedStaticSiteOwner::issue)
                .collect()
        })
    }

    pub fn product(&self) -> &Rc<SelectedExecutableProduct> {
        &self.product
    }

    pub fn dependencies(&self) -> &[SelectedStaticProductDependency] {
        &self.dependencies
    }

    pub fn linked_native(&self) -> &[SelectedLinkedNativeSite] {
        &self.linked_native
    }
}

impl SelectedStaticSiteOwner {
    pub fn new(
        edge: SelectedStaticProductDependency,
        function: FunctionId,
        definition: Arc<FunctionDefinitionOwner>,
        dependencies: Vec<(usize, Arc<FunctionDefinitionOwner>)>,
    ) -> Result<Self, &'static str> {
        if !edge.target.source_owner().is_live()
            || edge.target.body_digest(edge.import.target).is_none()
        {
            return Err("selected static owner has no admitted target body");
        }
        Ok(Self {
            edge,
            function,
            definition,
            dependencies: dependencies.into(),
        })
    }

    pub fn edge(&self) -> &SelectedStaticProductDependency {
        &self.edge
    }

    pub fn function(&self) -> FunctionId {
        self.function
    }

    pub fn definition(&self) -> &Arc<FunctionDefinitionOwner> {
        &self.definition
    }

    pub fn same_dependency_owners(&self, other: &Self) -> bool {
        self.dependencies.len() == other.dependencies.len()
            && self.dependencies.iter().zip(other.dependencies.iter()).all(
                |((left_id, left), (right_id, right))| {
                    left_id == right_id && Arc::ptr_eq(left, right)
                },
            )
    }

    fn issue(&self) -> StaticSiteDefinitionLease {
        StaticSiteDefinitionLease {
            caller_unit: self.edge.caller.full_digest(),
            caller: match self.edge.import.caller {
                runmat_execution::executable::ExecutableStaticCaller::Script => None,
                runmat_execution::executable::ExecutableStaticCaller::Function(origin) => {
                    Some(origin)
                }
            },
            site: self.edge.import.site,
            target_unit: self.edge.target.full_digest(),
            target: self.edge.import.target,
            definition: FunctionDefinitionOwner::issue(&self.definition, self.function.0),
            dependencies: self
                .dependencies
                .iter()
                .map(|(function, owner)| FunctionDefinitionOwner::issue(owner, *function))
                .collect::<Vec<_>>()
                .into(),
        }
    }
}

impl SelectedStaticProductDependency {
    pub fn new(
        caller: Rc<SelectedExecutableProduct>,
        import: ExecutableStaticImport,
        target: Rc<SelectedExecutableProduct>,
    ) -> Result<Self, &'static str> {
        SelectedStaticProductTarget::new(import, Rc::clone(&target))?;
        if !caller.source_owner().is_live()
            || !caller
                .source_owner()
                .session_lease()
                .ptr_eq(&target.source_owner().session_lease())
        {
            return Err("selected static edge differs from its caller or issuing session");
        }
        Ok(Self {
            caller,
            import,
            target,
        })
    }

    pub fn caller(&self) -> &Rc<SelectedExecutableProduct> {
        &self.caller
    }

    pub fn import(&self) -> ExecutableStaticImport {
        self.import
    }

    pub fn target(&self) -> &Rc<SelectedExecutableProduct> {
        &self.target
    }
}

impl SelectedStaticProductTarget {
    pub fn new(
        import: ExecutableStaticImport,
        product: Rc<SelectedExecutableProduct>,
    ) -> Result<Self, &'static str> {
        if !product.source_owner().is_live()
            || product.content_digest() != import.target_content
            || product.body_digest(import.target) != Some(import.target_body)
        {
            return Err("selected static target differs from its sealed code/source product");
        }
        Ok(Self {
            import,
            product,
            dependencies: Rc::new(Vec::new()),
        })
    }

    pub fn with_dependencies(
        mut self,
        dependencies: Vec<SelectedStaticProductDependency>,
    ) -> Result<Self, &'static str> {
        validate_static_product_dependencies(&self.product, &dependencies)?;
        self.dependencies = Rc::new(dependencies);
        Ok(self)
    }

    pub fn import(&self) -> ExecutableStaticImport {
        self.import
    }

    pub fn product(&self) -> &Rc<SelectedExecutableProduct> {
        &self.product
    }

    pub fn dependencies(&self) -> &[SelectedStaticProductDependency] {
        &self.dependencies
    }
}

fn validate_static_product_dependencies(
    root: &Rc<SelectedExecutableProduct>,
    dependencies: &[SelectedStaticProductDependency],
) -> Result<(), &'static str> {
    if dependencies.iter().any(|edge| {
        !root
            .source_owner()
            .session_lease()
            .ptr_eq(&edge.caller.source_owner().session_lease())
    }) {
        return Err("selected static capture crosses issuing sessions");
    }
    let mut outgoing = HashMap::<usize, Vec<&SelectedStaticProductDependency>>::new();
    let mut sites = HashMap::new();
    for edge in dependencies {
        let caller_key = Rc::as_ptr(&edge.caller) as usize;
        if sites
            .insert((caller_key, edge.import.caller, edge.import.site), edge)
            .is_some()
        {
            return Err("selected static capture repeats a sealed site");
        }
        outgoing.entry(caller_key).or_default().push(edge);
    }
    let mut pending = vec![Rc::clone(root)];
    let mut visited = HashSet::new();
    while let Some(product) = pending.pop() {
        let key = Rc::as_ptr(&product) as usize;
        if !visited.insert(key) {
            continue;
        }
        let edges = outgoing.get(&key).map(Vec::as_slice).unwrap_or(&[]);
        if edges.len() != product.static_imports().len()
            || product.static_imports().iter().any(|import| {
                sites
                    .get(&(key, import.caller, import.site))
                    .is_none_or(|edge: &&SelectedStaticProductDependency| edge.import != *import)
            })
        {
            return Err("selected static capture omits or repeats a sealed site");
        }
        pending.extend(edges.iter().map(|edge| Rc::clone(&edge.target)));
    }
    if outgoing.keys().any(|key| !visited.contains(key)) {
        return Err("selected static capture contains an unrelated product edge");
    }
    Ok(())
}

impl SelectedExecutableProduct {
    pub fn canonical_v3(&self) -> &Arc<[u8]> {
        &self.code.canonical_v3
    }

    pub fn full_digest(&self) -> Digest {
        self.code.full_digest
    }

    pub fn foreign_basis(&self) -> Option<Digest> {
        match self.foreign_basis {
            graph::SelectedForeignBasis::Local(basis) => basis,
            graph::SelectedForeignBasis::ArchiveAdmitted => None,
        }
    }

    pub fn interop_receipt(&self) -> Option<&std::rc::Rc<crate::SelectedUnitInteropReceipt>> {
        self.interop_receipt.as_ref()
    }

    pub fn source_owner(&self) -> &SelectedProgramSourceOwner {
        &self.source_owner
    }

    pub fn content_digest(&self) -> SelectedUnitContentDigest {
        self.code.content_digest
    }

    pub fn body_digest(&self, origin: FunctionBindingOrigin) -> Option<SelectedBodySemanticDigest> {
        self.code.body_semantic_digests.get(&origin).copied()
    }

    pub fn body_fingerprint(&self, origin: FunctionBindingOrigin) -> Option<&str> {
        self.code.body_fingerprints.get(&origin).map(String::as_str)
    }

    pub fn static_imports(&self) -> &[ExecutableStaticImport] {
        &self.code.static_imports
    }

    pub fn linked_native_imports(
        &self,
    ) -> &[runmat_execution::executable::LinkedNativeStaticImport] {
        &self.code.linked_native_imports
    }

    pub fn fingerprint_imports(&self) -> Vec<ExecutableStaticImport> {
        self.code
            .static_imports
            .iter()
            .copied()
            .chain(self.code.linked_native_imports.iter().copied().map(
                runmat_execution::executable::LinkedNativeStaticImport::fingerprint_projection,
            ))
            .collect()
    }

    pub fn admit(
        canonical_v3: Arc<[u8]>,
        source_owner: SelectedProgramSourceOwner,
    ) -> Result<Rc<Self>, String> {
        Self::admit_with_basis(canonical_v3, source_owner, None)
    }

    pub fn admit_with_basis(
        canonical_v3: Arc<[u8]>,
        source_owner: SelectedProgramSourceOwner,
        foreign_basis: Option<Digest>,
    ) -> Result<Rc<Self>, String> {
        Self::admit_with_basis_and_receipt(canonical_v3, source_owner, foreign_basis, None)
    }

    pub fn admit_with_basis_and_receipt(
        canonical_v3: Arc<[u8]>,
        source_owner: SelectedProgramSourceOwner,
        foreign_basis: Option<Digest>,
        interop_receipt: Option<std::rc::Rc<crate::SelectedUnitInteropReceipt>>,
    ) -> Result<Rc<Self>, String> {
        Self::admit_with_basis_receipt_and_dynamic_admission(
            canonical_v3,
            source_owner,
            foreign_basis,
            interop_receipt,
            None,
        )
    }

    pub fn admit_with_basis_receipt_and_dynamic_admission(
        canonical_v3: Arc<[u8]>,
        source_owner: SelectedProgramSourceOwner,
        foreign_basis: Option<Digest>,
        interop_receipt: Option<std::rc::Rc<crate::SelectedUnitInteropReceipt>>,
        dynamic_admission: Option<
            crate::program_execution::dynamic_admission::DynamicEvalAdmissionPort,
        >,
    ) -> Result<Rc<Self>, String> {
        if !source_owner.is_live() {
            return Err("selected executable source issuer has retired".into());
        }
        let code = SelectedExecutableCode::admit(canonical_v3)?;
        Self::bind_admitted_code(
            code,
            source_owner,
            foreign_basis,
            interop_receipt,
            dynamic_admission,
        )
    }

    /// Bind fresh invocation authority to already authenticated immutable code.
    /// Code reuse never reuses a source issuance, foreign receipt or dynamic port.
    pub fn bind_admitted_code(
        code: Rc<SelectedExecutableCode>,
        source_owner: SelectedProgramSourceOwner,
        foreign_basis: Option<Digest>,
        interop_receipt: Option<Rc<crate::SelectedUnitInteropReceipt>>,
        dynamic_admission: Option<
            crate::program_execution::dynamic_admission::DynamicEvalAdmissionPort,
        >,
    ) -> Result<Rc<Self>, String> {
        if !source_owner.is_live() {
            return Err("selected executable source issuer has retired".into());
        }
        if code.requires_interop() != foreign_basis.is_some()
            || code.requires_interop() != interop_receipt.is_some()
        {
            return Err(
                "selected executable foreign basis differs from its interop manifest".into(),
            );
        }
        if interop_receipt.as_ref().is_some_and(|receipt| {
            !receipt.matches_selected_unit(code.canonical_v3(), &source_owner)
        }) {
            return Err("selected executable foreign receipt differs from its issued V3".into());
        }
        if &code.source_map != source_owner.portable_map() {
            return Err("selected executable differs from its issued source map".into());
        }
        Ok(Rc::new(Self {
            code,
            source_owner,
            foreign_basis: graph::SelectedForeignBasis::Local(foreign_basis),
            interop_receipt,
            dynamic_admission,
        }))
    }

    pub fn code(&self) -> &Rc<SelectedExecutableCode> {
        &self.code
    }

    pub fn dynamic_admission(
        &self,
    ) -> Option<&crate::program_execution::dynamic_admission::DynamicEvalAdmissionPort> {
        self.dynamic_admission.as_ref()
    }
}

fn validate_product_source_and_prior(
    body: &super::FunctionBytecode,
    id: FunctionId,
    product: &Rc<SelectedExecutableProduct>,
) -> Result<(), SelectedExecutableAttachmentError> {
    if body
        .selected_source_owner
        .as_ref()
        .is_none_or(|owner| !owner.same_issuance(product.source_owner()))
        || !body
            .source_id
            .and_then(super::BytecodeSourceRef::runtime)
            .is_some_and(|source| product.source_owner().owns_source(source))
    {
        return Err(SelectedExecutableAttachmentError::AuthorityMismatch(id));
    }
    if body
        .selected_executable_product
        .as_ref()
        .is_some_and(|existing| {
            existing.full_digest() != product.full_digest()
                || !existing
                    .source_owner()
                    .same_issuance(product.source_owner())
                || existing.foreign_basis != product.foreign_basis
                || match (existing.interop_receipt(), product.interop_receipt()) {
                    (None, None) => false,
                    (Some(left), Some(right)) => !left.same_selected_admission(right),
                    _ => true,
                }
        })
    {
        return Err(SelectedExecutableAttachmentError::PriorProductConflict(id));
    }
    Ok(())
}

fn validate_selected_product_body(
    body: &super::FunctionBytecode,
    id: FunctionId,
    fingerprint: Option<&(FunctionBindingOrigin, String)>,
    product: &Rc<SelectedExecutableProduct>,
) -> Result<(), SelectedExecutableAttachmentError> {
    let origin = body
        .binding_origin
        .ok_or(SelectedExecutableAttachmentError::MissingOrigin(id))?;
    if origin.program != product.code.program
        || fingerprint.is_none_or(|(checked, value)| {
            *checked != origin || product.code.body_fingerprints.get(&origin) != Some(value)
        })
    {
        return Err(SelectedExecutableAttachmentError::AuthorityMismatch(id));
    }
    validate_product_source_and_prior(body, id, product)
}

impl FunctionRegistry {
    /// Publish an already checked selected product into the named session
    /// registry. The executing registry is the compiler-selected complete
    /// graph; the named registry deliberately omits private companions.
    /// Recheck the complete graph here, then prove every named body is the
    /// same definition owner and semantic body before transferring the
    /// product. No missing local can be supplied by a later numeric slot.
    pub fn attach_published_selected_executable_product(
        &mut self,
        ids: impl IntoIterator<Item = FunctionId>,
        executing: &FunctionRegistry,
        product: Rc<SelectedExecutableProduct>,
    ) -> Result<(), SelectedExecutableAttachmentError> {
        let ids = ids.into_iter().collect::<Vec<_>>();
        let mut selected_bodies = HashMap::with_capacity(product.code.body_fingerprints.len());
        let mut selected_origins = HashSet::with_capacity(product.code.body_fingerprints.len());
        for body in executing.selected_execution_bodies() {
            let Some(origin) = body.binding_origin else {
                continue;
            };
            if !product.code.body_fingerprints.contains_key(&origin) {
                continue;
            }
            if !selected_origins.insert(origin)
                || selected_bodies.insert(body.function, body).is_some()
            {
                return Err(SelectedExecutableAttachmentError::Fingerprints(
                    crate::RetainedCallableStageError::DuplicateOrigin,
                ));
            }
        }
        let mut selected_ids = selected_bodies.keys().copied().collect::<Vec<_>>();
        selected_ids.sort_by_key(|id| id.0);
        let selected_fingerprints = executing
            .selected_runtime_body_fingerprints_with_imports(
                &selected_ids,
                &product.fingerprint_imports(),
            )
            .map_err(SelectedExecutableAttachmentError::Fingerprints)?;
        if selected_origins.len() != product.code.body_fingerprints.len() {
            return Err(SelectedExecutableAttachmentError::Fingerprints(
                crate::RetainedCallableStageError::IncompleteDefinitionGraph,
            ));
        }
        for id in selected_ids {
            let selected = selected_bodies
                .get(&id)
                .copied()
                .ok_or(SelectedExecutableAttachmentError::MissingDefinition(id))?;
            validate_selected_product_body(selected, id, selected_fingerprints.get(&id), &product)?;
            if selected
                .selected_executable_product
                .as_ref()
                .is_none_or(|attached| !Rc::ptr_eq(attached, &product))
            {
                return Err(SelectedExecutableAttachmentError::AuthorityMismatch(id));
            }
        }
        for id in &ids {
            let selected = selected_bodies
                .get(id)
                .copied()
                .ok_or(SelectedExecutableAttachmentError::MissingDefinition(*id))?;
            let published = self
                .functions
                .get(id)
                .ok_or(SelectedExecutableAttachmentError::MissingDefinition(*id))?;
            let origin = selected
                .binding_origin
                .ok_or(SelectedExecutableAttachmentError::MissingOrigin(*id))?;
            let selected_body =
                super::semantic_fingerprint::semantic_definition_fingerprint(selected)
                    .map_err(|_| SelectedExecutableAttachmentError::AuthorityMismatch(*id))?;
            let published_body =
                super::semantic_fingerprint::semantic_definition_fingerprint_with_source(
                    published,
                    selected.source_id,
                )
                .map_err(|_| SelectedExecutableAttachmentError::AuthorityMismatch(*id))?;
            let source_alias_matches = selected
                .source_id
                .and_then(super::BytecodeSourceRef::runtime)
                .zip(
                    published
                        .source_id
                        .and_then(super::BytecodeSourceRef::runtime),
                )
                .and_then(|(issued, durable)| {
                    product
                        .source_owner()
                        .source_ordinal_for(issued)
                        .zip(product.source_owner().source_ordinal_for(durable))
                })
                .is_some_and(|(issued, durable)| issued == durable);
            if selected.function != published.function
                || !Arc::ptr_eq(&selected.definition_owner, &published.definition_owner)
                || published.binding_origin != Some(origin)
                || selected_body != published_body
                || !source_alias_matches
            {
                return Err(SelectedExecutableAttachmentError::AuthorityMismatch(*id));
            }
            validate_product_source_and_prior(published, *id, &product)?;
        }
        for id in ids {
            self.functions
                .get_mut(&id)
                .expect("published selected executable body was validated")
                .selected_executable_product = Some(Rc::clone(&product));
        }
        Ok(())
    }

    /// Capture the complete immutable import graph on only the freshly
    /// selected current definitions. Reused prior bodies keep their capture.
    pub fn attach_current_selected_static_capture(
        &mut self,
        ids: impl IntoIterator<Item = FunctionId>,
        capture: &Rc<SelectedStaticCapture>,
    ) -> Result<(), &'static str> {
        let ids = ids.into_iter().collect::<Vec<_>>();
        for id in &ids {
            let body = self
                .functions
                .get(id)
                .ok_or("selected static capture has no current definition")?;
            let product = body
                .selected_executable_product
                .as_ref()
                .ok_or("selected static capture has no checked body product")?;
            if !Rc::ptr_eq(product, capture.product())
                || body
                    .binding_origin
                    .is_none_or(|origin| product.body_digest(origin).is_none())
                || body
                    .selected_source_owner
                    .as_ref()
                    .is_none_or(|owner| !owner.same_issuance(capture.product().source_owner()))
                || body
                    .selected_static_capture
                    .as_ref()
                    .is_some_and(|existing| !Rc::ptr_eq(existing, capture))
            {
                return Err("selected static capture differs from exact definition authority");
            }
        }
        for id in ids {
            self.functions
                .get_mut(&id)
                .expect("selected static capture body was checked")
                .selected_static_capture = Some(Rc::clone(capture));
        }
        Ok(())
    }

    /// Attach one validated product to only the exact current definitions
    /// listed by the selected compiler. Every body is checked before mutation.
    pub fn attach_current_selected_executable_product(
        &mut self,
        ids: impl IntoIterator<Item = FunctionId>,
        product: Rc<SelectedExecutableProduct>,
    ) -> Result<(), SelectedExecutableAttachmentError> {
        let ids = ids.into_iter().collect::<Vec<_>>();
        let selected_fingerprints = self
            .selected_runtime_body_fingerprints_with_imports(&ids, &product.fingerprint_imports())
            .map_err(SelectedExecutableAttachmentError::Fingerprints)?;
        for id in &ids {
            let body = self
                .functions
                .get(id)
                .ok_or(SelectedExecutableAttachmentError::MissingDefinition(*id))?;
            validate_selected_product_body(body, *id, selected_fingerprints.get(id), &product)?;
        }
        for id in ids {
            self.functions
                .get_mut(&id)
                .expect("selected executable body was validated")
                .selected_executable_product = Some(Rc::clone(&product));
        }
        Ok(())
    }
}

#[cfg(test)]
mod attachment_error_tests {
    use std::error::Error;

    use super::SelectedExecutableAttachmentError;
    use crate::RetainedCallableStageError;

    #[test]
    fn fingerprint_rejection_retains_its_typed_stage_source() {
        let error = SelectedExecutableAttachmentError::Fingerprints(
            RetainedCallableStageError::MissingPortableSource,
        );
        assert!(matches!(
            &error,
            SelectedExecutableAttachmentError::Fingerprints(
                RetainedCallableStageError::MissingPortableSource
            )
        ));
        assert!(error.source().is_some_and(|source| {
            source
                .downcast_ref::<RetainedCallableStageError>()
                .is_some_and(|source| {
                    matches!(source, RetainedCallableStageError::MissingPortableSource)
                })
        }));
    }
}
