//! Exact code/source product carried by each selected definition. Its optional
//! local dynamic-source port owns only an invocation-captured capability and
//! a weak session evaluator; it has no strong edge back to a function body or
//! registry, so the last external definition lease can retire it normally.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use runmat_execution::executable::{
    selected_unit_content_digest, ExecutableStaticImport, SelectedBodySemanticDigest,
    SelectedUnitContentDigest,
};
use runmat_execution::{Digest, FunctionBindingOrigin};
use runmat_runtime::source_context::SelectedProgramSourceOwner;
use runmat_value::{FunctionDefinitionOwner, StaticSiteDefinitionLease};

use runmat_hir::FunctionId;

use super::FunctionRegistry;

#[derive(Debug)]
pub struct SelectedExecutableProduct {
    canonical_v3: Arc<[u8]>,
    full_digest: Digest,
    source_owner: SelectedProgramSourceOwner,
    content_digest: SelectedUnitContentDigest,
    program: Digest,
    body_fingerprints: HashMap<FunctionBindingOrigin, String>,
    body_semantic_digests: HashMap<FunctionBindingOrigin, SelectedBodySemanticDigest>,
    static_imports: Vec<ExecutableStaticImport>,
    foreign_basis: Option<Digest>,
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
        validate_static_product_dependencies(&product, &dependencies)?;
        Ok(Rc::new(Self {
            product,
            dependencies: Rc::new(dependencies),
            site_owners: Rc::new(Vec::new()),
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
        &self.canonical_v3
    }

    pub fn full_digest(&self) -> Digest {
        self.full_digest
    }

    pub fn foreign_basis(&self) -> Option<Digest> {
        self.foreign_basis
    }

    pub fn interop_receipt(&self) -> Option<&std::rc::Rc<crate::SelectedUnitInteropReceipt>> {
        self.interop_receipt.as_ref()
    }

    pub fn source_owner(&self) -> &SelectedProgramSourceOwner {
        &self.source_owner
    }

    pub fn content_digest(&self) -> SelectedUnitContentDigest {
        self.content_digest
    }

    pub fn body_digest(&self, origin: FunctionBindingOrigin) -> Option<SelectedBodySemanticDigest> {
        self.body_semantic_digests.get(&origin).copied()
    }

    pub fn body_fingerprint(&self, origin: FunctionBindingOrigin) -> Option<&str> {
        self.body_fingerprints.get(&origin).map(String::as_str)
    }

    pub fn static_imports(&self) -> &[ExecutableStaticImport] {
        &self.static_imports
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
        let admitted =
            crate::admit_executable_unit(&canonical_v3).map_err(|error| error.to_string())?;
        let interop = &admitted.envelope().manifest.interop;
        let has_interop = !interop.foreign_types.is_empty()
            || !interop.adapters.is_empty()
            || !interop.adapter_contracts.is_empty();
        if has_interop != foreign_basis.is_some() || has_interop != interop_receipt.is_some() {
            return Err(
                "selected executable foreign basis differs from its interop manifest".into(),
            );
        }
        if interop_receipt
            .as_ref()
            .is_some_and(|receipt| !receipt.matches_selected_unit(&canonical_v3, &source_owner))
        {
            return Err("selected executable foreign receipt differs from its issued V3".into());
        }
        if admitted.admitted_source_map() != source_owner.portable_map() {
            return Err("selected executable differs from its issued source map".into());
        }
        let content_digest =
            selected_unit_content_digest(admitted.envelope()).map_err(|error| error.to_string())?;
        let program = admitted
            .envelope()
            .manifest
            .identity
            .program
            .identity_digest()
            .map_err(|error| error.to_string())?;
        let body_fingerprints = admitted
            .retained_callable_origin_fingerprints()
            .map_err(|error| error.to_string())?;
        let body_semantic_digests = admitted
            .admitted_semantic_body_digests()
            .map_err(|error| error.to_string())?;
        let static_imports = admitted.static_imports().to_vec();
        Ok(Rc::new(Self {
            full_digest: Digest::sha256(&canonical_v3),
            canonical_v3,
            source_owner,
            content_digest,
            program,
            body_fingerprints,
            body_semantic_digests,
            static_imports,
            foreign_basis,
            interop_receipt,
            dynamic_admission,
        }))
    }

    pub fn dynamic_admission(
        &self,
    ) -> Option<&crate::program_execution::dynamic_admission::DynamicEvalAdmissionPort> {
        self.dynamic_admission.as_ref()
    }
}

impl FunctionRegistry {
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
    ) -> Result<(), &'static str> {
        let ids = ids.into_iter().collect::<Vec<_>>();
        let selected_fingerprints = self
            .selected_runtime_body_fingerprints_with_imports(&ids, product.static_imports())
            .map_err(|_| "selected executable body fingerprints are invalid")?;
        for id in &ids {
            let body = self
                .functions
                .get(id)
                .ok_or("selected executable has no matching current definition")?;
            let origin = body
                .binding_origin
                .ok_or("selected executable body has no portable origin")?;
            if origin.program != product.program
                || selected_fingerprints
                    .get(id)
                    .is_none_or(|(checked, fingerprint)| {
                        *checked != origin
                            || product.body_fingerprints.get(&origin) != Some(fingerprint)
                    })
                || body
                    .selected_source_owner
                    .as_ref()
                    .is_none_or(|owner| !owner.same_issuance(product.source_owner()))
                || !body
                    .source_id
                    .and_then(super::BytecodeSourceRef::runtime)
                    .is_some_and(|source| product.source_owner().owns_source(source))
            {
                return Err("selected executable differs from definition/source authority");
            }
            if body
                .selected_executable_product
                .as_ref()
                .is_some_and(|existing| {
                    existing.full_digest() != product.full_digest()
                        || !existing
                            .source_owner()
                            .same_issuance(product.source_owner())
                        || existing.foreign_basis() != product.foreign_basis()
                        || match (existing.interop_receipt(), product.interop_receipt()) {
                            (None, None) => false,
                            (Some(left), Some(right)) => !left.same_selected_admission(right),
                            _ => true,
                        }
                })
            {
                return Err("selected executable would replace an exact prior product");
            }
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
