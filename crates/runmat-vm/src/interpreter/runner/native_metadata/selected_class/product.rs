use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use runmat_hir::FunctionId;
use runmat_runtime::source_context::SelectedProgramSourceOwner;
use runmat_types::ClassIdentity;

use super::{checked_class_instruction, mex, selected_declarations_from, Instr};
use crate::FunctionRegistry;

#[path = "product/object_facts.rs"]
mod object_facts;

/// One admitted constructor's complete class ancestry and the exact issued
/// source tables that produced it. It owns neither a RuntimeContext nor an
/// execution service, and is retained with the entrypoint definition product.
#[derive(Debug)]
pub struct SelectedClassProduct {
    class: ClassIdentity,
    declarations: Arc<Vec<Instr>>,
    constructor_source_owner: Option<SelectedProgramSourceOwner>,
    _source_owners: Rc<Vec<SelectedProgramSourceOwner>>,
    /// Class-only dependencies have no callable definition lease. Their
    /// admitted bytes and issued source owners therefore live with this exact
    /// constructor product, not with the generation's callable-owner list.
    declaration_products: Rc<Vec<SelectedDeclarationProduct>>,
    projection: RefCell<Option<Rc<runmat_runtime::class_registry::SelectedClassProjection>>>,
    receiver_seal: Arc<runmat_value::SelectedReceiverDeclarationSeal>,
    receiver_key: runmat_value::SelectedReceiverDeclarationKey,
}

#[derive(Clone, Debug)]
pub struct SelectedDeclarationProduct {
    class: ClassIdentity,
    canonical_bytes: Arc<[u8]>,
    source_owner: SelectedProgramSourceOwner,
    interop_receipt: Option<std::rc::Rc<crate::SelectedUnitInteropReceipt>>,
    static_imports: Rc<Vec<runmat_execution::executable::ExecutableStaticImport>>,
    static_targets: Rc<Vec<crate::SelectedStaticProductTarget>>,
    declarations: Arc<Vec<Instr>>,
}

impl SelectedDeclarationProduct {
    /// Retain a metadata-only selected unit against its already-issued source
    /// owner. No callable ordinal, constructor body, or second source bind is
    /// manufactured for a class-only interactive publication.
    pub fn admit_local(
        canonical_bytes: Arc<[u8]>,
        source_owner: SelectedProgramSourceOwner,
        interop_receipt: Option<std::rc::Rc<crate::SelectedUnitInteropReceipt>>,
    ) -> Result<Rc<Self>, runmat_runtime::RuntimeError> {
        let admitted = crate::admit_executable_unit(&canonical_bytes).map_err(|_| {
            mex(
                "ClassProductInvalid",
                "declaration-only selected product is not an admitted executable unit",
            )
        })?;
        let class = admitted.declaration_class().ok_or_else(|| {
            mex(
                "ClassProductInvalid",
                "selected declaration product has a callable entrypoint",
            )
        })?;
        let interop = &admitted.envelope().manifest.interop;
        let has_interop = !interop.foreign_types.is_empty()
            || !interop.adapters.is_empty()
            || !interop.adapter_contracts.is_empty();
        if !source_owner.is_live()
            || source_owner.portable_map() != admitted.admitted_source_map()
            || interop_receipt.is_some() != has_interop
            || interop_receipt.as_ref().is_some_and(|receipt| {
                !receipt.matches_selected_unit(&canonical_bytes, &source_owner)
            })
        {
            return Err(mex(
                "ClassProductInvalid",
                "selected declaration differs from its source or foreign receipt",
            ));
        }
        Ok(Rc::new(Self {
            class: class.clone(),
            canonical_bytes,
            source_owner,
            interop_receipt,
            static_imports: Rc::new(admitted.static_imports().to_vec()),
            static_targets: Rc::new(Vec::new()),
            declarations: Arc::new(
                admitted
                    .bytecode()
                    .instructions
                    .iter()
                    .filter(|instruction| checked_class_instruction(instruction).is_some())
                    .cloned()
                    .collect(),
            ),
        }))
    }

    /// Attach only checked, port-free code products for the declaration's
    /// already-sealed static sites. No registry or counted definition lease is
    /// retained, so mutually importing classes cannot self-pin their bodies.
    pub fn with_static_targets(
        self: &Rc<Self>,
        static_targets: Vec<crate::SelectedStaticProductTarget>,
    ) -> Result<Rc<Self>, runmat_runtime::RuntimeError> {
        if self.static_imports.len() != static_targets.len()
            || self
                .static_imports
                .iter()
                .zip(&static_targets)
                .any(|(import, target)| {
                    *import != target.import()
                        || !self
                            .source_owner
                            .session_lease()
                            .ptr_eq(&target.product().source_owner().session_lease())
                })
        {
            return Err(mex(
                "ClassProductStaticImportMismatch",
                "selected declaration targets differ from its sealed static imports",
            ));
        }
        Ok(Rc::new(Self {
            class: self.class.clone(),
            canonical_bytes: Arc::clone(&self.canonical_bytes),
            source_owner: self.source_owner.clone(),
            interop_receipt: self.interop_receipt.clone(),
            static_imports: Rc::clone(&self.static_imports),
            static_targets: Rc::new(static_targets),
            declarations: Arc::clone(&self.declarations),
        }))
    }

    pub fn class(&self) -> &ClassIdentity {
        &self.class
    }

    pub fn canonical_bytes(&self) -> &Arc<[u8]> {
        &self.canonical_bytes
    }

    pub fn source_owner(&self) -> &SelectedProgramSourceOwner {
        &self.source_owner
    }

    pub fn interop_receipt(&self) -> Option<&std::rc::Rc<crate::SelectedUnitInteropReceipt>> {
        self.interop_receipt.as_ref()
    }

    pub fn static_targets(&self) -> &[crate::SelectedStaticProductTarget] {
        &self.static_targets
    }

    pub fn declarations(&self) -> &[Instr] {
        &self.declarations
    }
}

impl SelectedClassProduct {
    /// Capture class-only ancestry from the exact generation selected at
    /// constructor publication. Each retained declaration includes only
    /// immutable code/source/foreign facts and port-free static target
    /// products; no registry or counted body lease can point back here.
    pub fn with_declaration_products(
        mut self,
        products: &[Rc<SelectedDeclarationProduct>],
    ) -> Result<Self, runmat_runtime::RuntimeError> {
        if !self.declaration_products.is_empty() {
            return Err(mex(
                "ClassProductAlreadySelected",
                "selected class ancestry was already attached",
            ));
        }
        // A declaration already selected by this constructor's own exact
        // code/source product must not acquire a later same-named generation
        // product merely because its RegisterClass metadata happens to match.
        let (_, owned_indices) = selected_declarations_from(self.declarations.iter(), &self.class)?;
        let (selected, indices) = selected_declarations_from(
            self.declarations.iter().chain(
                products
                    .iter()
                    .filter(|product| !owned_indices.contains_key(product.class()))
                    .flat_map(|product| product.declarations()),
            ),
            &self.class,
        )?;
        let retained = products
            .iter()
            .filter(|product| {
                indices.contains_key(product.class())
                    && !owned_indices.contains_key(product.class())
            })
            .map(|product| product.as_ref().clone())
            .collect::<Vec<_>>();
        let mut owners = self._source_owners.as_ref().clone();
        owners.extend(
            retained
                .iter()
                .map(|product: &SelectedDeclarationProduct| product.source_owner().clone()),
        );
        self.declarations = Arc::new(selected.into_iter().cloned().collect());
        self._source_owners = Rc::new(owners);
        self.declaration_products = Rc::new(retained);
        Ok(self)
    }

    /// Current interactive publication uses its exact already-issued source
    /// owner and only the declarations in that compilation's bytecode. It
    /// never mixes same-named classes from older registry generations.
    pub fn from_admitted_instructions<'a>(
        instructions: impl IntoIterator<Item = &'a Instr>,
        registry: &'a FunctionRegistry,
        constructor: FunctionId,
        class: &ClassIdentity,
    ) -> Result<Self, runmat_runtime::RuntimeError> {
        let owners = registry
            .definition_dependency_owners(constructor)
            .ok_or_else(|| {
                mex(
                    "ClassDependencyMissing",
                    "selected constructor dependency graph is incomplete",
                )
            })?;
        let constructor_source_owner = registry
            .get_current(constructor)
            .and_then(|body| body.selected_source_owner.as_ref())
            .cloned()
            .ok_or_else(|| {
                mex(
                    "ClassSourceOwnerMissing",
                    "selected constructor lacks its issued source owner",
                )
            })?;
        let mut selected_sources = Vec::with_capacity(owners.len());
        let mut dependency_instructions = Vec::new();
        for (function, owner) in owners {
            let body = registry
                .get_selected_definition(function, &owner)
                .ok_or_else(|| {
                    mex(
                        "ClassDependencyRetired",
                        "selected constructor dependency body retired",
                    )
                })?;
            if let Some(source_owner) = &body.selected_source_owner {
                selected_sources.push(source_owner.clone());
            } else if body
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instr::RegisterClass { .. }))
            {
                return Err(mex(
                    "ClassSourceOwnerMissing",
                    "selected class declaration lacks its issued source owner",
                ));
            }
            dependency_instructions.extend(body.instructions.iter());
        }
        let (selected, _) = selected_declarations_from(
            instructions.into_iter().chain(dependency_instructions),
            class,
        )?;
        if selected.is_empty() {
            return Err(mex(
                "ClassMetadataMissing",
                "selected constructor lacks its class declaration",
            ));
        }
        let (receiver_seal, receiver_key) = runmat_value::SelectedReceiverDeclarationSeal::issue();
        Ok(Self {
            class: class.clone(),
            declarations: Arc::new(selected.into_iter().cloned().collect()),
            constructor_source_owner: Some(constructor_source_owner),
            _source_owners: Rc::new(selected_sources),
            declaration_products: Rc::new(Vec::new()),
            projection: RefCell::new(None),
            receiver_seal,
            receiver_key,
        })
    }

    pub fn from_bound_units(
        units: &[&crate::RuntimeBoundExecutableUnit],
        class: &ClassIdentity,
    ) -> Result<Self, runmat_runtime::RuntimeError> {
        let receipts = vec![None; units.len()];
        Self::from_bound_units_with_receipts(units, &receipts, class)
    }

    pub fn from_bound_units_with_receipts(
        units: &[&crate::RuntimeBoundExecutableUnit],
        receipts: &[Option<std::rc::Rc<crate::SelectedUnitInteropReceipt>>],
        class: &ClassIdentity,
    ) -> Result<Self, runmat_runtime::RuntimeError> {
        if units.len() != receipts.len() {
            return Err(mex(
                "ClassInteropReceiptMismatch",
                "selected class source and foreign receipt counts differ",
            ));
        }
        let mut source_owners = Vec::with_capacity(units.len());
        let mut constructor_source_owner = None;
        let mut declaration_products = Vec::new();
        for (unit, receipt) in units.iter().zip(receipts) {
            let source_owner = unit.selected_source_owner().ok_or_else(|| {
                mex(
                    "ClassSourceOwnerMissing",
                    "selected class product lacks its admitted source lease",
                )
            })?;
            if receipt.is_some() != unit.declares_interop()
                || receipt.as_ref().is_some_and(|receipt| {
                    !unit.canonical_v3().is_some_and(|canonical| {
                        receipt.matches_selected_unit(canonical, source_owner)
                    })
                })
            {
                return Err(mex(
                    "ClassInteropReceiptMismatch",
                    "selected class foreign receipt differs from its issued V3",
                ));
            }
            source_owners.push(source_owner.clone());
            for body in unit.registry().functions.values() {
                if body.class_method_owner.as_ref().is_some_and(|owner| {
                    owner.role == runmat_types::ClassMethodRole::Constructor
                        && !owner.is_static
                        && &owner.declaring_class == class
                }) {
                    let exact = body.selected_source_owner.as_ref().ok_or_else(|| {
                        mex(
                            "ClassSourceOwnerMissing",
                            "selected constructor body lacks its issued source owner",
                        )
                    })?;
                    if !exact.same_issuance(source_owner)
                        || constructor_source_owner.replace(exact.clone()).is_some()
                    {
                        return Err(mex(
                            "ClassProductInvalid",
                            "selected class product has ambiguous constructor source",
                        ));
                    }
                }
            }
            if let Some(declaration_class) = unit.declaration_class() {
                declaration_products.push(SelectedDeclarationProduct {
                    class: declaration_class.clone(),
                    canonical_bytes: Arc::clone(unit.canonical_v3().ok_or_else(|| {
                        mex(
                            "ClassProductMissing",
                            "declaration-only dependency has no admitted product bytes",
                        )
                    })?),
                    source_owner: source_owner.clone(),
                    interop_receipt: receipt.clone(),
                    static_imports: Rc::new(unit.static_imports().to_vec()),
                    static_targets: Rc::new(Vec::new()),
                    declarations: Arc::new(
                        unit.bytecode()
                            .instructions
                            .iter()
                            .filter(|instruction| checked_class_instruction(instruction).is_some())
                            .cloned()
                            .collect(),
                    ),
                });
            }
        }
        let instructions = units.iter().flat_map(|unit| {
            unit.bytecode().instructions.iter().chain(
                unit.registry()
                    .functions
                    .values()
                    .flat_map(|body| body.instructions.iter()),
            )
        });
        let (selected, indices) = selected_declarations_from(instructions, class)?;
        if declaration_products
            .iter()
            .any(|product| !indices.contains_key(&product.class))
        {
            return Err(mex(
                "ClassMetadataMismatch",
                "declaration-only dependency is outside selected class ancestry",
            ));
        }
        if selected.is_empty() {
            return Err(mex(
                "ClassMetadataMissing",
                "selected constructor lacks its class declaration",
            ));
        }
        let constructor_source_owner = constructor_source_owner.ok_or_else(|| {
            mex(
                "ClassSourceOwnerMissing",
                "selected class product has no constructor definition",
            )
        })?;
        let (receiver_seal, receiver_key) = runmat_value::SelectedReceiverDeclarationSeal::issue();
        Ok(Self {
            class: class.clone(),
            declarations: Arc::new(selected.into_iter().cloned().collect()),
            constructor_source_owner: Some(constructor_source_owner),
            _source_owners: Rc::new(source_owners),
            declaration_products: Rc::new(declaration_products),
            projection: RefCell::new(None),
            receiver_seal,
            receiver_key,
        })
    }

    pub(crate) fn receiver_seal(&self) -> &Arc<runmat_value::SelectedReceiverDeclarationSeal> {
        &self.receiver_seal
    }

    pub(crate) fn admits_constructor_source(&self, owner: &SelectedProgramSourceOwner) -> bool {
        self.constructor_source_owner
            .as_ref()
            .is_some_and(|selected| selected.same_issuance(owner))
    }

    /// Bind the already-proved constructor lease to this product's one-time
    /// pure metadata slot. Worker Graph admission performs this after exact
    /// unit/body/source checks and before materializing any selected object.
    pub(crate) fn seal_receiver_declarations(
        &self,
        binding: &runmat_value::BoundFunctionBinding,
        declarations: Arc<runmat_value::SelectedReceiverDeclarations>,
    ) -> Result<(), &'static str> {
        let owner = binding
            .authority
            .local_lease(binding.function)
            .ok_or("selected constructor definition lease is absent")?
            .owner_arc();
        if !owner.admit_selected_receiver_seal(Arc::clone(&self.receiver_seal))
            || !self.receiver_seal.fill(&self.receiver_key, declarations)
        {
            return Err("selected constructor owns another declaration product");
        }
        Ok(())
    }

    pub fn class(&self) -> &ClassIdentity {
        &self.class
    }

    pub fn declarations(&self) -> &[Instr] {
        &self.declarations
    }

    /// Exact declaration-only products retained by this selected constructor.
    /// These are metadata/source units, not callable definitions.
    pub fn declaration_products(&self) -> &[SelectedDeclarationProduct] {
        &self.declaration_products
    }

    pub fn invocation_runtime(
        &self,
        runtime: runmat_runtime::context::RuntimeContext,
    ) -> Result<runmat_runtime::context::RuntimeContext, runmat_runtime::RuntimeError> {
        Ok(runtime.with_selected_class_projection(self.invocation_projection()?))
    }

    /// Materialize defaults/access once for this exact immutable product,
    /// never from a later mutable class-name registration. The RootedValue
    /// lease inside the projection covers suspended and nested invocations.
    pub fn invocation_projection(
        &self,
    ) -> Result<
        Rc<runmat_runtime::class_registry::SelectedClassProjection>,
        runmat_runtime::RuntimeError,
    > {
        if let Some(projection) = self.projection.borrow().as_ref() {
            return Ok(Rc::clone(projection));
        }
        let classes = self
            .declarations
            .iter()
            .filter_map(checked_class_instruction)
            .collect();
        let projection =
            runmat_runtime::class_registry::prepare_selected_class_projection(classes)?;
        *self.projection.borrow_mut() = Some(Rc::clone(&projection));
        Ok(projection)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use runmat_hir::FunctionId;
    use runmat_value::{FunctionDefinitionLease, FunctionDefinitionOwner};

    use super::*;
    use crate::bytecode::FunctionBytecode;

    #[test]
    fn selected_metadata_does_not_self_pin_a_retired_definition() {
        let owner = Arc::new(FunctionDefinitionOwner::default());
        let retired = Arc::downgrade(&owner);
        let external: FunctionDefinitionLease = FunctionDefinitionOwner::issue(&owner, 1);
        let (receiver_seal, receiver_key) = runmat_value::SelectedReceiverDeclarationSeal::issue();
        let product = Rc::new(SelectedClassProduct {
            class: ClassIdentity::new("RetiredConstructor").expect("class identity"),
            declarations: Arc::new(Vec::new()),
            constructor_source_owner: None,
            _source_owners: Rc::new(Vec::new()),
            declaration_products: Rc::new(Vec::new()),
            projection: RefCell::new(None),
            receiver_seal,
            receiver_key,
        });
        let original = FunctionBytecode {
            function: FunctionId(1),
            display_name: "retired_constructor".into(),
            definition_owner: owner,
            selected_class_product: Some(product),
            ..Default::default()
        };
        let mut registry = FunctionRegistry::new(HashMap::from([(FunctionId(1), original)]));
        registry.insert_replacing_name(FunctionBytecode {
            function: FunctionId(1),
            display_name: "replacement_constructor".into(),
            ..Default::default()
        });
        assert!(
            retired.upgrade().is_some(),
            "external handle keeps old body"
        );
        drop(external);
        registry.insert_replacing_name(FunctionBytecode {
            function: FunctionId(2),
            display_name: "another_definition".into(),
            ..Default::default()
        });
        assert!(
            retired.upgrade().is_none(),
            "selected class metadata must not count as a callable handle"
        );
    }
}
