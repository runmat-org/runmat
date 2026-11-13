use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use runmat_execution::{Digest, FunctionBindingOrigin};
use runmat_runtime::{build_runtime_error, RuntimeError};

use super::SelectedStaticTarget;

/// A class-only product and the exact imported code leases selected with its
/// sealed declaration sites. This lives in the existing generation, not in a
/// name-keyed class registry; neither the product nor targets own runtime ports.
#[derive(Debug)]
pub(in crate::session) struct SelectedDeclarationAdmission {
    product: Rc<runmat_vm::SelectedDeclarationProduct>,
    // Retains the exact issuing definition leases through this generation;
    // the VM product separately owns their port-free transitive code closure.
    _static_targets: Vec<SelectedStaticTarget>,
}

impl SelectedDeclarationAdmission {
    pub(in crate::session) fn new(
        product: Rc<runmat_vm::SelectedDeclarationProduct>,
        static_targets: Vec<SelectedStaticTarget>,
    ) -> Result<Rc<Self>, RuntimeError> {
        let admitted = runmat_vm::admit_executable_unit(product.canonical_bytes())
            .map_err(|error| build_runtime_error(error.to_string()).build())?;
        let imports = admitted.static_imports();
        if imports.len() != static_targets.len()
            || imports
                .iter()
                .zip(&static_targets)
                .any(|(import, selected)| *import != selected.import)
        {
            return Err(build_runtime_error(
                "selected declaration imports differ from exact target leases",
            )
            .with_identifier("RunMat:ClassProduct:StaticImportMismatch")
            .build());
        }
        let portable_targets = static_targets
            .iter()
            .map(|selected| {
                if !selected
                    .product
                    .source_owner()
                    .same_issuance(&selected.source_owner)
                {
                    return Err(build_runtime_error(
                        "selected declaration target changed its source issuance",
                    )
                    .with_identifier("RunMat:ClassProduct:StaticImportMismatch")
                    .build());
                }
                let target = runmat_vm::SelectedStaticProductTarget::new(
                    selected.import,
                    Rc::clone(&selected.product),
                )
                .map_err(|error| {
                    build_runtime_error(error)
                        .with_identifier("RunMat:ClassProduct:StaticImportMismatch")
                        .build()
                })?;
                let dependencies = capture_static_product_dependencies(selected)?;
                target.with_dependencies(dependencies).map_err(|error| {
                    build_runtime_error(error)
                        .with_identifier("RunMat:ClassProduct:StaticImportMismatch")
                        .build()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let product = product.with_static_targets(portable_targets)?;
        Ok(Rc::new(Self {
            product,
            _static_targets: static_targets,
        }))
    }

    pub(in crate::session) fn product(&self) -> &Rc<runmat_vm::SelectedDeclarationProduct> {
        &self.product
    }
}

pub(in crate::session) fn capture_static_product_dependencies(
    selected: &SelectedStaticTarget,
) -> Result<Vec<runmat_vm::SelectedStaticProductDependency>, RuntimeError> {
    let mismatch = |message| {
        build_runtime_error(message)
            .with_identifier("RunMat:ClassProduct:StaticImportMismatch")
            .build()
    };
    let body = selected
        .view
        .get_owned(selected.function, &selected.definition)
        .ok_or_else(|| mismatch("selected static target lost its exact definition lease"))?;
    if body.binding_origin != Some(selected.import.target)
        || body
            .selected_executable_product
            .as_ref()
            .is_none_or(|product| !Rc::ptr_eq(product, &selected.product))
        || body
            .selected_source_owner
            .as_ref()
            .is_none_or(|owner| !owner.same_issuance(&selected.source_owner))
        || !selected
            .product
            .source_owner()
            .same_issuance(&selected.source_owner)
    {
        return Err(mismatch(
            "selected static target changed its exact body or source",
        ));
    }
    match body.selected_static_capture.as_ref() {
        Some(capture) if Rc::ptr_eq(capture.product(), &selected.product) => {
            Ok(capture.dependencies().to_vec())
        }
        None if selected.product.static_imports().is_empty() => Ok(Vec::new()),
        _ => Err(mismatch(
            "selected static target lacks its exact captured imports",
        )),
    }
}

fn retain_latest_declaration(
    declarations: &mut Vec<Rc<SelectedDeclarationAdmission>>,
    incoming: Rc<SelectedDeclarationAdmission>,
) {
    declarations.retain(|prior| prior.product.class() != incoming.product.class());
    declarations.push(incoming);
}

#[path = "generation/index_error.rs"]
mod index_error;
pub(super) use index_error::CallableArtifactIndexError;

#[derive(Clone, Debug, Default)]
pub(super) struct CallableArtifactIndex {
    units: HashMap<FunctionBindingOrigin, IndexedCallableUnit>,
}

#[derive(Clone, Debug)]
struct IndexedCallableUnit {
    fingerprint: String,
    bytes: Arc<[u8]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CallableArtifactSelectionError {
    Missing(Digest),
    Limit { field: &'static str, limit: usize },
}

impl CallableArtifactIndex {
    pub(super) fn canonical_unit_for_origin(
        &self,
        origin: &FunctionBindingOrigin,
    ) -> Option<Arc<[u8]>> {
        self.units.get(origin).map(|unit| Arc::clone(&unit.bytes))
    }

    pub(super) fn insert_canonical_unit(
        &mut self,
        bytes: Vec<u8>,
    ) -> Result<(), CallableArtifactIndexError> {
        self.insert_canonical_arc_unit(Arc::from(bytes))
    }

    /// Share an already admitted immutable product with its retained source
    /// owner rather than copying the full V3 envelope during publication.
    pub(super) fn insert_canonical_arc_unit(
        &mut self,
        bytes: Arc<[u8]>,
    ) -> Result<(), CallableArtifactIndexError> {
        let admitted = runmat_vm::admit_executable_unit(&bytes)
            .map_err(CallableArtifactIndexError::InvalidUnit)?;
        let fingerprints = admitted
            .retained_callable_origin_fingerprints()
            .map_err(CallableArtifactIndexError::InvalidDefinitions)?;
        for (origin, fingerprint) in fingerprints {
            match self.units.get(&origin) {
                Some(existing) if existing.fingerprint != fingerprint => {
                    return Err(CallableArtifactIndexError::ConflictingOrigin);
                }
                Some(_) => {}
                None => {
                    self.units.insert(
                        origin,
                        IndexedCallableUnit {
                            fingerprint,
                            bytes: Arc::clone(&bytes),
                        },
                    );
                }
            }
        }
        Ok(())
    }

    pub(super) fn units_for_origins(
        &self,
        origins: &HashSet<FunctionBindingOrigin>,
        max_units: usize,
        max_unit_bytes: usize,
    ) -> Result<Vec<Vec<u8>>, CallableArtifactSelectionError> {
        let mut origins = origins.iter().copied().collect::<Vec<_>>();
        origins.sort_by_key(|origin| (origin.program, origin.function));
        let mut seen = HashSet::new();
        let mut selected = Vec::new();
        let mut total_bytes = 0usize;
        for origin in origins {
            let unit = self
                .units
                .get(&origin)
                .ok_or(CallableArtifactSelectionError::Missing(origin.program))?;
            let digest = Digest::sha256(unit.bytes.as_ref());
            if seen.insert(digest) {
                if selected.len() == max_units {
                    return Err(CallableArtifactSelectionError::Limit {
                        field: "callable_artifact.units",
                        limit: max_units,
                    });
                }
                total_bytes = total_bytes.checked_add(unit.bytes.len()).ok_or(
                    CallableArtifactSelectionError::Limit {
                        field: "callable_artifact.unit_bytes",
                        limit: max_unit_bytes,
                    },
                )?;
                if total_bytes > max_unit_bytes {
                    return Err(CallableArtifactSelectionError::Limit {
                        field: "callable_artifact.unit_bytes",
                        limit: max_unit_bytes,
                    });
                }
                selected.push(unit);
            }
        }
        Ok(selected
            .into_iter()
            .map(|unit| unit.bytes.to_vec())
            .collect())
    }

    fn retain_origins_from(&mut self, source: &Self, origins: &HashSet<FunctionBindingOrigin>) {
        for origin in origins {
            if let Some(unit) = source.units.get(origin) {
                self.units.insert(*origin, unit.clone());
            }
        }
    }

    fn retain_origins(&mut self, origins: &HashSet<FunctionBindingOrigin>) {
        self.units.retain(|origin, _| origins.contains(origin));
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FunctionRegistryGenerationState {
    Active,
    Committed,
    Invalidated,
}

#[derive(Debug)]
pub(super) struct FunctionRegistryGeneration {
    registry: RefCell<Rc<runmat_vm::FunctionRegistry>>,
    artifacts: RefCell<Rc<CallableArtifactIndex>>,
    source_owners: RefCell<Rc<Vec<Rc<runmat_vm::RetainedExecutableSourceOwner>>>>,
    selected_declarations: RefCell<Rc<Vec<Rc<SelectedDeclarationAdmission>>>>,
    state: Cell<FunctionRegistryGenerationState>,
    epoch: RefCell<Rc<()>>,
}

pub(super) struct PreparedGenerationImport {
    registry: Rc<runmat_vm::FunctionRegistry>,
    artifacts: Rc<CallableArtifactIndex>,
    source_owners: Rc<Vec<Rc<runmat_vm::RetainedExecutableSourceOwner>>>,
    selected_declarations: Rc<Vec<Rc<SelectedDeclarationAdmission>>>,
}

pub(super) struct GenerationRetirement {
    _registry: Rc<runmat_vm::FunctionRegistry>,
    _artifacts: Rc<CallableArtifactIndex>,
    _source_owners: Rc<Vec<Rc<runmat_vm::RetainedExecutableSourceOwner>>>,
    _selected_declarations: Rc<Vec<Rc<SelectedDeclarationAdmission>>>,
}

impl PreparedGenerationImport {
    pub(super) fn new(
        registry: runmat_vm::FunctionRegistry,
        artifacts: CallableArtifactIndex,
        source_owners: Vec<Rc<runmat_vm::RetainedExecutableSourceOwner>>,
    ) -> Self {
        Self {
            registry: Rc::new(registry),
            artifacts: Rc::new(artifacts),
            source_owners: Rc::new(source_owners),
            selected_declarations: Rc::new(Vec::new()),
        }
    }
}

impl FunctionRegistryGeneration {
    pub(super) fn committed(registry: runmat_vm::FunctionRegistry) -> Self {
        Self {
            registry: RefCell::new(Rc::new(registry)),
            artifacts: RefCell::new(Rc::new(CallableArtifactIndex::default())),
            source_owners: RefCell::new(Rc::new(Vec::new())),
            selected_declarations: RefCell::new(Rc::new(Vec::new())),
            state: Cell::new(FunctionRegistryGenerationState::Committed),
            epoch: RefCell::new(Rc::new(())),
        }
    }

    pub(super) fn active(
        mut registry: runmat_vm::FunctionRegistry,
        artifacts: CallableArtifactIndex,
        source_owners: Rc<Vec<Rc<runmat_vm::RetainedExecutableSourceOwner>>>,
        selected_declarations: Rc<Vec<Rc<SelectedDeclarationAdmission>>>,
        inherited: &runmat_vm::FunctionRegistry,
    ) -> Self {
        registry.retain_live_owned_definitions_from(inherited);
        Self {
            registry: RefCell::new(Rc::new(registry)),
            artifacts: RefCell::new(Rc::new(artifacts)),
            source_owners: RefCell::new(source_owners),
            selected_declarations: RefCell::new(selected_declarations),
            state: Cell::new(FunctionRegistryGenerationState::Active),
            epoch: RefCell::new(Rc::new(())),
        }
    }

    pub(super) fn registry(&self) -> Result<Rc<runmat_vm::FunctionRegistry>, RuntimeError> {
        self.ensure_available()?;
        Ok(self.registry.borrow().clone())
    }

    /// Retain an already-admitted class-only product in the active generation.
    /// No callable definition lease, name lookup, or second source issuance is
    /// created. The active publication guard owns rollback if commit fails.
    pub(super) fn attach_selected_declaration(
        &self,
        product: Rc<SelectedDeclarationAdmission>,
    ) -> Result<(), RuntimeError> {
        self.ensure_available()?;
        if self.state.get() != FunctionRegistryGenerationState::Active {
            return Err(
                build_runtime_error("selected declaration requires an active generation")
                    .with_identifier("RunMat:ClassProduct:InactiveGeneration")
                    .build(),
            );
        }
        let current = self.selected_declarations.borrow().clone();
        if current.iter().any(|prior| Rc::ptr_eq(prior, &product)) {
            return Ok(());
        }
        let mut next = current.as_ref().clone();
        retain_latest_declaration(&mut next, product);
        let retired = self.selected_declarations.replace(Rc::new(next));
        self.rotate_epoch();
        drop(retired);
        Ok(())
    }

    pub(super) fn selected_declarations(
        &self,
    ) -> Result<Rc<Vec<Rc<SelectedDeclarationAdmission>>>, RuntimeError> {
        self.ensure_available()?;
        Ok(self.selected_declarations.borrow().clone())
    }

    pub(super) fn artifacts(&self) -> Result<Rc<CallableArtifactIndex>, RuntimeError> {
        self.ensure_available()?;
        // A provisional eval can create its first escaping handle after an
        // earlier artifact lookup. Prune only after publication settles; Drop
        // copies precisely the origins whose definition leases escaped.
        if self.state.get() == FunctionRegistryGenerationState::Active {
            return Ok(self.artifacts.borrow().clone());
        }
        let origins = self.registry.borrow().retained_callable_origins();
        let current = self.artifacts.borrow().clone();
        if current.units.keys().all(|origin| origins.contains(origin)) {
            return Ok(current);
        }
        let mut retained = current.as_ref().clone();
        retained.retain_origins(&origins);
        let retained = Rc::new(retained);
        let retired = self.artifacts.replace(retained.clone());
        self.rotate_epoch();
        drop(retired);
        Ok(retained)
    }

    pub(super) fn source_owners(
        &self,
    ) -> Result<Rc<Vec<Rc<runmat_vm::RetainedExecutableSourceOwner>>>, RuntimeError> {
        self.ensure_available()?;
        let current = self.source_owners.borrow().clone();
        if self.state.get() == FunctionRegistryGenerationState::Active {
            return Ok(current);
        }
        if current
            .iter()
            .all(|owner| owner.has_live_definition_owners())
        {
            return Ok(current);
        }
        let retained = Rc::new(
            current
                .iter()
                .filter(|owner| owner.has_live_definition_owners())
                .cloned()
                .collect::<Vec<_>>(),
        );
        let retired = self.source_owners.replace(retained.clone());
        self.rotate_epoch();
        drop(retired);
        Ok(retained)
    }

    pub(super) fn epoch(&self) -> Rc<()> {
        self.epoch.borrow().clone()
    }

    pub(super) fn matches_epoch(&self, expected: &Rc<()>) -> bool {
        Rc::ptr_eq(&self.epoch.borrow(), expected)
    }

    fn rotate_epoch(&self) {
        let retired = self.epoch.replace(Rc::new(()));
        drop(retired);
    }

    pub(super) fn commit_import(
        &self,
        expected_epoch: &Rc<()>,
        prepared: PreparedGenerationImport,
    ) -> Result<GenerationRetirement, RuntimeError> {
        self.ensure_available()?;
        if !self.matches_epoch(expected_epoch) {
            return Err(build_runtime_error(
                "callable artifact destination changed during admission",
            )
            .with_identifier("RunMat:CallableArtifact:DestinationChanged")
            .build());
        }
        let PreparedGenerationImport {
            registry,
            artifacts,
            source_owners,
            selected_declarations: imported_declarations,
        } = prepared;
        let mut selected_declarations = self.selected_declarations.borrow().as_ref().clone();
        for product in imported_declarations.iter() {
            retain_latest_declaration(&mut selected_declarations, Rc::clone(product));
        }
        let retired = GenerationRetirement {
            _registry: self.registry.replace(registry),
            _artifacts: self.artifacts.replace(artifacts),
            _source_owners: self.source_owners.replace(source_owners),
            _selected_declarations: self
                .selected_declarations
                .replace(Rc::new(selected_declarations)),
        };
        self.rotate_epoch();
        Ok(retired)
    }

    fn ensure_available(&self) -> Result<(), RuntimeError> {
        if self.state.get() == FunctionRegistryGenerationState::Invalidated {
            return Err(build_runtime_error(
                "source evaluation executable generation did not commit",
            )
            .with_identifier("RunMat:SourceEvaluation:GenerationInvalidated")
            .build());
        }
        Ok(())
    }

    pub(super) fn commit(&self, mut committed: runmat_vm::FunctionRegistry) {
        if self.state.get() == FunctionRegistryGenerationState::Active {
            let active = self.registry.borrow().clone();
            committed.retain_live_owned_definitions_from(&active);
            let retired = {
                let mut registry = self.registry.borrow_mut();
                std::mem::replace(&mut *registry, Rc::new(committed))
            };
            self.state.set(FunctionRegistryGenerationState::Committed);
            self.rotate_epoch();
            drop(retired);
        }
    }

    /// Preserve bodies and artifact envelopes that escaped through legitimate
    /// partial language effects before a candidate publication failed. They
    /// remain exact-bound only; no names or current-source membership publish.
    pub(super) fn retain_escaped_into(&self, restore: &Self) {
        if self.state.get() != FunctionRegistryGenerationState::Active {
            return;
        }
        let candidate = self.registry.borrow().clone();
        let candidate_artifacts = self.artifacts.borrow().clone();
        let candidate_source_owners = self.source_owners.borrow().clone();
        let origins = {
            let mut restored = restore.registry.borrow().as_ref().clone();
            let origins = restored.retain_live_owned_definitions_from(&candidate);
            let retired = restore.registry.replace(Rc::new(restored));
            restore.rotate_epoch();
            drop(retired);
            origins
        };
        if !origins.is_empty() {
            let mut restored = restore.artifacts.borrow().as_ref().clone();
            restored.retain_origins_from(&candidate_artifacts, &origins);
            let retired = restore.artifacts.replace(Rc::new(restored));
            restore.rotate_epoch();
            drop(retired);

            let mut restored = restore.source_owners.borrow().as_ref().clone();
            for owner in candidate_source_owners
                .iter()
                .filter(|owner| owner.has_live_definition_owners())
            {
                if !restored.iter().any(|existing| Rc::ptr_eq(existing, owner)) {
                    restored.push(Rc::clone(owner));
                }
            }
            let retired = restore.source_owners.replace(Rc::new(restored));
            restore.rotate_epoch();
            drop(retired);
        }
    }

    pub(super) fn invalidate(&self) {
        if self.state.get() == FunctionRegistryGenerationState::Active {
            self.state.set(FunctionRegistryGenerationState::Invalidated);
            self.rotate_epoch();
            let retired_registry = self
                .registry
                .replace(Rc::new(runmat_vm::FunctionRegistry::default()));
            let retired_artifacts = self
                .artifacts
                .replace(Rc::new(CallableArtifactIndex::default()));
            let retired_source_owners = self.source_owners.replace(Rc::new(Vec::new()));
            let retired_declarations = self.selected_declarations.replace(Rc::new(Vec::new()));
            drop(retired_registry);
            drop(retired_artifacts);
            drop(retired_source_owners);
            drop(retired_declarations);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::ProgramFunctionId;

    fn origin(name: &'static str, function: u32) -> FunctionBindingOrigin {
        FunctionBindingOrigin {
            program: Digest::sha256(name),
            function: ProgramFunctionId(function),
        }
    }

    #[test]
    fn artifact_selection_checks_count_and_bytes_before_copying_units() {
        let first = origin("first", 0);
        let second = origin("second", 0);
        let mut index = CallableArtifactIndex::default();
        index.units.insert(
            first,
            IndexedCallableUnit {
                fingerprint: "first".into(),
                bytes: Arc::from([1_u8, 2, 3]),
            },
        );
        index.units.insert(
            second,
            IndexedCallableUnit {
                fingerprint: "second".into(),
                bytes: Arc::from([4_u8, 5, 6]),
            },
        );
        let origins = HashSet::from([first, second]);

        assert_eq!(
            index.units_for_origins(&origins, 1, usize::MAX),
            Err(CallableArtifactSelectionError::Limit {
                field: "callable_artifact.units",
                limit: 1,
            })
        );
        assert_eq!(
            index.units_for_origins(&origins, 2, 5),
            Err(CallableArtifactSelectionError::Limit {
                field: "callable_artifact.unit_bytes",
                limit: 5,
            })
        );
        let selected = index
            .units_for_origins(&origins, 2, 6)
            .expect("bounded unit selection");
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&vec![1, 2, 3]));
        assert!(selected.contains(&vec![4, 5, 6]));
    }

    #[test]
    fn active_generation_keeps_unescaped_artifact_until_rollback_decides_retention() {
        let current = origin("dynamic", 0);
        let mut artifacts = CallableArtifactIndex::default();
        artifacts.units.insert(
            current,
            IndexedCallableUnit {
                fingerprint: "selected".into(),
                bytes: Arc::from([7_u8]),
            },
        );
        let empty = runmat_vm::FunctionRegistry::default();
        let active = FunctionRegistryGeneration::active(
            empty.clone(),
            artifacts,
            Rc::new(Vec::new()),
            Rc::new(Vec::new()),
            &empty,
        );
        assert!(active.artifacts().unwrap().units.contains_key(&current));
        let restore = FunctionRegistryGeneration::committed(empty);
        active.retain_escaped_into(&restore);
        assert!(!restore.artifacts().unwrap().units.contains_key(&current));
    }
}
