use std::cell::OnceCell;
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::Arc;

use runmat_execution::{Digest, FunctionBindingOrigin, ProgramRevision};
use runmat_hir::FunctionId;
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::source_context::SelectedProgramSourceOwner;
use runmat_types::ProgramSourceId;
use runmat_vm::FunctionRegistry;

use super::SelectedStaticTarget;
use crate::executable::ExecutableUnit;

#[path = "selected_graph/selection.rs"]
mod selection;
use selection::select_local_body;

#[path = "selected_graph/function_bindings.rs"]
mod function_bindings;
pub(crate) use function_bindings::bind_function_resolvers;
#[path = "selected_graph/snapshot.rs"]
mod snapshot;
#[path = "selected_graph/product.rs"]
mod product;
pub(in crate::session) use snapshot::SelectedGraphSnapshot;

/// Immutable selected code and its already-issued source authority. This
/// contains no RuntimeContext or execution service, and serializes V3 only if
/// an accepted graph task needs it. The current callable-artifact product can
/// seed the same cell when publication already produced that exact envelope.
pub(in crate::session) struct SelectedGraphPrimary {
    unit: Option<Rc<ExecutableUnit>>,
    revision: Option<ProgramRevision>,
    expected_program: Digest,
    retained_origin: Option<FunctionBindingOrigin>,
    retained_function: Option<FunctionId>,
    source_owner: SelectedProgramSourceOwner,
    selected_registry: Rc<FunctionRegistry>,
    current_functions: HashSet<FunctionId>,
    primary_source: Option<ProgramSourceId>,
    interop: runmat_types::InteropManifest,
    interop_receipt: Option<std::rc::Rc<runmat_vm::SelectedUnitInteropReceipt>>,
    static_targets: Vec<SelectedStaticTarget>,
    canonical: OnceCell<Result<Arc<[u8]>, String>>,
}

/// The compiler decides which exact current definitions are execution-only.
/// The durable registry must contain every other selected definition; it is
/// never replaced by the active request-local registry for that proof.
pub(in crate::session) struct SelectedGraphPublication<'a> {
    durable: &'a FunctionRegistry,
    execution_only: &'a HashSet<FunctionId>,
}

impl<'a> SelectedGraphPublication<'a> {
    pub(in crate::session) fn new(
        durable: &'a FunctionRegistry,
        execution_only: &'a HashSet<FunctionId>,
    ) -> Self {
        Self {
            durable,
            execution_only,
        }
    }
}

/// Exact selected source and optional already-published V3 bytes. The latter
/// are checked against the static dependencies before becoming primary code.
pub(in crate::session) struct SelectedGraphSource {
    pub unit: Rc<ExecutableUnit>,
    pub primary_source: Option<ProgramSourceId>,
    pub code: SelectedGraphCodeSource,
}

/// Exactly one immutable code state; no invocation authority can be cached here.
pub(in crate::session) enum SelectedGraphCodeSource {
    Unpublished,
    PublishedBytes(Arc<[u8]>),
    Admitted(Rc<runmat_vm::SelectedExecutableCode>),
}

/// Definitions and dependencies chosen by the same compiler publication.
pub(in crate::session) struct SelectedGraphSelection {
    pub current_functions: HashSet<FunctionId>,
    pub interop: runmat_types::InteropManifest,
    pub static_targets: Vec<SelectedStaticTarget>,
}

impl SelectedGraphPrimary {
    pub(in crate::session) fn new(
        source: SelectedGraphSource,
        runtime: &RuntimeContext,
        mut selected_registry: FunctionRegistry,
        publication: SelectedGraphPublication<'_>,
        selection: SelectedGraphSelection,
    ) -> Result<Self, String> {
        let SelectedGraphSource {
            unit,
            primary_source,
            code,
        } = source;
        let (already_published, already_admitted_code) = match code {
            SelectedGraphCodeSource::Unpublished => (None, None),
            SelectedGraphCodeSource::PublishedBytes(bytes) => (Some(bytes), None),
            SelectedGraphCodeSource::Admitted(code) => (None, Some(code)),
        };
        let SelectedGraphSelection {
            current_functions,
            interop,
            static_targets,
        } = selection;
        let program = unit
            .revision()
            .program_revision
            .identity_digest()
            .map_err(|error| error.to_string())?;
        let selected_ids = current_functions
            .iter()
            .filter(|id| {
                selected_registry
                    .get_current(**id)
                    .and_then(|body| body.binding_origin)
                    .is_some_and(|origin| origin.program == program)
            })
            .copied()
            .collect::<Vec<_>>();
        let durable = selected_ids
            .iter()
            .map(|id| {
                let selected = selected_registry
                    .get_current(*id)
                    .ok_or("selected graph definition is absent from its executable")?;
                if publication.execution_only.contains(id) {
                    if publication.durable.get_current(*id).is_some() {
                        return Err("execution-only graph definition was named-published");
                    }
                    return Ok(None);
                }
                let body = publication
                    .durable
                    .get_current(*id)
                    .ok_or("selected graph definition was not durably published")?;
                if body.binding_origin != selected.binding_origin {
                    return Err("published graph definition differs from selected product");
                }
                Ok(Some(body))
            })
            .collect::<Result<Vec<_>, &'static str>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        // Definition publication may replace the request-local SourceIds with
        // durable aliases. Every selected body must carry the same validated
        // post-publication owner; a script-only unit keeps its original issued
        // owner instead. Neither branch binds a second source map.
        let source_owner = if let Some(first) = durable.first() {
            let owner = first
                .selected_source_owner
                .as_ref()
                .ok_or("published graph definition has no selected source owner")?;
            if durable.iter().any(|body| {
                body.selected_source_owner
                    .as_ref()
                    .is_none_or(|candidate| !candidate.same_issuance(owner))
            }) {
                return Err("published graph definitions have mixed source owners".into());
            }
            owner.clone()
        } else {
            unit.selected_source_owner(runtime)?
        };
        if !selected_ids.is_empty() {
            selected_registry
                .attach_selected_program_source_owner(selected_ids, program, &source_owner)
                .map_err(str::to_string)?;
        }
        if let Some(bytes) = already_published.as_ref() {
            let admitted =
                runmat_vm::admit_executable_unit(bytes).map_err(|error| error.to_string())?;
            let actual = admitted
                .envelope()
                .manifest
                .optional_sections
                .iter()
                .find(|section| {
                    section.name == runmat_execution::executable::STATIC_IMPORT_SECTION
                });
            let expected = static_import_section_for(&static_targets)?;
            if actual != expected.as_ref() {
                return Err("published selected product changes its static imports".into());
            }
        }
        if let Some(code) = already_admitted_code.as_ref() {
            if code.program_digest() != program || code.source_map() != source_owner.portable_map()
            {
                return Err("cached selected code changes its program/source facts".into());
            }
            let actual = if code.static_imports().is_empty() {
                None
            } else {
                Some(
                    runmat_execution::executable::static_import_section(
                        code.static_imports().to_vec(),
                    )
                    .map_err(|error| error.to_string())?,
                )
            };
            if actual != static_import_section_for(&static_targets)? {
                return Err("cached selected code changes its static imports".into());
            }
        }
        let canonical = OnceCell::new();
        let published = already_published
            .or_else(|| already_admitted_code.map(|code| Arc::clone(code.canonical_v3())));
        if let Some(bytes) = published {
            let _ = canonical.set(Ok(bytes));
        }
        Ok(Self {
            revision: Some(unit.revision().program_revision.clone()),
            expected_program: program,
            retained_origin: None,
            retained_function: None,
            unit: Some(unit),
            source_owner,
            selected_registry: Rc::new(selected_registry),
            current_functions,
            primary_source,
            interop,
            interop_receipt: None,
            static_targets,
            canonical,
        })
    }

    pub(in crate::session) fn source_owner(&self) -> &SelectedProgramSourceOwner {
        &self.source_owner
    }

    pub(super) fn revision(&self) -> Option<&ProgramRevision> {
        self.revision.as_ref()
    }

    pub(super) fn expected_program(&self) -> Digest {
        self.expected_program
    }

    pub(super) fn retained_origin(&self) -> Option<FunctionBindingOrigin> {
        self.retained_origin
    }

    pub(super) fn retained_function(&self) -> Option<FunctionId> {
        self.retained_function
    }

    pub(super) fn selected_registry(&self) -> &FunctionRegistry {
        &self.selected_registry
    }

    pub(in crate::session) fn static_targets(&self) -> &[SelectedStaticTarget] {
        &self.static_targets
    }

    pub(in crate::session) fn selected_interop_receipt(
        &self,
    ) -> Option<&std::rc::Rc<runmat_vm::SelectedUnitInteropReceipt>> {
        self.interop_receipt.as_ref()
    }

    pub(in crate::session) fn with_selected_interop_receipt(
        mut self,
        receipt: Option<std::rc::Rc<runmat_vm::SelectedUnitInteropReceipt>>,
    ) -> Result<Self, String> {
        if receipt.is_none()
            && self.interop == runmat_types::InteropManifest::empty()
            && self.canonical.get().is_none()
            && self
                .unit
                .as_ref()
                .is_some_and(|unit| !unit.requires_python_runtime())
        {
            return Ok(self);
        }
        let canonical = self.canonical_v3()?;
        if receipt
            .as_ref()
            .is_some_and(|receipt| !receipt.matches_selected_unit(&canonical, &self.source_owner))
        {
            return Err("selected primary foreign receipt differs from code/source owner".into());
        }
        let admitted =
            runmat_vm::admit_executable_unit(&canonical).map_err(|error| error.to_string())?;
        let empty = admitted.envelope().manifest.interop == runmat_types::InteropManifest::empty();
        if empty == receipt.is_some() {
            return Err("selected primary foreign receipt presence differs from V3 interop".into());
        }
        self.interop_receipt = receipt;
        Ok(self)
    }

    /// Two selected invocations may share one stateful foreign host only when
    /// their immutable executable view and original source issuance are the
    /// same. A later name publication or a different static dependency graph
    /// cannot borrow an older host merely because its V3 bytes match.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn same_foreign_callback_basis(&self, other: &Self) -> Result<bool, String> {
        Ok(self.expected_program == other.expected_program
            && self.revision == other.revision
            && self.retained_origin == other.retained_origin
            && self.retained_function == other.retained_function
            && self.current_functions == other.current_functions
            && self.primary_source == other.primary_source
            && self.interop == other.interop
            && same_static_targets(&self.static_targets, &other.static_targets)
            && match (&self.interop_receipt, &other.interop_receipt) {
                (None, None) => true,
                (Some(left), Some(right)) => left.same_selected_admission(right),
                _ => false,
            }
            && self.source_owner.same_issuance(&other.source_owner)
            && self
                .selected_registry
                .same_immutable_execution_view(&other.selected_registry)
            && self.canonical_v3()?.as_ref() == other.canonical_v3()?.as_ref())
    }

    /// Match the ordinary primary's published bodies against its lazy V3
    /// product one body at a time. Origin-only maps cannot represent a second
    /// equal-byte reimport with a different selected source issuance.
    pub(super) fn current_body_fingerprints(
        &self,
    ) -> Result<std::collections::HashMap<FunctionBindingOrigin, String>, String> {
        let mut fingerprints = std::collections::HashMap::new();
        let mut selected_functions = Vec::new();
        for function in &self.current_functions {
            let Some(body) = self.selected_registry.get_current(*function) else {
                continue;
            };
            let Some(origin) = body.binding_origin else {
                continue;
            };
            if origin.program != self.expected_program {
                continue;
            }
            if !body
                .selected_source_owner
                .as_ref()
                .is_some_and(|owner| owner.same_issuance(&self.source_owner))
            {
                return Err("current selected body has a different source issuance".into());
            }
            selected_functions.push((*function, origin));
        }
        let selected_ids = selected_functions
            .iter()
            .map(|(function, _)| *function)
            .collect::<Vec<_>>();
        let product = selected_ids
            .first()
            .and_then(|function| self.selected_registry.get_current(*function))
            .and_then(|body| body.selected_executable_product.as_ref());
        let imports = if selected_ids.is_empty() {
            Vec::new()
        } else {
            let product = product.ok_or("current selected body has no exact code product")?;
            if selected_ids.iter().any(|function| {
                self.selected_registry
                    .get_current(*function)
                    .and_then(|body| body.selected_executable_product.as_ref())
                    .is_none_or(|candidate| !Rc::ptr_eq(candidate, product))
            }) {
                return Err("current selected bodies do not share one exact code product".into());
            }
            product.fingerprint_imports()
        };
        let mut selected_fingerprints = self
            .selected_registry
            .selected_runtime_body_fingerprints_with_imports(&selected_ids, &imports)
            .map_err(|error| error.to_string())?;
        for (function, origin) in selected_functions {
            let (checked_origin, fingerprint) = selected_fingerprints
                .remove(&function)
                .ok_or("current selected V3 body has no normalized fingerprint")?;
            if checked_origin != origin || fingerprints.insert(origin, fingerprint).is_some() {
                return Err("current V3 selection repeats or changes one portable body".into());
            }
        }
        Ok(fingerprints)
    }

    /// A retained bound callback selects the exact indexed V3 bytes and
    /// definition/source owner that issued its handle. No current entrypoint
    /// or later name registry can replace either half of this pair.
    pub(super) fn retained(
        bytes: Arc<[u8]>,
        source_owner: SelectedProgramSourceOwner,
        selected_registry: FunctionRegistry,
        function: FunctionId,
        origin: runmat_execution::FunctionBindingOrigin,
    ) -> Result<Self, String> {
        let selected_body = selected_registry.get(function);
        if !source_owner.is_live()
            || !selected_body.is_some_and(|body| {
                body.binding_origin == Some(origin)
                    && body
                        .selected_executable_product
                        .as_ref()
                        .is_some_and(|product| {
                            product.source_owner().same_issuance(&source_owner)
                                && product.full_digest() == Digest::sha256(bytes.as_ref())
                                && product.body_fingerprint(origin).is_some()
                        })
                    && body
                        .selected_source_owner
                        .as_ref()
                        .is_some_and(|owner| owner.same_issuance(&source_owner))
                    && body
                        .source_id
                        .and_then(runmat_vm::BytecodeSourceRef::runtime)
                        .is_some_and(|source| source_owner.owns_source(source))
            })
        {
            return Err("retained graph callable lacks its exact live source owner".into());
        }
        let admitted =
            runmat_vm::admit_executable_unit(&bytes).map_err(|error| error.to_string())?;
        let interop = admitted.envelope().manifest.interop.clone();
        let interop_receipt = selected_body
            .and_then(|body| body.selected_executable_product.as_ref())
            .and_then(|product| product.interop_receipt())
            .cloned();
        if (interop == runmat_types::InteropManifest::empty()) == interop_receipt.is_some()
            || interop_receipt
                .as_ref()
                .is_some_and(|receipt| !receipt.matches_selected_unit(&bytes, &source_owner))
        {
            return Err("retained graph foreign receipt differs from its selected unit".into());
        }
        let canonical = OnceCell::new();
        let _ = canonical.set(Ok(bytes));
        Ok(Self {
            unit: None,
            revision: None,
            expected_program: origin.program,
            retained_origin: Some(origin),
            retained_function: Some(function),
            source_owner,
            selected_registry: Rc::new(selected_registry),
            current_functions: HashSet::new(),
            primary_source: None,
            interop,
            interop_receipt,
            static_targets: Vec::new(),
            canonical,
        })
    }

    #[tracing::instrument(
        name = "runtime.executable.canonical_graph", skip_all,
        fields(cached = self.canonical.get().is_some())
    )]
    pub(in crate::session) fn canonical_v3(&self) -> Result<Arc<[u8]>, String> {
        self.canonical
            .get_or_init(|| {
                let unit = self
                    .unit
                    .as_ref()
                    .ok_or_else(|| "retained graph product has no indexed V3 bytes".to_string())?;
                let envelope = unit.portable_primary_source_envelope_with_static_imports(
                    &self.current_functions,
                    self.primary_source,
                    self.interop.clone(),
                    self.static_targets
                        .iter()
                        .map(|target| target.import)
                        .collect(),
                )?;
                envelope
                    .canonical_bytes()
                    .map(Arc::<[u8]>::from)
                    .map_err(|error| error.to_string())
            })
            .clone()
    }
}

fn static_import_section_for(
    targets: &[SelectedStaticTarget],
) -> Result<Option<runmat_execution::executable::ExecutableOptionalSection>, String> {
    if targets.is_empty() {
        return Ok(None);
    }
    runmat_execution::executable::static_import_section(
        targets.iter().map(|target| target.import).collect(),
    )
    .map(Some)
    .map_err(|error| error.to_string())
}

#[cfg(not(target_arch = "wasm32"))]
fn same_static_targets(left: &[SelectedStaticTarget], right: &[SelectedStaticTarget]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.import == right.import
                && left.function == right.function
                && left.definition == right.definition
                && left.binding == right.binding
                && Rc::ptr_eq(&left.product, &right.product)
                && left.source_owner.same_issuance(&right.source_owner)
        })
}
