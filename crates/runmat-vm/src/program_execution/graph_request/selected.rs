use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use runmat_execution::value::{
    GraphBoundOwner, GraphLexicalLayout, GraphProgramUnitId, GraphStaticCaller,
};
use runmat_execution::{ExecutableSourceMapV2, LanguageCompatibilityMode, ProgramRevision};
use runmat_execution_artifact::{
    archive::ArchiveLimits, AdmittedGraphUnitBundle, ArchiveAdmittedGraphExecutionRequest,
};
use runmat_runtime::context::{
    RuntimeContext, RuntimeForeignService, RuntimeLanguageMode, RuntimeLanguagePolicy,
};
use runmat_runtime::execution::value_codec::{
    GraphCallableProof, GraphSourceIssuer, GraphSourceUnitFacts,
};
use runmat_value::BoundFunctionAuthority;
use runmat_value::BoundFunctionBinding;

use crate::program_execution::unit_admission::RetainedExecutableUnit;
use crate::program_execution::{
    admit_executable_unit, selected_graph_callable_facts_for_owner, RuntimeBoundExecutableUnit,
};
#[path = "selected/admission.rs"]
mod admission;
#[path = "selected/code.rs"]
mod code;
pub(super) use code::RetainedSelectedGraph;

#[path = "selected/object_classes.rs"]
mod object_classes;
#[path = "selected/static_capture.rs"]
mod static_capture;

/// The wire order is the source and callable namespace. Each entry is bound
/// separately; the general retained-unit staging transaction coalesces equal
/// origins and is therefore not a valid V8 authority owner.
pub(super) struct SelectedWorkerGraph {
    code: Rc<RetainedSelectedGraph>,
    pub(super) unit_runtimes: Vec<RuntimeContext>,
}
impl std::ops::Deref for SelectedWorkerGraph {
    type Target = RetainedSelectedGraph;
    fn deref(&self) -> &Self::Target {
        &self.code
    }
}

fn selected_unit_runtime(
    base: &RuntimeContext,
    selected_foreign_ports: Option<&[Option<Rc<dyn RuntimeForeignService>>]>,
    position: usize,
    interop: &runmat_types::InteropManifest,
    compatibility_mode: LanguageCompatibilityMode,
) -> Result<RuntimeContext, String> {
    let policy = RuntimeLanguagePolicy {
        language_mode: match compatibility_mode {
            LanguageCompatibilityMode::RunMat => RuntimeLanguageMode::RunMat,
            LanguageCompatibilityMode::Matlab => RuntimeLanguageMode::Matlab,
            LanguageCompatibilityMode::Strict => RuntimeLanguageMode::Strict,
        },
        extensions_enabled: matches!(compatibility_mode, LanguageCompatibilityMode::RunMat),
        top_level_await_enabled: base.top_level_await_enabled(),
        dynamic_eval_enabled: base.dynamic_eval_enabled(),
    };
    let selected = base.clone().with_language_policy(policy);
    let has_interop = !interop.foreign_types.is_empty()
        || !interop.adapters.is_empty()
        || !interop.adapter_contracts.is_empty();
    let Some(ports) = selected_foreign_ports else {
        if has_interop {
            return Err("selected graph interop requires a unit-owned foreign service".into());
        }
        let services = selected.service_ports().clone().without_foreign();
        return Ok(selected.with_service_ports(services));
    };
    let port = ports
        .get(position)
        .ok_or("selected graph foreign port is absent for V3 unit")?;
    if has_interop && port.is_none() {
        return Err("selected graph V3 interop has no unit-owned foreign service".into());
    }
    if !has_interop && port.is_some() {
        return Err("selected graph V3 unit has an undeclared foreign service".into());
    }
    let services = match port {
        Some(port) => selected
            .service_ports()
            .clone()
            .with_foreign(Rc::clone(port)),
        None => selected.service_ports().clone().without_foreign(),
    };
    Ok(selected.with_service_ports(services))
}

fn validate_selected_port_count(
    retained_units: usize,
    selected_foreign_ports: Option<&[Option<Rc<dyn RuntimeForeignService>>]>,
) -> Result<(), String> {
    if selected_foreign_ports.is_some_and(|ports| ports.len() != retained_units + 1) {
        return Err("selected graph foreign-port count differs from V3 unit count".into());
    }
    Ok(())
}

pub(super) fn selected_bound_body<'a>(
    registry: &'a crate::FunctionRegistry,
    binding: &BoundFunctionBinding,
) -> Result<&'a crate::FunctionBytecode, String> {
    let function = runmat_hir::FunctionId(binding.function);
    let lease = binding
        .authority
        .local_lease(binding.function)
        .ok_or("selected graph binding has no exact local definition lease")?;
    registry
        .owned_view(function, &binding.authority)
        .ok_or("selected graph binding has no complete owned definition view")?;
    registry
        .get_owned(function, lease)
        .ok_or_else(|| "selected graph lease has no admitted body".into())
}

#[cfg(test)]
#[path = "selected/foreign_tests.rs"]
mod foreign_tests;

impl SelectedWorkerGraph {
    pub(super) fn active_foreign_for_unit(
        &self,
        unit: GraphProgramUnitId,
    ) -> Result<Option<&Rc<dyn RuntimeForeignService>>, String> {
        self.unit_runtimes
            .get(usize::from(unit.0))
            .map(|runtime| runtime.service_ports().foreign())
            .ok_or_else(|| "selected graph runtime unit is absent".into())
    }

    pub(super) fn into_code(self) -> Rc<RetainedSelectedGraph> {
        self.code
    }

    pub(super) fn runtime_for_unit(
        &self,
        unit: GraphProgramUnitId,
    ) -> Result<RuntimeContext, String> {
        self.unit_runtimes
            .get(usize::from(unit.0))
            .cloned()
            .ok_or_else(|| "graph selected runtime unit is absent".into())
    }

    pub(super) fn admit(
        admitted_request: &ArchiveAdmittedGraphExecutionRequest,
        bundles: Vec<AdmittedGraphUnitBundle>,
        runtime: &RuntimeContext,
        selected_foreign_ports: Option<&[Option<Rc<dyn RuntimeForeignService>>]>,
    ) -> Result<Self, String> {
        Self::admit_selection(
            admission::Admission::Worker(admitted_request),
            bundles,
            runtime,
            selected_foreign_ports,
        )
    }

    pub(super) fn admit_captured(
        admitted: &runmat_execution_artifact::ArchiveAdmittedCapturedGraphCode,
        bundles: Vec<AdmittedGraphUnitBundle>,
        runtime: &RuntimeContext,
        selected_foreign_ports: Option<&[Option<Rc<dyn RuntimeForeignService>>]>,
    ) -> Result<Self, String> {
        Self::admit_selection(
            admission::Admission::Captured(admitted),
            bundles,
            runtime,
            selected_foreign_ports,
        )
    }

    fn admit_selection(
        admission: admission::Admission<'_>,
        bundles: Vec<AdmittedGraphUnitBundle>,
        runtime: &RuntimeContext,
        selected_foreign_ports: Option<&[Option<Rc<dyn RuntimeForeignService>>]>,
    ) -> Result<Self, String> {
        let request = admission.view();
        validate_selected_port_count(request.retained_units.len(), selected_foreign_ports)?;
        if bundles.len() != request.retained_units.len() + 1 {
            return Err("selected graph bundle count differs from V3 unit count".into());
        }
        let bundles = bundles
            .into_iter()
            .enumerate()
            .map(|(position, ticket)| admission.consume_bundle(position, ticket))
            .collect::<Result<Vec<_>, _>>()?;
        let expected_map = admission.source_map()?;
        let mut units = Vec::with_capacity(request.retained_units.len() + 1);
        let mut receipts = Vec::with_capacity(units.capacity());
        let mut facts = Vec::with_capacity(units.capacity());
        let mut body_digests = Vec::with_capacity(units.capacity());
        let mut revisions = Vec::<ProgramRevision>::with_capacity(units.capacity());
        for (position, (bytes, bundle)) in std::iter::once(request.primary_unit)
            .chain(request.retained_units.iter().map(Vec::as_slice))
            .zip(bundles)
            .enumerate()
        {
            let admitted = admit_executable_unit(bytes).map_err(|error| error.worker_message())?;
            body_digests.push(
                admitted
                    .admitted_semantic_body_digests()
                    .map_err(|error| error.to_string())?,
            );
            let revision = admitted.envelope().manifest.identity.program.clone();
            revision
                .validate_current_compiler()
                .map_err(|error| error.to_string())?;
            let admitted_map = Arc::new(admitted.admitted_source_map().clone());
            let selected_runtime = selected_unit_runtime(
                runtime,
                selected_foreign_ports,
                position,
                &admitted.envelope().manifest.interop,
                revision.environment().compatibility_mode,
            )?;
            let (bound, receipt) = admitted
                .bind_runtime_sources_with_bundle(
                    selected_runtime,
                    bundle,
                    ArchiveLimits::default(),
                )
                .map_err(|error| error.worker_message())?;
            bound
                .validate_runtime_owner(runtime)
                .map_err(str::to_owned)?;
            let owner = bound
                .selected_source_owner()
                .ok_or("admitted graph unit has no exact issued source owner")?
                .clone();
            let selected = GraphSourceUnitFacts::capture_owner(owner, &revision, admitted_map)
                .map_err(|error| error.to_string())?;
            if facts
                .iter()
                .any(|prior: &GraphSourceUnitFacts| prior.same_selected_issuance(&selected))
            {
                return Err("graph repeats an exact selected source issuance".into());
            }
            units.push(bound);
            receipts.push(receipt);
            facts.push(selected);
            revisions.push(revision);
        }
        if request
            .primary_revision
            .is_some_and(|revision| revisions.first() != Some(revision))
        {
            return Err("primary graph V3 revision differs from its recipe".into());
        }
        let (issuer, source_map) =
            GraphSourceIssuer::capture_selected_units(&facts).map_err(|error| error.to_string())?;
        if source_map != expected_map {
            return Err("graph source projection differs from admitted V3 unit order".into());
        }
        let owners = units
            .iter()
            .enumerate()
            .map(|(index, unit)| {
                Ok((
                    GraphProgramUnitId(
                        u16::try_from(index)
                            .map_err(|_| "graph unit index exceeds its ABI".to_string())?,
                    ),
                    unit.selected_source_owner()
                        .ok_or("admitted graph source owner is absent")?
                        .clone(),
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        static_capture::attach(&mut units, &receipts, request.static_dependencies)?;
        let mut layouts = HashMap::new();
        let mut proofs = HashMap::new();
        let mut bindings = HashMap::new();
        for (index, unit) in units.iter().enumerate() {
            let (position, owner) = &owners[index];
            let selected = selected_graph_callable_facts_for_owner(
                unit.registry(),
                *position,
                owner,
                &owners,
            )?;
            let (unit_layouts, unit_proofs, unit_bindings) = selected.into_parts();
            for (key, value) in unit_layouts {
                if layouts.insert(key, value).is_some() {
                    return Err("graph repeats an exact unit-qualified lexical layout".into());
                }
            }
            for (key, value) in unit_proofs {
                if proofs.insert(key, value).is_some() {
                    return Err("graph repeats an exact unit-qualified callable proof".into());
                }
            }
            for (key, value) in unit_bindings {
                if bindings.insert(key, value).is_some() {
                    return Err("graph repeats an exact unit-qualified definition".into());
                }
            }
        }
        let mut static_bindings = HashMap::new();
        let mut sealed_sites = HashSet::new();
        for (index, unit) in units.iter().enumerate() {
            let selected_unit = GraphProgramUnitId(
                u16::try_from(index).map_err(|_| "graph unit index exceeds its ABI")?,
            );
            for import in unit.static_imports() {
                let caller = match import.caller {
                    runmat_execution::executable::ExecutableStaticCaller::Script => {
                        GraphStaticCaller::Script(selected_unit)
                    }
                    runmat_execution::executable::ExecutableStaticCaller::Function(origin) => {
                        GraphStaticCaller::Function(GraphBoundOwner {
                            unit: selected_unit,
                            origin,
                        })
                    }
                };
                if !sealed_sites.insert((caller, import.site)) {
                    return Err("selected unit repeats a sealed static site".into());
                }
            }
        }
        for edge in request.static_dependencies {
            let target_index = usize::from(edge.target.unit.0);
            let target = units
                .get(target_index)
                .ok_or("selected static target unit is absent")?;
            let binding = bindings
                .get(&edge.target)
                .ok_or("selected static target has no exact admitted binding")?;
            let body = selected_bound_body(target.registry(), binding)?;
            if body.binding_origin != Some(edge.target.origin)
                || !body.selected_source_owner.as_ref().is_some_and(|issued| {
                    target
                        .selected_source_owner()
                        .is_some_and(|owner| issued.same_issuance(owner))
                })
                || body_digests[target_index].get(&edge.target.origin)
                    != Some(&edge.target_body_digest)
            {
                return Err("selected static target differs from its admitted body/source".into());
            }
            if !sealed_sites.contains(&(edge.caller, edge.site))
                || static_bindings
                    .insert((edge.caller, edge.site), binding.clone())
                    .is_some()
            {
                return Err("selected static edge differs from its sealed site".into());
            }
        }
        if static_bindings.len() != sealed_sites.len() {
            return Err("selected graph lacks an exact static target for a sealed site".into());
        }
        let static_bindings = Rc::new(static_bindings);
        let sealed_sites = Rc::new(sealed_sites);
        let unit_runtimes = units
            .iter()
            .enumerate()
            .map(|(index, unit)| {
                let unit_id = GraphProgramUnitId(
                    u16::try_from(index).map_err(|_| "graph unit index exceeds its ABI")?,
                );
                let bindings = Rc::clone(&static_bindings);
                let sealed = Rc::clone(&sealed_sites);
                let resolve = Rc::new(move |origin, site| {
                    let caller = match origin {
                        Some(origin) => GraphStaticCaller::Function(GraphBoundOwner {
                            unit: unit_id,
                            origin,
                        }),
                        None => GraphStaticCaller::Script(unit_id),
                    };
                    if sealed.contains(&(caller, site)) {
                        bindings
                            .get(&(caller, site))
                            .cloned()
                            .map(Some)
                            .ok_or("sealed static site has no selected target")
                    } else {
                        Ok(None)
                    }
                });
                Ok(unit
                    .runtime()
                    .clone()
                    .with_selected_static_site_resolver(resolve))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut grouped = HashMap::<GraphBoundOwner, Vec<usize>>::new();
        for edge in request.class_dependencies {
            grouped
                .entry(edge.caller)
                .or_default()
                .push(usize::from(edge.declaration_unit.0));
        }
        let mut class_method_units = HashMap::<GraphBoundOwner, Vec<usize>>::new();
        let mut class_method_targets = HashMap::new();
        for edge in request.class_method_dependencies {
            if edge.method.role != runmat_types::ClassMethodRole::Ordinary {
                return Err("selected class method edge has a nonordinary role".into());
            }
            let position = usize::from(edge.target.unit.0);
            let unit = units
                .get(position)
                .ok_or("selected class method unit is absent")?;
            if unit.canonical_v3().is_none_or(|bytes| {
                runmat_execution::Digest::sha256(bytes) != edge.target_unit_digest
            }) {
                return Err("selected class method unit differs from its exact bytes".into());
            }
            let binding = bindings
                .get(&edge.target)
                .ok_or("selected class method has no admitted binding")?;
            let body = selected_bound_body(unit.registry(), binding)?;
            if body.class_method_owner.as_ref() != Some(&edge.method)
                || body.binding_origin != Some(edge.target.origin)
                || body.selected_source_owner.as_ref().is_none_or(|issued| {
                    unit.selected_source_owner()
                        .is_none_or(|owner| !issued.same_issuance(owner))
                })
                || body_digests[position].get(&edge.target.origin) != Some(&edge.target_body_digest)
            {
                return Err("selected class method differs from its admitted body/source".into());
            }
            if class_method_targets
                .insert((edge.caller, edge.method.clone()), edge.target)
                .is_some()
            {
                return Err("selected constructor repeats a class method edge".into());
            }
            class_method_units
                .entry(edge.caller)
                .or_default()
                .push(position);
        }
        let mut class_products = HashMap::new();
        for (caller, declarations) in grouped {
            let binding = bindings
                .get(&caller)
                .ok_or("class dependency caller has no admitted callable binding")?;
            let caller_unit = units
                .get(usize::from(caller.unit.0))
                .ok_or("class dependency caller unit is absent")?;
            let body = selected_bound_body(caller_unit.registry(), binding)?;
            if body.binding_origin != Some(caller.origin) {
                return Err("class dependency caller differs from its exact body".into());
            }
            let owner = body
                .class_method_owner
                .as_ref()
                .filter(|owner| {
                    owner.role == runmat_types::ClassMethodRole::Constructor && !owner.is_static
                })
                .ok_or("class dependency caller is not a concrete constructor")?;
            let mut positions = Vec::with_capacity(declarations.len() + 1);
            positions.push(usize::from(caller.unit.0));
            positions.extend(declarations);
            positions.extend(
                class_method_units
                    .get(&caller)
                    .into_iter()
                    .flatten()
                    .copied(),
            );
            positions.sort_unstable();
            positions.dedup();
            let mut selected = Vec::with_capacity(positions.len());
            let mut selected_receipts = Vec::with_capacity(positions.len());
            for position in positions {
                selected.push(
                    units
                        .get(position)
                        .ok_or("class dependency selected unit is absent")?,
                );
                selected_receipts.push(
                    receipts
                        .get(position)
                        .ok_or("class dependency selected receipt is absent")?
                        .clone(),
                );
            }
            let product = crate::SelectedClassProduct::from_bound_units_with_receipts(
                &selected,
                &selected_receipts,
                &owner.declaring_class,
            )
            .map_err(|error| error.to_string())?;
            class_products.insert(caller, Rc::new(product));
        }
        let object_classes = object_classes::admit(
            request.graph,
            &units,
            &receipts,
            &bindings,
            &class_method_targets,
            &mut class_products,
        )?;
        let mut bound_schema = |owner: GraphBoundOwner| -> Result<runmat_execution::value::GraphRequiredSchema, String> {
            let binding = bindings
                .get(&owner)
                .ok_or("selected graph root has no exact binding")?;
            let unit = units
                .get(usize::from(owner.unit.0))
                .ok_or("selected graph root unit is absent")?;
            let body = selected_bound_body(unit.registry(), binding)?;
            if body.binding_origin != Some(owner.origin) {
                return Err("selected graph root differs from admitted body".into());
            }
            Ok(crate::selected_graph_code_required_schema(
                &body.instructions,
            ))
        };
        let required_schema = match request.roots {
            Some(roots) => crate::selected_graph_execution_required_schema(
                roots,
                request.graph,
                &mut bound_schema,
                || {
                    let primary = units.first().ok_or("selected graph has no primary unit")?;
                    Ok(crate::selected_graph_code_required_schema(
                        &primary.bytecode().instructions,
                    ))
                },
            )?,
            None => crate::graph_schema::selected_callable_capture_required_schema(
                request.graph,
                &mut bound_schema,
            )?,
        };
        if runmat_execution::value::GraphRequiredSchema::from_version(request.schema_version)
            .is_none_or(|admitted| admitted < required_schema)
        {
            return Err("selected graph executable requires a newer schema".into());
        }
        let units = units
            .into_iter()
            .map(RuntimeBoundExecutableUnit::into_retained_code)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            code: Rc::new(RetainedSelectedGraph {
                units,
                _receipts: receipts,
                issuer,
                source_map: Arc::new(source_map),
                layouts,
                proofs,
                bindings,
                class_products,
                object_classes,
                static_bindings,
                sealed_sites,
            }),
            unit_runtimes,
        })
    }
}
