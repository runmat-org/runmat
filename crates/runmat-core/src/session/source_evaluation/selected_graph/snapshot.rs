use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};
use std::sync::Arc;

use runmat_execution::value::{
    GraphBoundOwner, GraphClassDependency, GraphClassMethodDependency, GraphLexicalLayout,
    GraphNodeRef, GraphProgramUnitId, GraphStaticCaller, GraphStaticDependency, PayloadGraph,
    ValueLimits,
};
use runmat_execution::ExecutableSourceMapV2;
use runmat_runtime::execution::value_codec::{
    GraphCallableProof, GraphCaptureProof, GraphSourceIssuer, ValueCodecError,
};
use runmat_runtime::execution::{
    DeferredGraphCapture, DeferredGraphProgram, DeferredGraphSnapshot, DeferredInvocation,
    ExecutionServiceError, GraphResultReceipt, RootedValueSequence,
};
use runmat_runtime::source_context::SelectedProgramSourceOwner;
use runmat_runtime::RuntimeError;
use runmat_value::{BoundFunctionBinding, ValueSequence};

use super::selection::{invalid, SelectedGraphUnits};
use super::{select_local_body, SelectedGraphPrimary};
use crate::session::source_evaluation::SessionSourceEvaluator;

/// An invocation-local selector owns only immutable code/source facts and a
/// weak path to the canonical generation. A delayed Future can therefore see
/// a newly written handle cell at spawn without retaining a service cycle or
/// selecting by its display name.
pub(in crate::session) struct SelectedGraphSnapshot {
    primary: Rc<SelectedGraphPrimary>,
    evaluator: Weak<SessionSourceEvaluator>,
}

impl SelectedGraphSnapshot {
    pub(in crate::session) fn program(
        primary: Rc<SelectedGraphPrimary>,
        evaluator: &Rc<SessionSourceEvaluator>,
    ) -> DeferredGraphProgram {
        DeferredGraphProgram::new(Rc::new(Self {
            primary,
            evaluator: Rc::downgrade(evaluator),
        }))
    }
}

#[path = "snapshot/capture.rs"]
mod capture;
#[path = "snapshot/restored.rs"]
mod restored;

impl DeferredGraphSnapshot for SelectedGraphSnapshot {
    fn program_revision(&self) -> Option<&runmat_execution::ProgramRevision> {
        self.primary.revision()
    }

    fn select_retained_worker_callable(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
        selected: runmat_runtime::call::descriptor::RepeatableCallable,
    ) -> Result<runmat_runtime::execution::SelectedWorkerCallable, RuntimeError> {
        if !self.primary.source_owner().is_bound_in(runtime) {
            return Err(runmat_runtime::build_runtime_error(
                "retained worker selection requires its original source invocation",
            )
            .with_identifier("RunMat:Execution:SelectedCallableUnavailable")
            .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
            .build());
        }
        runmat_vm::select_retained_worker_callable(
            runtime,
            selected,
            &self.primary.selected_registry,
        )
    }
    fn restore_callables(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
        portable: runmat_execution::value::PortableSelectedCode,
    ) -> Result<runmat_runtime::execution::RestoredCallableGraph, RuntimeError> {
        self.restore_selected_callables(runtime, portable)
    }
    fn capture_callables(
        &self,
        selected: &[runmat_runtime::call::descriptor::CallableDescriptor],
        limits: ValueLimits,
        workspace: Option<&runmat_runtime::execution::value_codec::ValidationWorkspaceOwner>,
    ) -> Result<runmat_runtime::execution::DeferredSelectedCodeCapture, ValueCodecError> {
        let preliminary = GraphSourceIssuer::capture_owner(self.primary.source_owner());
        let prepared = runmat_runtime::execution::value_codec::prepare_callable_targets(
            selected,
            &preliminary,
            limits,
            workspace,
        )?;
        self.capture_prepared(prepared, None, limits)
    }

    fn select_repeatable_callable(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
        target: runmat_runtime::execution::RootedValue,
    ) -> Result<runmat_runtime::call::descriptor::RepeatableCallable, RuntimeError> {
        if !self.primary.source_owner().is_bound_in(runtime) {
            let mut error = runmat_runtime::runtime_error::semantic_error(
                "RunMat:Execution:SelectedCallableUnavailable",
                "local callback selection requires its original selected source invocation",
            );
            error.gpu_gather_retry = runmat_runtime::GpuGatherRetry::Never;
            return Err(error);
        }
        crate::callable_selection::select(runtime, &self.primary.selected_registry, target)
    }
    fn select_callable(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
        target: runmat_runtime::execution::RootedValue,
    ) -> Result<runmat_runtime::execution::SelectedWorkerCallable, RuntimeError> {
        runmat_vm::select_worker_callable(runtime, target, &self.primary.selected_registry)
    }

    fn snapshot(
        &self,
        invocation: &DeferredInvocation,
        limits: ValueLimits,
        workspace: Option<&runmat_runtime::execution::value_codec::ValidationWorkspaceOwner>,
    ) -> Result<DeferredGraphCapture, ValueCodecError> {
        let preliminary = GraphSourceIssuer::capture_owner(self.primary.source_owner());
        let (prepared, roots) =
            runmat_runtime::execution::value_codec::prepare_deferred_invocation_graph(
                invocation,
                &preliminary,
                limits,
                workspace,
            )?;
        self.capture_prepared(prepared, Some(&roots), limits)
            .map(|capture| capture.into_invocation(roots))
    }
}

fn append_class_method_edges(
    units: &SelectedGraphUnits,
    view: &runmat_vm::FunctionRegistry,
    caller: GraphBoundOwner,
    facts: &runmat_value::SelectedObjectClassFacts,
    edges: &mut HashMap<
        (GraphBoundOwner, runmat_types::ClassMethodOwner),
        GraphClassMethodDependency,
    >,
) -> Result<(), ValueCodecError> {
    for method in facts.owner_methods() {
        let Some(binding) = method.binding.as_ref() else {
            continue;
        };
        let lease = binding
            .authority
            .local_lease(binding.function)
            .ok_or_else(|| invalid("selected class method lacks its counted definition lease"))?;
        let body = view
            .get_owned(runmat_hir::FunctionId(binding.function), lease)
            .ok_or_else(|| invalid("selected class method lacks its exact bound body"))?;
        if body.class_method_owner.as_ref() != Some(&method.owner) {
            return Err(invalid(
                "selected class method differs from its declaration",
            ));
        }
        let origin = body
            .binding_origin
            .ok_or_else(|| invalid("selected class method has no portable origin"))?;
        let product = body
            .selected_executable_product
            .as_ref()
            .ok_or_else(|| invalid("selected class method lacks its selected product"))?;
        if body
            .selected_source_owner
            .as_ref()
            .is_none_or(|owner| !owner.same_issuance(product.source_owner()))
        {
            return Err(invalid("selected class method changed its source issuance"));
        }
        let target = GraphBoundOwner {
            unit: units.unit_for_product(product)?,
            origin,
        };
        let edge = GraphClassMethodDependency {
            caller,
            method: method.owner.clone(),
            target,
            target_unit_digest: product.full_digest(),
            target_body_digest: product
                .body_digest(origin)
                .ok_or_else(|| invalid("selected class method has no body digest"))?,
        };
        match edges.entry((caller, method.owner.clone())) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(edge);
            }
            std::collections::hash_map::Entry::Occupied(slot) if slot.get() == &edge => {}
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(invalid(
                    "selected class method has conflicting exact products",
                ));
            }
        }
    }
    Ok(())
}

fn append_captured_dependencies(
    units: &mut SelectedGraphUnits,
    artifacts: &crate::session::source_evaluation::generation::CallableArtifactIndex,
    root: &runmat_vm::SelectedStaticProductTarget,
    edges: &mut Vec<GraphStaticDependency>,
    sites: &mut HashMap<
        (
            GraphStaticCaller,
            runmat_execution::value::GraphStaticReferenceSite,
        ),
        GraphStaticDependency,
    >,
) -> Result<(), ValueCodecError> {
    append_product_dependencies(units, artifacts, root.dependencies(), edges, sites)
}

fn append_product_dependencies(
    units: &mut SelectedGraphUnits,
    artifacts: &crate::session::source_evaluation::generation::CallableArtifactIndex,
    dependencies: &[runmat_vm::SelectedStaticProductDependency],
    edges: &mut Vec<GraphStaticDependency>,
    sites: &mut HashMap<
        (
            GraphStaticCaller,
            runmat_execution::value::GraphStaticReferenceSite,
        ),
        GraphStaticDependency,
    >,
) -> Result<(), ValueCodecError> {
    let mut target_units = Vec::with_capacity(dependencies.len());
    for dependency in dependencies {
        let target = runmat_vm::SelectedStaticProductTarget::new(
            dependency.import(),
            Rc::clone(dependency.target()),
        )
        .map_err(invalid)?;
        let target_unit = units.add_static_product_target(artifacts, &target)?;
        target_units.push(target_unit);
    }
    for (dependency, target_unit) in dependencies.iter().zip(target_units) {
        let caller_unit = units.unit_for_product(dependency.caller())?;
        append_static_edge(
            dependency.import(),
            caller_unit,
            target_unit,
            dependency.target().full_digest(),
            edges,
            sites,
        )?;
    }
    Ok(())
}

fn append_static_edge(
    import: runmat_execution::executable::ExecutableStaticImport,
    caller_unit: GraphProgramUnitId,
    target_unit: GraphProgramUnitId,
    target_unit_digest: runmat_execution::Digest,
    edges: &mut Vec<GraphStaticDependency>,
    sites: &mut HashMap<
        (
            GraphStaticCaller,
            runmat_execution::value::GraphStaticReferenceSite,
        ),
        GraphStaticDependency,
    >,
) -> Result<(), ValueCodecError> {
    let caller = match import.caller {
        runmat_execution::executable::ExecutableStaticCaller::Script => {
            GraphStaticCaller::Script(caller_unit)
        }
        runmat_execution::executable::ExecutableStaticCaller::Function(origin) => {
            GraphStaticCaller::Function(GraphBoundOwner {
                unit: caller_unit,
                origin,
            })
        }
    };
    let edge = GraphStaticDependency {
        caller,
        site: import.site,
        target: GraphBoundOwner {
            unit: target_unit,
            origin: import.target,
        },
        target_unit_digest,
        target_body_digest: import.target_body,
    };
    match sites.get(&(caller, import.site)) {
        Some(prior) if prior == &edge => Ok(()),
        Some(_) => Err(invalid(
            "one static site selects conflicting target products",
        )),
        None => {
            sites.insert((caller, import.site), edge);
            edges.push(edge);
            Ok(())
        }
    }
}

type SelectedCallableFacts = (
    HashMap<GraphBoundOwner, GraphLexicalLayout>,
    HashMap<GraphBoundOwner, GraphCallableProof>,
    HashMap<GraphBoundOwner, BoundFunctionBinding>,
);

fn selected_callable_facts(
    units: &SelectedGraphUnits,
    selected_owners: &[(GraphProgramUnitId, SelectedProgramSourceOwner)],
    views: &[Rc<runmat_vm::FunctionRegistry>],
) -> Result<SelectedCallableFacts, ValueCodecError> {
    let mut layouts = HashMap::new();
    let mut proofs = HashMap::new();
    let mut bindings: HashMap<GraphBoundOwner, BoundFunctionBinding> = HashMap::new();
    for view in views {
        for (position, unit) in units.units().iter().enumerate() {
            let ordinal = GraphProgramUnitId(
                u16::try_from(position).map_err(|_| invalid("graph unit index exceeds u16"))?,
            );
            let selected = runmat_vm::selected_graph_callable_facts_for_owner(
                view,
                ordinal,
                &unit.source_owner,
                selected_owners,
            )
            .map_err(|_| invalid("selected callable facts are inconsistent"))?;
            let (new_layouts, new_proofs, new_bindings) = selected.into_parts();
            for (owner, binding) in new_bindings {
                if let Some(existing) = bindings.get(&owner) {
                    if existing != &binding {
                        return Err(invalid(
                            "one graph callable has conflicting definition leases",
                        ));
                    }
                } else {
                    bindings.insert(owner, binding);
                }
            }
            for (owner, layout) in new_layouts {
                if let Some(existing) = layouts.get(&owner) {
                    if existing != &layout {
                        return Err(invalid("one graph callable has conflicting lexical layout"));
                    }
                } else {
                    layouts.insert(owner, layout);
                }
            }
            for (owner, proof) in new_proofs {
                if let Some(existing) = proofs.get(&owner) {
                    if !same_proof(existing, &proof) {
                        return Err(invalid("one graph callable has conflicting source proof"));
                    }
                } else {
                    proofs.insert(owner, proof);
                }
            }
        }
    }
    Ok((layouts, proofs, bindings))
}

fn same_proof(left: &GraphCallableProof, right: &GraphCallableProof) -> bool {
    let captures = match (left.captures, right.captures) {
        (GraphCaptureProof::Exact(left), GraphCaptureProof::Exact(right)) => left == right,
        (
            GraphCaptureProof::NestedShared {
                vm_copied: left_copied,
                zero_copy: left_zero,
            },
            GraphCaptureProof::NestedShared {
                vm_copied: right_copied,
                zero_copy: right_zero,
            },
        ) => {
            left_copied == right_copied
                && std::mem::discriminant(&left_zero) == std::mem::discriminant(&right_zero)
        }
        (GraphCaptureProof::InstanceMethodReceiver, GraphCaptureProof::InstanceMethodReceiver) => {
            true
        }
        _ => false,
    };
    captures
        && left.qualified_name == right.qualified_name
        && left.source_role == right.source_role
        && left.issued_source_file == right.issued_source_file
        && left.source_file == right.source_file
}

struct SelectedGraphResult {
    source_owner: SelectedProgramSourceOwner,
    required_schema: runmat_execution::value::GraphRequiredSchema,
    selected_units: Vec<(SelectedProgramSourceOwner, Arc<[u8]>)>,
    selected_bundle_archives: Vec<Option<Arc<[u8]>>>,
    class_dependencies: Vec<GraphClassDependency>,
    class_method_dependencies: Vec<GraphClassMethodDependency>,
    static_dependencies: Vec<GraphStaticDependency>,
    issuer: GraphSourceIssuer,
    source_map: ExecutableSourceMapV2,
    layouts: HashMap<GraphBoundOwner, GraphLexicalLayout>,
    proofs: HashMap<GraphBoundOwner, GraphCallableProof>,
    bindings: HashMap<GraphBoundOwner, BoundFunctionBinding>,
    object_classes: HashMap<GraphBoundOwner, Rc<runmat_value::SelectedObjectClassFacts>>,
    limits: ValueLimits,
}

impl GraphResultReceipt for SelectedGraphResult {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn same_selected_admission(&self, other: &dyn GraphResultReceipt) -> bool {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return false;
        };
        self.source_owner.same_issuance(&other.source_owner)
            && self.required_schema == other.required_schema
            && self
                .source_owner
                .durable_aliases()
                .eq(other.source_owner.durable_aliases())
            && self.selected_units.len() == other.selected_units.len()
            && self.selected_bundle_archives == other.selected_bundle_archives
            && self.selected_units.iter().zip(&other.selected_units).all(
                |((left_owner, left_bytes), (right_owner, right_bytes))| {
                    left_owner.same_issuance(right_owner)
                        && left_owner
                            .durable_aliases()
                            .eq(right_owner.durable_aliases())
                        && (Arc::ptr_eq(left_bytes, right_bytes)
                            || left_bytes.as_ref() == right_bytes.as_ref())
                },
            )
            && self.class_dependencies == other.class_dependencies
            && self.class_method_dependencies == other.class_method_dependencies
            && self.static_dependencies == other.static_dependencies
            && self.source_map == other.source_map
            && self.layouts == other.layouts
            && self.bindings == other.bindings
            && self.object_classes == other.object_classes
            && self.proofs.len() == other.proofs.len()
            && self.proofs.iter().all(|(owner, proof)| {
                other
                    .proofs
                    .get(owner)
                    .is_some_and(|other_proof| same_proof(proof, other_proof))
            })
            && self.limits == other.limits
    }

    fn check_issuing_context(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<(), ExecutionServiceError> {
        if !self.source_owner.is_live() {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        if !self.source_owner.is_issued_in(runtime) {
            return Err(ExecutionServiceError::ForeignScope);
        }
        Ok(())
    }

    fn decode_in_context(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
        graph: &PayloadGraph,
    ) -> Result<RootedValueSequence, ExecutionServiceError> {
        self.check_issuing_context(runtime)?;
        if graph.required_schema() > self.required_schema {
            return Err(ExecutionServiceError::Failed(
                "selected graph response requires an unadmitted value schema".into(),
            ));
        }
        let values =
            runmat_runtime::execution::value_codec::decode_payload_graph_with_selected_classes_in_context(
                graph,
                runtime,
                &self.source_map,
                &self.issuer,
                &self.layouts,
                &self.proofs,
                &self.bindings,
                &self.object_classes,
                self.limits,
            )
            .map_err(|error| match error {
                ValueCodecError::Allocation { source, .. } => {
                    ExecutionServiceError::HostAllocation(source)
                }
                error => ExecutionServiceError::Failed(error.to_string()),
            })?;
        values.with_owned_output_storage(|values| {
            let sequence = ValueSequence::from_output_storage(values)
                .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
            RootedValueSequence::new(sequence)
        })
    }

    fn decode_failure_in_context(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
        failure: &runmat_execution::ProgramRuntimeFailure,
    ) -> Result<RuntimeError, ExecutionServiceError> {
        self.check_issuing_context(runtime)?;
        runmat_runtime::execution::decode_runtime_failure_with_source_issuer_in_context(
            failure,
            runtime,
            &self.issuer,
            &self.source_map,
        )
        .map_err(|error| {
            ExecutionServiceError::Infrastructure(format!(
                "selected graph failure violates its source contract: {error}"
            ))
        })
    }
}

fn generation_error(error: RuntimeError) -> ValueCodecError {
    ValueCodecError::Invalid {
        path: "$".into(),
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::{Digest, ExecutableSourceRecordV2};
    use runmat_runtime::source_context::{
        bind_portable_source_map_v2, lease_source_context_in_context,
    };
    use runmat_types::ProgramSourceId;

    fn runtime() -> runmat_runtime::context::RuntimeContext {
        runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
    }

    fn sample_receipt(
        source_owner: SelectedProgramSourceOwner,
        source_map: ExecutableSourceMapV2,
        bytes: &[u8],
    ) -> SelectedGraphResult {
        SelectedGraphResult {
            issuer: GraphSourceIssuer::capture_owner(&source_owner),
            required_schema: runmat_execution::value::GraphRequiredSchema::V10,
            selected_units: vec![(source_owner.clone(), Arc::from(bytes))],
            selected_bundle_archives: vec![None],
            class_dependencies: Vec::new(),
            class_method_dependencies: Vec::new(),
            static_dependencies: Vec::new(),
            source_owner,
            source_map,
            layouts: HashMap::new(),
            proofs: HashMap::new(),
            bindings: HashMap::new(),
            object_classes: HashMap::new(),
            limits: ValueLimits::default(),
        }
    }

    #[test]
    fn graph_result_receipt_rejects_foreign_fetch_before_decode() {
        let text = "value = 1;";
        let map = ExecutableSourceMapV2::new(vec![ExecutableSourceRecordV2 {
            source_id: ProgramSourceId(0),
            owner_identity: "fixture".into(),
            relative_path: "source.m".into(),
            content_revision: Digest::sha256(text.as_bytes()),
            display_name: "source.m".into(),
            text: text.into(),
        }])
        .expect("portable source map");
        let owning = bind_portable_source_map_v2(runtime(), &map).expect("issued source map");
        let source_owner = SelectedProgramSourceOwner::capture(&owning, &map).expect("exact owner");
        let receipt = sample_receipt(source_owner.clone(), map.clone(), b"selected V3");
        let identical = sample_receipt(source_owner, map.clone(), b"selected V3");
        let different_bytes =
            sample_receipt(receipt.source_owner.clone(), map.clone(), b"other V3");
        let issued = owning
            .program_source_ref(ProgramSourceId(0))
            .expect("issued source");
        let durable = lease_source_context_in_context(&owning, "source.m", None, text)
            .expect("durable source");
        let aliased = receipt
            .source_owner
            .with_durable_aliases(&owning, [(issued, durable.source_ref())])
            .expect("validated durable alias");
        let different_aliases = sample_receipt(aliased, map.clone(), b"selected V3");
        let foreign = bind_portable_source_map_v2(runtime(), &map).expect("foreign source map");
        let foreign_owner =
            SelectedProgramSourceOwner::capture(&foreign, &map).expect("foreign source owner");
        let different_issuance = sample_receipt(foreign_owner, map, b"selected V3");
        let mut different_foreign_archive = sample_receipt(
            receipt.source_owner.clone(),
            receipt.source_map.clone(),
            b"selected V3",
        );
        different_foreign_archive.selected_bundle_archives =
            vec![Some(Arc::from(b"foreign".as_slice()))];
        assert!(receipt.same_selected_admission(&identical));
        assert!(!receipt.same_selected_admission(&different_bytes));
        assert!(!receipt.same_selected_admission(&different_aliases));
        assert!(!receipt.same_selected_admission(&different_foreign_archive));
        assert!(!receipt.same_selected_admission(&different_issuance));
        assert_eq!(receipt.check_issuing_context(&owning), Ok(()));
        assert_eq!(
            receipt.check_issuing_context(&runtime()),
            Err(ExecutionServiceError::ForeignScope)
        );
        let invalid_failure = runmat_execution::ProgramRuntimeFailure {
            message: String::new(),
            identifier: None,
            span: None,
            builtin: None,
            task_id: None,
            call_frames: Vec::new(),
            call_frames_elided: 0,
            call_stack: Vec::new(),
            phase: None,
        };
        assert!(matches!(
            receipt.decode_failure_in_context(&runtime(), &invalid_failure),
            Err(ExecutionServiceError::ForeignScope)
        ));
        assert!(matches!(
            receipt.decode_failure_in_context(&owning, &invalid_failure),
            Err(ExecutionServiceError::Infrastructure(_))
        ));
    }
}
