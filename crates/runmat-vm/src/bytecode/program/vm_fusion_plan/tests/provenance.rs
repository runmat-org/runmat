use super::super::*;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::{
    CancellationReason, Digest, PlacementExecutionAuthorityRevision, PlacementInvalidation,
    PlacementRevision, ProgramEnvironment, ProgramRevision,
};
use runmat_runtime::context::{
    RuntimeContext, RuntimePlacementService, RuntimeServicePorts, RuntimeVmFusionGenerationPermit,
    VmFusionRegionRegistration, VmFusionSessionRegistration,
};
use runmat_runtime::execution::RuntimeExecutionService;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::{Arc, Weak};

struct CurrentPermit;

impl RuntimeVmFusionGenerationPermit for CurrentPermit {
    fn is_current(&self) -> bool {
        true
    }
}

#[derive(Default)]
struct RevisionService {
    invalidations: RefCell<Vec<PlacementInvalidation>>,
    permits: RefCell<BTreeMap<u64, Arc<CurrentPermit>>>,
}

impl RuntimePlacementService for RevisionService {
    fn plan(
        &self,
        _: runmat_execution::PlacementPlanRequest,
    ) -> Result<runmat_execution::PlacementDecision, runmat_runtime::RuntimeError> {
        unreachable!("VM fusion-plan provenance fixture provides revision identity only")
    }

    fn observe(
        &self,
        _: runmat_execution::PlacementFeedback,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        unreachable!("VM fusion-plan provenance fixture does not publish placement feedback")
    }

    fn invalidate(&self, invalidation: PlacementInvalidation) {
        self.invalidations.borrow_mut().push(invalidation);
    }

    fn whole_work_vm_fusion_region_session_revision(
        &self,
        program: Option<ProgramRevision>,
        region: VmFusionRegionRegistration,
        liveness: Weak<()>,
    ) -> Option<VmFusionSessionRegistration> {
        liveness.upgrade()?;
        let authority = Digest::sha256_serialized(&(
            "vm-plan-provenance-test-v2",
            region.semantic_identity().digest(),
        ))
        .ok()?;
        let revision = PlacementRevision {
            program,
            catalog: Digest::sha256(b"vm-plan-provenance-catalog"),
            compiler: Digest::sha256(b"vm-plan-provenance-compiler"),
            authority: PlacementExecutionAuthorityRevision::RuntimeBindingSession(authority),
            policy: Digest::sha256(b"vm-plan-provenance-policy"),
        };
        let permit = self
            .permits
            .borrow_mut()
            .entry(region.installation())
            .or_insert_with(|| Arc::new(CurrentPermit))
            .clone();
        Some(VmFusionSessionRegistration::new(revision, permit))
    }
}

fn program(label: &[u8]) -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256(label),
        Digest::sha256(b"vm-plan-provenance-source"),
        ProgramEnvironment::new(
            2,
            2,
            Digest::sha256(b"vm-plan-provenance-runtime"),
            Digest::sha256(b"vm-plan-provenance-builtins"),
            "runmat",
        )
        .unwrap(),
    )
    .unwrap()
}

fn runtime_context(
    cancellation: &CancellationSource,
    selected_program: ProgramRevision,
    service: Rc<RevisionService>,
) -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_placement(service))
        .with_program_revision(Some(selected_program))
        .with_cancellation_token(cancellation.token())
}

fn compiled_reduction() -> Bytecode {
    let ast = runmat_parser::parse("x = single([1, 2; 3, 4]); y = sum(x, 1);")
        .expect("parse reduction fixture");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty())
        .expect("lower reduction fixture");
    let mir =
        runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower reduction fixture MIR");
    crate::bytecode::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id)
        .expect("compile reduction fixture")
}

#[test]
fn installed_right_requires_retained_reconciled_plan_group_and_runtime_provenance() {
    let cancellation = CancellationSource::new();
    let selected_program = program(b"selected");
    let service = Rc::new(RevisionService::default());
    let runtime = runtime_context(&cancellation, selected_program.clone(), service.clone());
    let bytecode = compiled_reduction();
    assert!(
        !bytecode.resume_points.is_empty(),
        "authentic selected reduction must exercise typed resume-point identity"
    );
    assert!(vm_fusion_executable_identity(&bytecode).is_some());
    let (plan, graph, installation) = bytecode.prepare_vm_fusion_plan_with_installation(&runtime);
    let plan = plan.expect("compiled reduction must retain its reconciled plan");
    assert!(graph.is_some(), "compiled reduction must retain its graph");
    let installation = installation.expect("compiled plan must install exact groups");
    let selected = plan
        .groups
        .iter()
        .find(|group| group.group.kind.is_reduction() && group.kernel.supported)
        .expect("compiled reduction must retain one supported group")
        .clone();

    let exact = installation
        .right_for(&plan, &selected, &runtime)
        .expect("exact retained plan and group must issue a right");
    let decline = installation
        .decline_for(&plan, &selected, &runtime)
        .expect("same selected region can decline only before effects");
    assert!(decline.is_current(&runtime));
    assert!(decline.matches_group(
        selected.group.id,
        selected.group.span.start,
        selected.group.span.end
    ));
    assert!(exact.planning_revision(&runtime).is_some());
    let publication_pc = installation
        .publication_pc_for(&plan, &selected)
        .expect("compiled reduction output has a checked following store");
    assert_eq!(publication_pc, selected.group.span.end + 1);
    assert!(matches!(
        bytecode.instructions[publication_pc],
        Instr::StoreVar(_) | Instr::StoreLocal(_)
    ));

    let equal_but_foreign = Rc::new((*plan).clone());
    assert!(installation
        .publication_pc_for(&equal_but_foreign, &selected)
        .is_none());
    assert!(installation
        .right_for(&equal_but_foreign, &selected, &runtime)
        .is_none());
    assert!(installation
        .decline_for(&equal_but_foreign, &selected, &runtime)
        .is_none());

    let mut changed = selected.clone();
    changed.index = usize::MAX;
    assert!(installation.right_for(&plan, &changed, &runtime).is_none());
    let mut changed = selected.clone();
    changed.group.id = usize::MAX;
    assert!(installation.right_for(&plan, &changed, &runtime).is_none());
    let mut changed = selected.clone();
    changed.group.span.end = changed.group.span.end.saturating_add(1);
    assert!(installation.right_for(&plan, &changed, &runtime).is_none());
    let mut changed = selected.clone();
    changed.kernel.supported = false;
    assert!(installation.right_for(&plan, &changed, &runtime).is_none());

    let foreign_cancellation = CancellationSource::new();
    let foreign = runtime_context(
        &foreign_cancellation,
        selected_program.clone(),
        service.clone(),
    );
    assert!(installation.right_for(&plan, &selected, &foreign).is_none());
    assert!(installation
        .decline_for(&plan, &selected, &foreign)
        .is_none());

    let changed_program = runtime
        .clone()
        .with_program_revision(Some(program(b"changed")));
    assert_eq!(
        service.invalidations.borrow().as_slice(),
        &[PlacementInvalidation::Program {
            revision: selected_program.clone(),
        }]
    );
    assert!(installation
        .right_for(&plan, &selected, &changed_program)
        .is_none());
    assert!(!decline.is_current(&changed_program));

    let changed_generation = CancellationSource::new();
    let changed_generation_runtime = runtime
        .clone()
        .with_cancellation_token(changed_generation.token());
    assert!(installation
        .right_for(&plan, &selected, &changed_generation_runtime)
        .is_none());
    assert!(!decline.is_current(&changed_generation_runtime));

    drop(installation);
    assert!(!decline.is_current(&runtime));
    assert!(exact.placement_revision().is_none());
    assert!(exact.planning_revision(&runtime).is_none());

    let cancelled_source = CancellationSource::new();
    let cancelled_runtime = runtime_context(&cancelled_source, selected_program, service);
    let (cancelled_plan, _, cancelled_installation) =
        bytecode.prepare_vm_fusion_plan_with_installation(&cancelled_runtime);
    let cancelled_plan = cancelled_plan.expect("cancel fixture plan");
    let cancelled_installation = cancelled_installation.expect("cancel fixture installation");
    let cancelled_group = cancelled_plan
        .groups
        .iter()
        .find(|group| group.group.kind.is_reduction() && group.kernel.supported)
        .expect("cancel fixture supported reduction");
    cancelled_source.cancel(CancellationReason::User);
    assert!(cancelled_installation
        .right_for(&cancelled_plan, cancelled_group, &cancelled_runtime)
        .is_none());
}

#[test]
fn post_span_nonstore_cannot_certify_reduction_publication() {
    let cancellation = CancellationSource::new();
    let runtime = runtime_context(
        &cancellation,
        program(b"nonstore"),
        Rc::new(RevisionService::default()),
    );
    let mut bytecode = compiled_reduction();
    let (baseline, _, _) = bytecode.prepare_vm_fusion_plan_with_installation(&runtime);
    let baseline = baseline.expect("authentic source has a plan");
    let selected = baseline
        .groups
        .iter()
        .find(|group| group.group.kind.is_reduction() && group.kernel.supported)
        .expect("authentic reduction group");
    let following_pc = selected.group.span.end + 1;
    bytecode.instructions[following_pc] = Instr::Pop;
    bytecode.reseal_fusion_stage_after_owned_remap();
    let (changed, _, installation) = bytecode.prepare_vm_fusion_plan_with_installation(&runtime);
    let changed = changed.expect("post-span Pop does not erase the selected reduction plan");
    let installation = installation.expect("changed plan still owns installed regions");
    let changed_group = changed
        .groups
        .iter()
        .find(|group| group.group.kind.is_reduction() && group.kernel.supported)
        .expect("same supported reduction still exists before publication");
    assert!(installation
        .publication_pc_for(&changed, changed_group)
        .is_none());
}
