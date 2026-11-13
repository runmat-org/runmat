use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use futures::executor::block_on;
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::execution::RuntimeExecutionService;
use runmat_value::{
    CellArray, ExactHostAllocationPermit, ExactHostAllocationReservation,
    ExactHostNumericAllocationError, HandleRef, OutputListStorage, Value,
};

use super::{interpret_parfor_task_in_context, ParforTaskExecution, ParforTaskMode};

thread_local! {
    static OBSERVED_TASK_MEMBER: RefCell<Option<runmat_gc::GcHandle>> = const { RefCell::new(None) };
}

struct ObservedTaskMember(Option<runmat_gc::GcHandle>);

#[derive(Debug)]
struct IterationPermit(Arc<AtomicUsize>);

#[derive(Debug)]
struct IterationReservation(Arc<AtomicUsize>);

impl ExactHostAllocationReservation for IterationReservation {}

impl Drop for IterationReservation {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl ExactHostAllocationPermit for IterationPermit {
    fn reserve_exact(
        &self,
        _: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(IterationReservation(Arc::clone(&self.0))))
    }
}

impl ObservedTaskMember {
    fn install(handle: runmat_gc::GcHandle) -> Self {
        Self(OBSERVED_TASK_MEMBER.with(|slot| slot.replace(Some(handle))))
    }
}

impl Drop for ObservedTaskMember {
    fn drop(&mut self) {
        OBSERVED_TASK_MEMBER.with(|slot| {
            slot.replace(self.0.take());
        });
    }
}

#[test]
fn task_results_retain_managed_members_through_collecting_root_retirement() {
    #[cfg(not(target_arch = "wasm32"))]
    if runmat_gc::test_support::run_isolated_gc_test(concat!(
        module_path!(),
        "::task_results_retain_managed_members_through_collecting_root_retirement"
    )) {
        return;
    }
    for mode in [
        ParforTaskMode::Sequential,
        ParforTaskMode::IndependentIterations,
    ] {
        runmat_gc::gc_test_context(|| {
            let ast = runmat_parser::parse(
                "source = {}; out = cell(1, 2); parfor i = 1:2; out{i} = source; end",
            )
            .expect("parse task");
            let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty())
                .expect("lower task HIR");
            let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower task MIR");
            let analysis = runmat_mir::analysis::analyze_assembly(&mir);
            assert_eq!(
                analysis.parallel.parfor_regions.len(),
                1,
                "task must have one classified parfor: {:?}",
                analysis.diagnostics
            );
            let mut bytecode = crate::bytecode::compile_with_analysis(
                &hir.assembly,
                &mir,
                hir.assembly.entrypoints[0].id,
                &analysis,
            )
            .expect("compile analyzed task");
            let compiled_functions = mir
                .bodies
                .keys()
                .map(|local| {
                    let facts = analysis
                        .function_for_local(*local)
                        .expect("exact analysis self binding");
                    (*local, facts.function)
                })
                .collect();
            let region_contracts = analysis
                .regions
                .iter()
                .map(|region| region.contract.clone())
                .collect::<Vec<_>>();
            bytecode
                .install_regions(&region_contracts, &compiled_functions)
                .expect("install analyzed regions");
            bytecode
                .install_parallel_regions(&mir, &analysis.parallel, &compiled_functions)
                .expect("install analyzed parallel regions");
            let region = bytecode
                .parfor_regions
                .first()
                .expect("compiled parfor region");
            assert_eq!(region.contract, analysis.parallel.parfor_regions[0]);
            let initial = runmat_gc::gc_allocate_rooted(Value::String("task payload".into()))
                .expect("allocate managed member");
            let handle = initial.handle();
            let member = Value::HandleObject(HandleRef {
                class_name: "TaskPayload".into(),
                target: handle,
                valid: true,
            });
            let input = Value::Cell(
                CellArray::new(vec![member.clone(), member], 1, 2).expect("task input"),
            );
            let inputs = runmat_runtime::execution::RootedValues::new(
                region.input_variables().map(|_| input.clone()).collect(),
            )
            .expect("retain task inputs before source retirement");
            initial.unroot().expect("transfer member to task");
            let _member_scope = ObservedTaskMember::install(handle);
            let origin_thread = std::thread::current().id();
            let collecting = Arc::new(AtomicBool::new(false));
            let retirements = Arc::new(AtomicUsize::new(0));
            let observed = Arc::clone(&retirements);
            let _observer = runmat_gc::RootRetirementObserver::register(Arc::new(move |_| {
                if std::thread::current().id() != origin_thread {
                    return;
                }
                if collecting.swap(true, Ordering::SeqCst) {
                    return;
                }
                runmat_gc::gc_collect_major().expect("collect during task handoff");
                let member = OBSERVED_TASK_MEMBER.with(|slot| {
                    slot.borrow()
                        .as_ref()
                        .copied()
                        .expect("origin-thread task member installed")
                });
                assert!(
                    runmat_gc::gc_clone_value(&member).is_ok(),
                    "task member remains rooted"
                );
                observed.fetch_add(1, Ordering::SeqCst);
                collecting.store(false, Ordering::SeqCst);
            }))
            .expect("observe task root retirement");
            let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
            let iteration_claims = Arc::new(AtomicUsize::new(0));
            let iterations = OutputListStorage::try_initialize(
                2,
                &IterationPermit(Arc::clone(&iteration_claims)),
                |index| Ok::<_, std::convert::Infallible>(Value::Num((index + 1) as f64)),
            )
            .expect("admit the original iteration collection");
            assert_eq!(iteration_claims.load(Ordering::SeqCst), 1);
            let outputs = block_on(inputs.with_owned_output_storage_async(|inputs| {
                interpret_parfor_task_in_context(ParforTaskExecution {
                    bytecode_owner: None,
                    bytecode: &bytecode,
                    region,
                    inputs,
                    iterations,
                    input_roots: None,
                    randomness: runmat_execution::ParallelRandomnessContext::Inherit,
                    mode,
                    current_function_name: "<task>",
                    runtime,
                })
            }))
            .expect("execute task");
            assert_eq!(iteration_claims.load(Ordering::SeqCst), 0);
            assert_eq!(outputs.values().len(), region.output_variables().count());
            assert!(
                retirements.load(Ordering::SeqCst) > 0,
                "exercise actual root retirement"
            );
            runmat_gc::gc_collect_major().expect("collect with returned outputs");
            assert_eq!(
                runmat_gc::gc_clone_value(&handle).expect("returned output owns member"),
                Value::String("task payload".into())
            );
            drop(_observer);
            drop(outputs);
            runmat_gc::gc_collect_major().expect("collect retired task output");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }
}
