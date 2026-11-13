//! Retire overlapping source publications in selection order, not drop order.
//!
//! A foreground owner can fail while a previously admitted callback is still
//! suspended. Its generation becomes invalid immediately, but rollback waits
//! until newer provisional selections retire. Code generations never own this
//! host-side lineage or its foreign service ports.

use super::*;
use std::cell::Cell;

#[derive(Clone, Copy)]
enum Disposition {
    Rollback,
    Commit,
}

#[derive(Clone, Copy)]
enum State {
    Active,
    Committed,
    Retired(Disposition),
}

pub(super) struct PublicationFrame {
    active: Rc<FunctionRegistryGeneration>,
    restore: Rc<FunctionRegistryGeneration>,
    previous: Option<Rc<PublicationFrame>>,
    state: Cell<State>,
    foreign_restore: RefCell<Option<Option<ForeignPreparation>>>,
    foreign_source_restore: RefCell<Option<Option<ForeignIssuanceSource>>>,
}

impl PublicationFrame {
    fn rollback_target(&self) -> &Rc<FunctionRegistryGeneration> {
        let mut target = &self.restore;
        let mut previous = self.previous.as_deref();
        while let Some(frame) = previous {
            if !matches!(frame.state.get(), State::Retired(Disposition::Rollback))
                || !Rc::ptr_eq(target, &frame.active)
            {
                break;
            }
            target = &frame.restore;
            previous = frame.previous.as_deref();
        }
        target
    }
}

pub(in crate::session) struct FunctionRegistryPublication {
    // Retire selection/foreign owners before releasing their host.
    frame: Rc<PublicationFrame>,
    evaluator: Rc<SessionSourceEvaluator>,
}

impl FunctionRegistryPublication {
    pub(super) fn new(
        evaluator: Rc<SessionSourceEvaluator>,
        active: Rc<FunctionRegistryGeneration>,
        restore: Rc<FunctionRegistryGeneration>,
    ) -> Self {
        let frame = Rc::new(PublicationFrame {
            active,
            restore,
            previous: evaluator.publication_head.borrow().clone(),
            state: Cell::new(State::Active),
            foreign_restore: RefCell::new(None),
            foreign_source_restore: RefCell::new(None),
        });
        let retired = evaluator.publication_head.replace(Some(Rc::clone(&frame)));
        drop(retired);
        Self { evaluator, frame }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(in crate::session) fn set_foreign_preparation(
        &mut self,
        prepared: Option<ForeignPreparation>,
    ) {
        let prior = self.evaluator.foreign_preparation.replace(prepared);
        self.frame.foreign_restore.replace(Some(prior));
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(in crate::session) fn set_foreign_issuance_source(
        &mut self,
        source: Option<ForeignIssuanceSource>,
    ) {
        let prior = self.evaluator.foreign_issuance_source.replace(source);
        self.frame.foreign_source_restore.replace(Some(prior));
    }

    pub(in crate::session) fn attach_selected_declaration(
        &self,
        product: Rc<SelectedDeclarationAdmission>,
    ) -> Result<(), RuntimeError> {
        self.frame.active.attach_selected_declaration(product)
    }

    pub(in crate::session) fn finish(self, committed: runmat_vm::FunctionRegistry) {
        self.frame.active.commit(committed);
        self.frame.state.set(State::Committed);
    }
}

impl Drop for FunctionRegistryPublication {
    fn drop(&mut self) {
        let disposition = match self.frame.state.get() {
            State::Active => {
                self.frame
                    .active
                    .retain_escaped_into(self.frame.rollback_target());
                self.frame.active.invalidate();
                Disposition::Rollback
            }
            State::Committed => Disposition::Commit,
            State::Retired(_) => unreachable!("publication guard retires once"),
        };
        self.frame.state.set(State::Retired(disposition));
        self.evaluator.retire_publications();
    }
}

impl runmat_vm::DynamicEvalPublication for FunctionRegistryPublication {}

impl SessionSourceEvaluator {
    fn retire_publications(&self) {
        // Keep the retired chain alive until all selections are restored.
        // This requires no retirement allocation and prevents foreign-owner
        // destructors from observing an intermediate invalidated ancestor.
        let retired_chain = self.publication_head.borrow().clone();
        loop {
            let Some(frame) = self.publication_head.borrow().clone() else {
                break;
            };
            let State::Retired(disposition) = frame.state.get() else {
                break;
            };
            // A successfully committed descendant is a newer authority. An
            // older failed parent cannot roll it or its foreign ports back.
            if matches!(disposition, Disposition::Rollback)
                && Rc::ptr_eq(&self.current_generation(), &frame.active)
            {
                // frame.active already retains the replaced generation.
                drop(self.replace_generation(Rc::clone(&frame.restore)));
                let prior = frame.foreign_restore.borrow_mut().take();
                if let Some(prior) = prior {
                    let retired = self.foreign_preparation.replace(prior);
                    frame.foreign_restore.replace(Some(retired));
                }
                let prior = frame.foreign_source_restore.borrow_mut().take();
                if let Some(prior) = prior {
                    let retired = self.foreign_issuance_source.replace(prior);
                    frame.foreign_source_restore.replace(Some(retired));
                }
            }
            let retired = self.publication_head.replace(frame.previous.clone());
            drop(retired);
        }
        drop(retired_chain);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    fn select_foreign_source(
        mut publication: FunctionRegistryPublication,
    ) -> (
        FunctionRegistryPublication,
        Arc<runmat_config::runtime::RunMatRuntimeConfig>,
    ) {
        let config = Arc::new(runmat_config::runtime::RunMatRuntimeConfig::default());
        publication.set_foreign_issuance_source(Some(ForeignIssuanceSource::Runtime {
            config: Arc::clone(&config),
        }));
        (publication, config)
    }

    #[test]
    fn overlapping_publications_preserve_the_live_head_and_retire_their_lineage() {
        #[derive(Clone, Copy)]
        enum Order {
            ChildFirst,
            ParentFirst,
            ParentCommitted,
            ChildCommitted,
        }
        for order in [
            Order::ChildFirst,
            Order::ParentFirst,
            Order::ParentCommitted,
            Order::ChildCommitted,
        ] {
            let runtime = RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ));
            let evaluator = SessionSourceEvaluator::new(
                runmat_runtime::workspace::BaseWorkspaceOwner::new(),
                runtime.local_identity(),
            );
            let original = evaluator.current_generation();
            let parent = evaluator
                .publish_for_execution(runmat_vm::FunctionRegistry::default(), None)
                .unwrap();
            let parent_generation = evaluator.current_generation();
            #[cfg(not(target_arch = "wasm32"))]
            let (parent, parent_source) = select_foreign_source(parent);
            let child = evaluator
                .publish_for_execution(runmat_vm::FunctionRegistry::default(), None)
                .unwrap();
            let child_generation = evaluator.current_generation();
            #[cfg(not(target_arch = "wasm32"))]
            let (child, child_source) = select_foreign_source(child);
            let parent_frame = Rc::downgrade(&parent.frame);
            let child_frame = Rc::downgrade(&child.frame);
            let expected = match order {
                Order::ChildFirst => {
                    drop(child);
                    drop(parent);
                    original
                }
                Order::ParentFirst => {
                    drop(parent);
                    assert!(Rc::ptr_eq(
                        &evaluator.current_generation(),
                        &child_generation
                    ));
                    assert!(parent_generation.registry().is_err());
                    assert!(child_generation.registry().is_ok());
                    assert!(Rc::ptr_eq(child.frame.rollback_target(), &original));
                    drop(child);
                    original
                }
                Order::ParentCommitted => {
                    parent.finish(runmat_vm::FunctionRegistry::default());
                    drop(child);
                    parent_generation
                }
                Order::ChildCommitted => {
                    child.finish(runmat_vm::FunctionRegistry::default());
                    drop(parent);
                    child_generation
                }
            };
            assert!(Rc::ptr_eq(&evaluator.current_generation(), &expected));
            assert!(expected.registry().is_ok());
            #[cfg(not(target_arch = "wasm32"))]
            {
                let expected_source = match order {
                    Order::ParentCommitted => Some(&parent_source),
                    Order::ChildCommitted => Some(&child_source),
                    Order::ChildFirst | Order::ParentFirst => None,
                };
                let actual = evaluator.foreign_issuance_source.borrow();
                match (actual.as_ref(), expected_source) {
                    (Some(ForeignIssuanceSource::Runtime { config }), Some(expected)) => {
                        assert!(Arc::ptr_eq(config, expected));
                    }
                    (None, None) => {}
                    _ => panic!("foreign selection did not follow generation retirement"),
                }
            }
            assert!(evaluator.publication_head.borrow().is_none());
            assert!(parent_frame.upgrade().is_none());
            assert!(child_frame.upgrade().is_none());
        }
    }
}
