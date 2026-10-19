mod collective;
mod distributed;
mod spmd;

pub(crate) struct CoreParallelServices {
    pub(crate) spmd: std::rc::Rc<dyn runmat_runtime::context::RuntimeSpmdService>,
    pub(crate) distributed: std::rc::Rc<dyn runmat_runtime::context::RuntimeDistributedService>,
}

impl CoreParallelServices {
    pub(crate) fn new() -> Self {
        let store = std::rc::Rc::new(std::cell::RefCell::new(
            runmat_execution_runner::DistributedStore::default(),
        ));
        Self {
            spmd: std::rc::Rc::new(spmd::CoreSpmdService::new(std::rc::Rc::clone(&store))),
            distributed: std::rc::Rc::new(distributed::CoreDistributedService::new(store)),
        }
    }
}
