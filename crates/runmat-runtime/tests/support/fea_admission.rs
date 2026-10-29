//! Explicit per-invocation test host; never replaces production admission globally.
use std::sync::{Arc, Mutex};

use runmat_execution::resource::ResourceUsage;
use runmat_runtime::analysis::fea_run_admission::{
    FeaRunAdmission, FeaRunAdmissionContext, FeaRunDenied, FeaRunOutcome, FeaRunPermit,
    FeaRunRequest,
};

#[derive(Clone, Default)]
pub(crate) struct Events {
    pub requests: Vec<FeaRunRequest>,
    pub settlements: Vec<(FeaRunOutcome, ResourceUsage)>,
}

#[derive(Clone)]
pub(crate) struct TestFeaHost {
    events: Arc<Mutex<Events>>,
    denial: Option<FeaRunDenied>,
}

impl TestFeaHost {
    pub fn new(denial: Option<FeaRunDenied>) -> Self {
        Self {
            events: Arc::new(Mutex::new(Events::default())),
            denial,
        }
    }

    pub fn context(&self) -> FeaRunAdmissionContext {
        FeaRunAdmissionContext::new(Arc::new(self.clone()))
    }

    pub fn events(&self) -> Events {
        self.events.lock().unwrap().clone()
    }
}

struct Permit(Arc<Mutex<Events>>);
impl FeaRunPermit for Permit {
    fn settle(self: Box<Self>, outcome: FeaRunOutcome, usage: &ResourceUsage) {
        self.0
            .lock()
            .unwrap()
            .settlements
            .push((outcome, usage.clone()));
    }
}

impl FeaRunAdmission for TestFeaHost {
    fn admit(&self, request: &FeaRunRequest) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
        self.events.lock().unwrap().requests.push(request.clone());
        if let Some(denial) = &self.denial {
            return Err(denial.clone());
        }
        Ok(Box::new(Permit(Arc::clone(&self.events))))
    }
}
