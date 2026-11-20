//! Browser FEA resource admission. A browser run's memory budget derives from the
//! platform that runs it, the same law native placement follows: the planner solves
//! within min(authored resident ceiling, this budget), the Run gate admits the plan
//! the planner produced, and every refusal names its term, requirement and maximum.
use runmat_analysis_fea::planning::resource::{FeaResourcePlan, ResourcePlanningError};

/// Bytes one wasm32 instance can address: 65,536 pages of 64 KiB. Every browser
/// FEA instance (the session worker and each owned numerical worker) is one.
pub const WASM32_ADDRESS_SPACE_BYTES: u64 = 1 << 32;

pub const BROWSER_RESOURCE_LIMIT_CODE: &str = "RM.FEA.RUN_STUDY.BROWSER_RESOURCE_LIMIT";
const NEXT_ACTION: &str = "Continue in the desktop app.";

/// What one browser instance of this module can hold, measured where it runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BrowserMemoryPlatform {
    /// Static data and stack of the module (its heap base). Every instance holds
    /// them before its first allocation.
    pub module_baseline_bytes: u64,
    /// Device memory the browser reports, when it reports one.
    pub device_memory_bytes: Option<u64>,
}

impl BrowserMemoryPlatform {
    /// The resident budget of one instance: its address space after the module's
    /// static footprint, within the device memory. A fresh owned worker starts
    /// with this much free heap.
    pub fn instance_budget_bytes(&self) -> u64 {
        let addressable = WASM32_ADDRESS_SPACE_BYTES.saturating_sub(self.module_baseline_bytes);
        self.device_memory_bytes
            .map_or(addressable, |device| device.min(addressable))
    }

    /// Free heap of a running instance: its budget less the bytes its allocator
    /// holds live. Grown but freed linear memory is free; it is reused.
    pub fn free_bytes(&self, live_heap_bytes: u64) -> u64 {
        self.instance_budget_bytes().saturating_sub(live_heap_bytes)
    }
}

/// The resource a browser run needs more of than its budget offers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BrowserResourceTerm {
    /// Resident memory of the planned solve, plus the request text the numerical
    /// worker holds while it reads the prepared study.
    SolveMemory,
    /// Scratch workspace of the planned solve.
    SolveScratch,
    /// The serialized prepared study, written in the session worker before transfer.
    PreparedRequest,
    /// The numerical result, read back into the session worker for publication.
    NumericalResult,
    /// A numerical worker's request against the allowance its parent granted.
    WorkerAllowance,
    /// The check-time estimate of the solve's resident memory with its prepared mesh.
    EstimatedMemory,
    /// The check-time estimate of the solve's scratch workspace.
    EstimatedScratch,
}

impl BrowserResourceTerm {
    /// The term's wire name in diagnostics and operation errors.
    pub fn name(self) -> &'static str {
        match self {
            Self::SolveMemory => "solve_memory",
            Self::SolveScratch => "solve_scratch",
            Self::PreparedRequest => "prepared_request",
            Self::NumericalResult => "numerical_result",
            Self::WorkerAllowance => "worker_allowance",
            Self::EstimatedMemory => "estimated_memory",
            Self::EstimatedScratch => "estimated_scratch",
        }
    }

    fn subject(self) -> &'static str {
        match self {
            Self::SolveMemory => "The solve",
            Self::SolveScratch => "The solve workspace",
            Self::PreparedRequest => "The prepared simulation",
            Self::NumericalResult => "The numerical result",
            Self::WorkerAllowance => "The numerical request",
            Self::EstimatedMemory => "The estimated solve",
            Self::EstimatedScratch => "The estimated solve workspace",
        }
    }
}

/// A coded refusal: which term tripped, what it requires and the maximum the
/// browser offers. `required_bytes` is absent only when the planner could not
/// size the requirement because another authored limit bound first.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BrowserResourceLimit {
    pub term: BrowserResourceTerm,
    pub required_bytes: Option<u64>,
    pub maximum_bytes: u64,
}

impl BrowserResourceLimit {
    pub fn check(term: BrowserResourceTerm, required: u64, maximum: u64) -> Result<(), Self> {
        if required > maximum {
            Err(Self {
                term,
                required_bytes: Some(required),
                maximum_bytes: maximum,
            })
        } else {
            Ok(())
        }
    }

    /// The planner's refusal when no solve path fits the browser budget.
    pub fn from_planning(error: &ResourcePlanningError) -> Option<Self> {
        match *error {
            ResourcePlanningError::StaticSolveCeiling {
                multigrid_resident_bytes,
                multigrid_scratch_bytes,
                maximum,
                ..
            } => Some(if multigrid_resident_bytes > maximum {
                Self {
                    term: BrowserResourceTerm::SolveMemory,
                    required_bytes: Some(multigrid_resident_bytes),
                    maximum_bytes: maximum,
                }
            } else {
                Self {
                    term: BrowserResourceTerm::SolveScratch,
                    required_bytes: multigrid_scratch_bytes,
                    maximum_bytes: maximum,
                }
            }),
            ResourcePlanningError::MemoryBudget {
                required_memory_bytes,
                available_memory_bytes,
                required_scratch_bytes,
                available_scratch_bytes,
            } => Some(if required_memory_bytes > available_memory_bytes {
                Self {
                    term: BrowserResourceTerm::SolveMemory,
                    required_bytes: Some(required_memory_bytes),
                    maximum_bytes: available_memory_bytes,
                }
            } else {
                Self {
                    term: BrowserResourceTerm::SolveScratch,
                    required_bytes: Some(required_scratch_bytes),
                    maximum_bytes: available_scratch_bytes,
                }
            }),
            _ => None,
        }
    }

    /// Restates a planner refusal made within the budget less the request text against
    /// the whole budget, as the Run gate counts it: the solve's resident requirement
    /// includes the text the worker holds while it reads the study.
    pub fn with_request(self, request_bytes: u64, budget_bytes: u64) -> Self {
        match self.term {
            BrowserResourceTerm::SolveMemory => Self {
                required_bytes: self
                    .required_bytes
                    .map(|required| required.saturating_add(request_bytes)),
                maximum_bytes: budget_bytes,
                ..self
            },
            _ => self,
        }
    }

    pub fn message(&self) -> String {
        let subject = self.term.subject();
        let maximum = self.maximum_bytes;
        match self.required_bytes {
            Some(required) => format!(
                "{subject} requires {required} bytes of browser memory, maximum {maximum} bytes. {NEXT_ACTION}"
            ),
            None => format!(
                "{subject} requires more than the browser memory maximum of {maximum} bytes. {NEXT_ACTION}"
            ),
        }
    }

    /// The runtime diagnostic the Desktop renders (identifier, message) with the
    /// typed numbers for the person and the agent.
    pub fn diagnostic(&self) -> serde_json::Value {
        let message = self.message();
        serde_json::json!({
            "kind": "runtime",
            "identifier": BROWSER_RESOURCE_LIMIT_CODE,
            "message": message,
            "diagnostic": message,
            "callstack": [],
            "resourceLimit": {
                "term": self.term.name(),
                "requiredBytes": self.required_bytes,
                "maximumBytes": self.maximum_bytes,
                "nextAction": NEXT_ACTION,
            },
        })
    }

    /// The same refusal on an execution candidate, for the Run card before Run.
    pub fn resource_limit(&self) -> runmat_runtime::analysis::AnalysisResourceLimit {
        runmat_runtime::analysis::AnalysisResourceLimit {
            code: BROWSER_RESOURCE_LIMIT_CODE.to_owned(),
            term: self.term.name().to_owned(),
            required_bytes: self.required_bytes,
            maximum_bytes: self.maximum_bytes,
            next_action: NEXT_ACTION.to_owned(),
        }
    }

    /// The same refusal as an RM operation error, for study steps a script runs.
    pub fn operation_error(
        &self,
        operation: &str,
        op_version: &str,
        context: &runmat_runtime::operations::OperationContext,
    ) -> runmat_runtime::operations::OperationErrorEnvelope {
        use runmat_runtime::operations::{
            OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
        };
        let mut details = std::collections::BTreeMap::from([
            ("term".to_owned(), self.term.name().to_owned()),
            ("maximum_bytes".to_owned(), self.maximum_bytes.to_string()),
            ("next_action".to_owned(), NEXT_ACTION.to_owned()),
        ]);
        if let Some(required) = self.required_bytes {
            details.insert("required_bytes".to_owned(), required.to_string());
        }
        runmat_runtime::operations::operation_error(
            operation,
            op_version,
            context,
            OperationErrorSpec {
                error_code: BROWSER_RESOURCE_LIMIT_CODE,
                error_type: OperationErrorType::Capacity,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            self.message(),
            details,
        )
    }
}

/// Admits a prepared solve into a fresh numerical worker of `budget_bytes`. The
/// worker holds the request text while it reads it into the study the plan counts,
/// so the resident requirement is the plan's memory plus the request bytes. The
/// worker grants scratch from the same budget, as native placement does.
pub fn admit_browser_solve(
    plan: &FeaResourcePlan,
    request_bytes: u64,
    budget_bytes: u64,
) -> Result<(), BrowserResourceLimit> {
    admit_solve_bytes(
        plan.request.memory_bytes,
        plan.request.scratch_bytes,
        request_bytes,
        budget_bytes,
    )
}

fn admit_solve_bytes(
    memory_bytes: u64,
    scratch_bytes: u64,
    request_bytes: u64,
    budget_bytes: u64,
) -> Result<(), BrowserResourceLimit> {
    BrowserResourceLimit::check(
        BrowserResourceTerm::SolveMemory,
        memory_bytes.saturating_add(request_bytes),
        budget_bytes,
    )?;
    BrowserResourceLimit::check(
        BrowserResourceTerm::SolveScratch,
        scratch_bytes,
        budget_bytes,
    )
}

/// A numerical worker accepts its parent's allowance only within its own budget,
/// and only a request text that fits that allowance.
pub fn admit_worker_request(
    request_bytes: u64,
    allowance_bytes: u64,
    worker_budget_bytes: u64,
) -> Result<(), BrowserResourceLimit> {
    BrowserResourceLimit::check(
        BrowserResourceTerm::WorkerAllowance,
        allowance_bytes,
        worker_budget_bytes,
    )?;
    BrowserResourceLimit::check(
        BrowserResourceTerm::WorkerAllowance,
        request_bytes,
        allowance_bytes,
    )
}

#[cfg(test)]
#[path = "fea_browser_budget_tests.rs"]
mod tests;
