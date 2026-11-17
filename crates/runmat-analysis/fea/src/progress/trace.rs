//! Trace spans of a study run, which the Desktop Trace pane shows as steps.
//!
//! A run is one INFO span [`STUDY_SPAN`] with the study file path and the run id. Each phase of
//! the run is an INFO child span with a string field `step` (the name the Trace pane shows)
//! and a string field `result`, recorded when the phase ends:
//!
//! | span | `step` | `result`, for example |
//! |---|---|---|
//! | `fea.study.check` | Check study | "0 errors" |
//! | `fea.study.mesh` | Mesh | "12,180 tet10" |
//! | `fea.study.assemble` | Assemble | "351,000 dof" |
//! | `fea.study.solve` | Solve | "1 step" |
//! | `fea.study.recover` | Recover fields | "21 outputs" |
//! | `fea.study.assess` | Assess acceptance | "1 of 1 passed" |
//!
//! A sweep, and an analysis that runs as several chained solves, opens one [`CASE_SPAN`] per
//! case with `step = "Case <name>"`; the phase spans of that case are its children. Spans
//! without a `step` field (meshing stages, solver iteration batches, factorizations) are the
//! detail beneath a step.
//!
//! The native runner and the browser host open these spans at the points where they switch the
//! run's progress phase, with the functions of this module, so both trace the same steps.
use tracing::field::Empty;
use tracing::Span;

/// Name of the span around a whole study run.
pub const STUDY_SPAN: &str = "fea.study";
/// Name of the span around one case of a sweep or a chained analysis.
pub const CASE_SPAN: &str = "fea.study.case";
/// Field that names a step in the Trace pane.
pub const STEP_FIELD: &str = "step";
/// Field with a step's one-line result.
pub const RESULT_FIELD: &str = "result";

/// One phase of a study run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudyStep {
    Check,
    Mesh,
    Assemble,
    Solve,
    Recover,
    Assess,
}

impl StudyStep {
    /// Every step, in run order.
    pub const ALL: [StudyStep; 6] = [
        Self::Check,
        Self::Mesh,
        Self::Assemble,
        Self::Solve,
        Self::Recover,
        Self::Assess,
    ];

    /// The span name.
    pub const fn span_name(self) -> &'static str {
        match self {
            Self::Check => "fea.study.check",
            Self::Mesh => "fea.study.mesh",
            Self::Assemble => "fea.study.assemble",
            Self::Solve => "fea.study.solve",
            Self::Recover => "fea.study.recover",
            Self::Assess => "fea.study.assess",
        }
    }

    /// The `step` field: the name the Trace pane shows.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Check => "Check study",
            Self::Mesh => "Mesh",
            Self::Assemble => "Assemble",
            Self::Solve => "Solve",
            Self::Recover => "Recover fields",
            Self::Assess => "Assess acceptance",
        }
    }

    /// Opens this step's span as a child of the current span. Record its result with
    /// [`finish`] before the span closes.
    pub fn span(self) -> Span {
        let step = self.label();
        match self {
            Self::Check => tracing::info_span!("fea.study.check", step, result = Empty),
            Self::Mesh => tracing::info_span!("fea.study.mesh", step, result = Empty),
            Self::Assemble => tracing::info_span!("fea.study.assemble", step, result = Empty),
            Self::Solve => tracing::info_span!("fea.study.solve", step, result = Empty),
            Self::Recover => tracing::info_span!("fea.study.recover", step, result = Empty),
            Self::Assess => tracing::info_span!("fea.study.assess", step, result = Empty),
        }
    }
}

/// Opens the span around a whole run of the study at `study` (its file path). Record the run
/// id with [`record_run_id`] once the run is admitted.
pub fn study_span(study: &str) -> Span {
    tracing::info_span!("fea.study", study, run_id = Empty)
}

/// Records the admitted run's id on its [`study_span`].
pub fn record_run_id(span: &Span, run_id: &str) {
    span.record("run_id", run_id);
}

/// Opens the span around one case named `name`, as a child of the current span.
pub fn case_span(name: &str) -> Span {
    let step = format!("Case {name}");
    tracing::info_span!("fea.study.case", step = step.as_str())
}

/// Records a step's one-line result.
pub fn finish(span: &Span, result: &str) {
    span.record("result", result);
}

/// "0 errors", "1 error", "3 errors".
pub fn check_result(errors: usize) -> String {
    plural(errors as u64, "error", "errors")
}

/// The element count and kind of a solver mesh: "12,180 tet10".
pub fn mesh_result(mesh: &runmat_meshing_core::SolverMeshArtifact) -> String {
    let elements = &mesh.topology.volume_elements;
    let kind = match elements.first().map(|element| element.order) {
        Some(runmat_meshing_core::ElementOrder::Tet4) => "tet4",
        Some(runmat_meshing_core::ElementOrder::Tet10) => "tet10",
        None => "elements",
    };
    format!("{} {kind}", grouped(elements.len() as u64))
}

/// "351,000 dof".
pub fn dof_result(dofs: u64) -> String {
    format!("{} dof", grouped(dofs))
}

/// "41,200 nodes".
pub fn node_result(nodes: u64) -> String {
    plural(nodes, "node", "nodes")
}

/// "1 step", "3 steps": the analysis steps a solve ran.
pub fn steps_result(steps: usize) -> String {
    plural(steps as u64, "step", "steps")
}

/// "21 outputs": the authored outputs a run recovered.
pub fn outputs_result(outputs: usize) -> String {
    plural(outputs as u64, "output", "outputs")
}

/// "1 of 1 passed", or "no criteria" for a study without acceptance criteria.
pub fn assess_result(passed: usize, total: usize) -> String {
    if total == 0 {
        "no criteria".into()
    } else {
        format!("{passed} of {total} passed")
    }
}

fn plural(count: u64, one: &str, many: &str) -> String {
    format!(
        "{} {}",
        grouped(count),
        if count == 1 { one } else { many }
    )
}

/// `12180` as "12,180".
pub fn grouped(value: u64) -> String {
    let digits = value.to_string();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    grouped
}

#[cfg(test)]
#[path = "trace_tests.rs"]
mod tests;
