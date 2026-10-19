use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{ProcedureStepId, ValidatedAnalysisSequence};
use runmat_analysis_fea::execution::{
    AnalysisTaskIdentity, SequenceIdentity, StepIdentity, StudyIdentity,
};
use runmat_execution::identity::ArtifactId;
use runmat_execution::value::ValuePayload;
use runmat_execution::TaskId;
use runmat_execution_artifact::ProgramExecutionRequest;
use runmat_execution_runner::TaskSubmission;

use crate::{
    build_analysis_task_submission, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisExecutionContext, AnalysisHostWorkload,
};

/// A fully prepared procedure-step workload awaiting semantic graph compilation.
pub struct PreparedAnalysisStepTask {
    pub step_id: ProcedureStepId,
    pub host: AnalysisHostWorkload,
    pub program: ProgramExecutionRequest,
    pub context: AnalysisExecutionContext,
}

#[derive(Clone, Debug)]
pub struct AnalysisSequenceTask {
    pub step_id: ProcedureStepId,
    pub identity: AnalysisTaskIdentity,
    pub program: ProgramExecutionRequest,
    pub submission: TaskSubmission,
}

#[derive(Clone, Debug)]
pub struct AnalysisSequenceExecutionGraph {
    pub study: StudyIdentity,
    pub tasks: Vec<AnalysisSequenceTask>,
}

impl AnalysisSequenceExecutionGraph {
    pub fn into_program_tasks(self) -> Vec<(ProgramExecutionRequest, TaskSubmission)> {
        self.tasks
            .into_iter()
            .map(|task| (task.program, task.submission))
            .collect()
    }
}

/// Projects a validated semantic sequence into generic scheduler tasks.
///
/// Only dependencies already present in the validated analysis sequence are
/// emitted. Placement remains in each execution context and cannot change task
/// identity or semantic ordering.
pub fn compile_analysis_sequence_graph(
    sequence: &ValidatedAnalysisSequence,
    prepared: Vec<PreparedAnalysisStepTask>,
) -> AnalysisBridgeResult<AnalysisSequenceExecutionGraph> {
    let mut by_step = BTreeMap::new();
    for task in prepared {
        let step_id = task.step_id.clone();
        if by_step.insert(step_id.clone(), task).is_some() {
            return Err(AnalysisBridgeError::Invalid(format!(
                "analysis sequence has duplicate prepared step `{}`",
                step_id.as_str()
            )));
        }
    }
    if by_step.len() != sequence.topological_order().len()
        || sequence
            .topological_order()
            .iter()
            .any(|step_id| !by_step.contains_key(step_id))
    {
        return Err(AnalysisBridgeError::Invalid(
            "prepared analysis steps do not exactly cover the validated sequence".into(),
        ));
    }

    let mut compiled = BTreeMap::<ProcedureStepId, (AnalysisTaskIdentity, TaskId)>::new();
    let mut tasks = Vec::with_capacity(by_step.len());
    let mut study = None;
    for step_id in sequence.topological_order() {
        let task = by_step
            .remove(step_id)
            .ok_or(AnalysisBridgeError::Identity(
                "validated sequence step is not prepared",
            ))?;
        task.host.validate()?;
        match study {
            Some(expected) if expected != task.host.workload.study => {
                return Err(AnalysisBridgeError::Identity(
                    "prepared sequence workloads belong to different studies",
                ));
            }
            None => study = Some(task.host.workload.study),
            _ => {}
        }
        let sequence_identity = SequenceIdentity::from_solver_contract(sequence.sequence())?;
        let authored_step = sequence
            .sequence()
            .steps
            .iter()
            .find(|candidate| candidate.step_id == *step_id)
            .ok_or(AnalysisBridgeError::Identity(
                "validated sequence topology references an absent procedure step",
            ))?;
        let expected_step = StepIdentity::from_solver_contract(sequence_identity, authored_step)?;
        if task.host.workload.step != expected_step {
            return Err(AnalysisBridgeError::Identity(
                "prepared workload step identity differs from its authored procedure step",
            ));
        }

        let predecessors =
            sequence
                .direct_predecessors()
                .get(step_id)
                .ok_or(AnalysisBridgeError::Identity(
                    "validated sequence topology has no predecessor inventory",
                ))?;
        let dependency_identities = predecessors
            .iter()
            .map(|predecessor| {
                compiled
                    .get(predecessor)
                    .map(|(identity, _)| *identity)
                    .ok_or(AnalysisBridgeError::Identity(
                        "validated sequence predecessor was not compiled before its successor",
                    ))
            })
            .collect::<AnalysisBridgeResult<Vec<_>>>()?;
        let dependency_tasks = predecessors
            .iter()
            .map(|predecessor| {
                compiled
                    .get(predecessor)
                    .map(|(_, task_id)| *task_id)
                    .ok_or(AnalysisBridgeError::Identity(
                        "validated sequence predecessor task is absent",
                    ))
            })
            .collect::<AnalysisBridgeResult<BTreeSet<_>>>()?;
        let identity =
            AnalysisTaskIdentity::new(task.host.workload.identity, dependency_identities.clone())?;
        task.program.validate().map_err(|error| {
            AnalysisBridgeError::Invalid(format!("analysis program request is invalid: {error}"))
        })?;
        let program_artifact_id = ArtifactId::derive(&[task.program.artifact.id.0.bytes()]);
        if task.context.program_artifact_id != program_artifact_id {
            return Err(AnalysisBridgeError::Identity(
                "analysis execution context names a different program artifact",
            ));
        }
        let program_roots = task
            .program
            .arguments
            .iter()
            .map(|argument| match argument {
                ValuePayload::Object(reference) => Ok(reference.as_ref().clone()),
                _ => Err(AnalysisBridgeError::Invalid(
                    "analysis sequence program inputs must remain externalized".into(),
                )),
            })
            .collect::<AnalysisBridgeResult<Vec<_>>>()?;
        let submission = build_analysis_task_submission(
            &task.host,
            program_roots,
            dependency_identities,
            dependency_tasks,
            task.context,
        )?;
        if submission.request.id != TaskId::derive(&[identity.digest().bytes()]) {
            return Err(AnalysisBridgeError::Identity(
                "compiled sequence task identity does not converge",
            ));
        }
        compiled.insert(step_id.clone(), (identity, submission.request.id));
        tasks.push(AnalysisSequenceTask {
            step_id: step_id.clone(),
            identity,
            program: task.program,
            submission,
        });
    }

    Ok(AnalysisSequenceExecutionGraph {
        study: study.ok_or_else(|| {
            AnalysisBridgeError::Invalid("validated sequence contains no prepared tasks".into())
        })?,
        tasks,
    })
}
