//! Borrowed task interface; phase ownership remains in the family coordinator.
use crate::{
    AnalysisHostWorkload, PreparedAnalysisProgram, PreparedObservationAssembly,
    PreparedObservationHistoryCheckpointTask, PreparedObservationHistoryFinalizationTask,
    PreparedObservationPartition, PreparedObservationReduction,
};
use runmat_analysis_fea::planning::resource::FeaResourcePlan;

pub struct SampledObservationTask<'a> {
    pub host: &'a AnalysisHostWorkload,
    pub program: &'a PreparedAnalysisProgram,
    pub resource_plan: &'a FeaResourcePlan,
}

pub(super) enum PendingObservationTask {
    Partition(Box<PreparedObservationPartition>),
    Reduction(Box<PreparedObservationReduction>),
    Checkpoint(Box<PreparedObservationHistoryCheckpointTask>),
    Assembly(Box<PreparedObservationAssembly>),
    Finalization(Box<PreparedObservationHistoryFinalizationTask>),
}

impl PendingObservationTask {
    pub(super) fn view(&self) -> SampledObservationTask<'_> {
        let (host, program, resource_plan) = match self {
            Self::Partition(task) => (&task.host, &task.program, &task.resource_plan),
            Self::Reduction(task) => (&task.host, &task.program, &task.resource_plan),
            Self::Checkpoint(task) => (&task.host, &task.program, &task.resource_plan),
            Self::Assembly(task) => (&task.host, &task.program, &task.resource_plan),
            Self::Finalization(task) => (&task.host, &task.program, &task.resource_plan),
        };
        SampledObservationTask {
            host,
            program,
            resource_plan,
        }
    }
}
