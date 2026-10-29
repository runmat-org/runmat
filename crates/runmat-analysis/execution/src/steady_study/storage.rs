//! Enumerate each owned buffer copy; shared content addresses do not imply shared heap storage.
use super::*;

fn sum(bytes: &mut u64, additional: u64) -> AnalysisBridgeResult<()> {
    *bytes = bytes
        .checked_add(additional)
        .ok_or_else(|| invalid("retained terminal storage overflow"))?;
    Ok(())
}
fn inputs(bytes: &mut u64, objects: &[PreparedAnalysisObjects]) -> AnalysisBridgeResult<()> {
    for object in objects {
        sum(bytes, object.retained_storage_bytes()?)?;
    }
    Ok(())
}
/// Retained metadata is resident typed data; charge its typed size, not a JSON tree.
fn metadata<T: serde::Serialize>(value: &T) -> AnalysisBridgeResult<u64> {
    Ok(runmat_canonical_codec::typed_resident_bytes_estimate(
        value,
    )?)
}
impl SteadyStudyTail {
    pub(crate) fn retained_storage_bytes(&self) -> AnalysisBridgeResult<u64> {
        let mut bytes = 0;
        inputs(&mut bytes, &self.source.context_inputs)?;
        sum(&mut bytes, self.source.result.retained_storage_bytes()?)?;
        sum(&mut bytes, self.source.receipt.retained_storage_bytes()?)?;
        sum(&mut bytes, metadata(&self.source.host.workload)?)?;
        if let TerminalStatePublicationSource::Profiles { profiles, .. } = &self.source.state {
            sum(&mut bytes, profiles.retained_storage_bytes()?)?;
        }
        if let Some(task) = &self.accuracy_task {
            inputs(&mut bytes, &task.inputs)?;
            sum(&mut bytes, task.program.retained_storage_bytes()?)?;
        }
        if let Some(host) = &self.completed_accuracy_host {
            sum(&mut bytes, metadata(&host.workload)?)?;
            sum(&mut bytes, metadata(&host.artifact_access)?)?;
        }
        if let Some(task) = &self.state_task {
            inputs(&mut bytes, &task.inputs)?;
            sum(&mut bytes, task.program.retained_storage_bytes()?)?;
        }
        if let Some(task) = &self.publication_task {
            inputs(&mut bytes, &task.inputs)?;
            sum(&mut bytes, task.program.retained_storage_bytes()?)?;
        }
        if let Some(accuracy) = &self.accuracy {
            sum(&mut bytes, accuracy.objects.retained_storage_bytes()?)?;
            sum(&mut bytes, metadata(&accuracy.artifact)?)?;
        }
        for state in &self.states {
            sum(&mut bytes, state.objects.retained_storage_bytes()?)?;
            sum(&mut bytes, metadata(&state.response)?)?;
        }
        for usage in [&self.usage, &self.baseline, &self.terminal]
            .into_iter()
            .flatten()
        {
            sum(&mut bytes, usage.objects.retained_storage_bytes()?)?;
            sum(&mut bytes, metadata(&usage.evidence)?)?;
        }
        if let Some(publication) = &self.publication {
            sum(&mut bytes, publication.objects.retained_storage_bytes()?)?;
        }
        Ok(bytes)
    }
}
