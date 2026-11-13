//! Shared live sampled-task buffers and typed source metadata, excluding allocator overhead.
use super::*;
use crate::{PreparedAnalysisObjects, PreparedAnalysisProgram};
use runmat_canonical_codec::{profile_serialized, CanonicalLimits};
use serde::Serialize;

fn add(total: &mut u64, bytes: u64) -> AnalysisBridgeResult<()> {
    *total = total
        .checked_add(bytes)
        .ok_or_else(|| invalid("sampled retained storage overflow"))?;
    Ok(())
}
fn metadata<T: Serialize>(
    total: &mut u64,
    value: &T,
    limits: CanonicalLimits,
) -> AnalysisBridgeResult<()> {
    add(
        total,
        profile_serialized(value, limits)?.materialization_resident_bytes_estimate,
    )
}
fn task(
    total: &mut u64,
    inputs: &[PreparedAnalysisObjects],
    program: &PreparedAnalysisProgram,
) -> AnalysisBridgeResult<()> {
    add(total, program.retained_storage_bytes()?)?;
    for input in inputs {
        add(total, input.retained_storage_bytes()?)?;
    }
    Ok(())
}
impl SampledStudyObservations {
    pub(crate) fn retained_storage_bytes(&self, ceiling: u64) -> AnalysisBridgeResult<u64> {
        let limits = CanonicalLimits::content_bounded(
            usize::try_from(ceiling).map_err(|_| invalid("sampled storage ceiling overflow"))?,
            65_536,
            64,
        );
        let mut bytes = 0;
        metadata(&mut bytes, &self.catalog, limits)?;
        if let Some(fanout) = &self.fanout {
            metadata(&mut bytes, &fanout.source_request, limits)?;
            for partition in &fanout.partitions {
                metadata(
                    &mut bytes,
                    &(
                        &partition.source_request,
                        &partition.request,
                        &partition.partition_plan,
                        &partition.partition,
                        &partition.source_fields,
                        &(&partition.host.workload, &partition.host.artifact_access),
                        &partition.resource_plan,
                    ),
                    limits,
                )?;
                task(&mut bytes, &partition.inputs, &partition.program)?;
                add(
                    &mut bytes,
                    partition.resource_plan_objects.retained_storage_bytes()?,
                )?;
            }
        }
        for contribution in self.contributions.values() {
            metadata(&mut bytes, &contribution.contribution, limits)?;
            add(&mut bytes, contribution.objects.retained_storage_bytes()?)?;
        }
        for host in &self.reduction_hosts {
            metadata(&mut bytes, &(&host.workload, &host.artifact_access), limits)?;
        }
        for reduced in &self.reduced {
            metadata(&mut bytes, &reduced.result, limits)?;
            add(&mut bytes, reduced.objects.retained_storage_bytes()?)?;
        }
        if let Some(reduction) = &self.reduction {
            metadata(
                &mut bytes,
                &(
                    &reduction.source_request,
                    &reduction.request,
                    &reduction.partition_plan,
                    &(&reduction.host.workload, &reduction.host.artifact_access),
                    &reduction.resource_plan,
                ),
                limits,
            )?;
            task(&mut bytes, &reduction.inputs, &reduction.program)?;
            add(
                &mut bytes,
                reduction.resource_plan_objects.retained_storage_bytes()?,
            )?;
        }
        if let Some(assembly) = &self.assembly {
            metadata(
                &mut bytes,
                &(
                    &assembly.source_request,
                    &assembly.request,
                    &(&assembly.host.workload, &assembly.host.artifact_access),
                    &assembly.resource_plan,
                ),
                limits,
            )?;
            task(&mut bytes, &assembly.inputs, &assembly.program)?;
            add(
                &mut bytes,
                assembly.resource_plan_objects.retained_storage_bytes()?,
            )?;
        }
        if let Some(completed) = &self.completed {
            metadata(
                &mut bytes,
                &(&completed.manifest, &completed.output, &completed.access),
                limits,
            )?;
            add(
                &mut bytes,
                completed.catalog_objects.retained_storage_bytes()?,
            )?;
        }
        Ok(bytes)
    }
}
