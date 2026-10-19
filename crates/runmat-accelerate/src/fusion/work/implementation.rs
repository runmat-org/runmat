//! Projection of source-rederived lowering records into cold algorithmic work.
//! These records describe selected code, not an independent execution receipt.
use super::*;
use runmat_accelerate_api::scalar_program::{EmittedScalarImplementation, ScalarImplementation};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn work_units(
    program: &ResolvedProgram,
    plan: &FusionGroupPlan,
    elements: u64,
    emitted: &[EmittedScalarImplementation],
) -> Result<u64> {
    let mut replacements = BTreeMap::new();
    let mut boundaries = BTreeSet::new();
    let mut boundary_units = 0_u64;
    for record in emitted {
        if record.invocations_per_element == 0
            || record.algorithmic_units_per_invocation == 0
            || program.floating_class(record.output)? != NumericClass::Single
        {
            return Err(anyhow!("invalid represented implementation work"));
        }
        let units = record
            .algorithmic_units_per_invocation
            .checked_mul(u64::from(record.invocations_per_element))
            .ok_or_else(|| anyhow!("implementation work overflows"))?;
        match record.implementation {
            ScalarImplementation::Binary32Product48RneV1
            | ScalarImplementation::Binary32Square48RneV1
            | ScalarImplementation::Binary32IdentityWordsV1
            | ScalarImplementation::Binary32ZeroPowerWordsV1 => {
                if replacements.insert(record.output, units).is_some() {
                    return Err(anyhow!("duplicate scalar implementation replacement"));
                }
            }
            ScalarImplementation::Binary32WidenToBinary64WordsV1 => {
                // A reduction boundary may consume a raw input root, with no
                // corresponding scalar Apply. It adds work rather than replacing
                // the operation that produced the Single value.
                if !boundaries.insert(record.output) {
                    return Err(anyhow!("duplicate scalar boundary conversion"));
                }
                boundary_units = boundary_units
                    .checked_add(units)
                    .ok_or_else(|| anyhow!("scalar boundary work overflows"))?;
            }
        }
    }
    let operations = plan.operations.iter().map(|operation| {
        let output = match operation {
            FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => output,
        }
        .ok_or_else(|| anyhow!("physical work requires a resolved output"))?;
        match replacements.remove(&output) {
            Some(units) => Ok(units),
            None => program.operation_work_units(operation),
        }
    });
    let work = total_work(
        elements,
        operations.chain(std::iter::once(Ok(boundary_units))),
    )?;
    if !replacements.is_empty() {
        return Err(anyhow!(
            "implementation work has no emitted source operation"
        ));
    }
    Ok(work)
}
