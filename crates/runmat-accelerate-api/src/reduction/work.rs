//! Algorithmic stage prior for the checked recipe's actual tree schedule.
//! No provider timing, numerical admission, or shader provenance is inferred.
use super::*;
use crate::{scalar_program::ScalarTransport, ProviderPrecision};
use anyhow::{anyhow, ensure, Result};
use runmat_types::FusionReductionWorkPrior;

impl ProviderReductionRecipe {
    pub fn terminal_work_prior(
        &self,
        strategy: ReductionStrategy,
    ) -> Result<FusionReductionWorkPrior> {
        let d = self.descriptor();
        let g = &d.geometry;
        let (length, slices) = d
            .contract
            .checked_work_geometry()
            .ok_or_else(|| anyhow!("unbounded reduction work geometry"))?;
        ensure!(
            length == u64::from(g.reduce_elements) && slices == u64::from(g.slices),
            "reduction work geometry differs from checked contract"
        );
        let samples = length
            .checked_mul(slices)
            .ok_or_else(|| anyhow!("reduction sample work overflow"))?;
        let (samples, combines) = if d.contract.identity {
            // Single pass forwards to output. Two pass additionally writes
            // one unchanged partial per slice; no floating reduction occurs.
            (
                if strategy == ReductionStrategy::TwoPass {
                    slices
                } else {
                    0
                },
                0,
            )
        } else {
            let (groups, partial_visits) = match strategy {
                ReductionStrategy::SinglePass => (u64::from(g.single.logical_groups), 0),
                ReductionStrategy::TwoPass => (
                    u64::from(g.pass1.logical_groups)
                        .checked_add(u64::from(g.pass2.logical_groups))
                        .ok_or_else(|| anyhow!("reduction group work overflow"))?,
                    u64::from(g.partial_elements),
                ),
            };
            // Each rr_value first combines into a lane accumulator. Pass2
            // combines existing partials without calling rr_sample. Each
            // logical group then executes exactly wg-1 tree combines, even
            // for lanes/chunks whose accumulator is empty.
            let tree = groups
                .checked_mul(u64::from(
                    d.config
                        .workgroup_size
                        .checked_sub(1)
                        .ok_or_else(|| anyhow!("zero reduction workgroup"))?,
                ))
                .ok_or_else(|| anyhow!("reduction tree work overflow"))?;
            (
                samples,
                samples
                    .checked_add(partial_visits)
                    .and_then(|n| n.checked_add(tree))
                    .ok_or_else(|| anyhow!("reduction combine work overflow"))?,
            )
        };
        let input_conversion_accounted = d.producer_transport == ScalarTransport::Binary32Words
            && d.partial.precision == ProviderPrecision::F64;
        let mut prior = d
            .contract
            .algorithmic_work_for_schedule(samples, combines, slices, input_conversion_accounted)
            .ok_or_else(|| anyhow!("reduction terminal work overflow or invalid policy"))?;
        if !d.contract.identity {
            // The selected RRPartial ABI's rr_combine adds the two counts even
            // when the public operation has no denominator/count requirement.
            // Keep this emitted state overhead out of the original host prior.
            prior.count_updates = combines;
        }
        prior
            .checked_units()
            .ok_or_else(|| anyhow!("reduction state work overflow"))?;
        Ok(prior)
    }
}
