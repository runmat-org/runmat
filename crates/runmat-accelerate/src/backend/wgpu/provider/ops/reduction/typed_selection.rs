//! Effect-free configuration and feasible physical candidates for one checked
//! recipe. Legacy benchmark autotuning is intentionally not invoked here.
use super::*;
use runmat_accelerate_api::{
    reduction::{
        ProviderReductionRecipe, ReductionRecipeConfig, ReductionSelection, ReductionStrategy,
    },
    ProviderExecutionError, ProviderExecutionResult, ReductionTwoPassMode,
};

fn ordered_strategies(
    mode: ReductionTwoPassMode,
    preferred: ReductionStrategy,
) -> [Option<ReductionStrategy>; 2] {
    match mode {
        ReductionTwoPassMode::ForceOff => [Some(ReductionStrategy::SinglePass), None],
        ReductionTwoPassMode::ForceOn => [Some(ReductionStrategy::TwoPass), None],
        ReductionTwoPassMode::Auto => [
            Some(preferred),
            Some(if preferred == ReductionStrategy::SinglePass {
                ReductionStrategy::TwoPass
            } else {
                ReductionStrategy::SinglePass
            }),
        ],
    }
}

/// Pure policy selection over an already authenticated device-feasible set.
/// It cannot manufacture a feasible route or launch a preparation trial.
pub(super) fn select_candidates(
    mode: ReductionTwoPassMode,
    preferred: ReductionStrategy,
    feasible: &[Option<ReductionStrategy>; 2],
) -> ProviderExecutionResult<[Option<ReductionStrategy>; 2]> {
    let mut candidates = [None, None];
    let mut count = 0;
    for strategy in ordered_strategies(mode, preferred).into_iter().flatten() {
        if feasible.contains(&Some(strategy)) {
            candidates[count] = Some(strategy);
            count += 1;
        }
    }
    if count == 0 {
        return Err(typed_recipe::preflight::unsupported(
            "no executable typed reduction strategy on this device",
        ));
    }
    Ok(candidates)
}

impl WgpuProvider {
    pub(crate) fn typed_reduction_config(
        &self,
        reduce_elements: u32,
        slices: u32,
    ) -> ProviderExecutionResult<ReductionRecipeConfig> {
        let limits = &self.adapter_limits;
        let workgroup_size = self.reduction_workgroup_size_default;
        let max_workgroup_size = limits
            .max_compute_workgroup_size_x
            .min(limits.max_compute_invocations_per_workgroup);
        if workgroup_size == 0
            || workgroup_size > max_workgroup_size
            || reduce_elements == 0
            || slices == 0
        {
            return Err(typed_recipe::preflight::unsupported(
                "typed reduction configuration is unavailable",
            ));
        }
        let chunk_elements = self
            .default_chunk_rows(reduce_elements as usize, workgroup_size)
            .max(1);
        Ok(ReductionRecipeConfig {
            workgroup_size,
            chunk_elements,
            max_workgroup_size,
            max_workgroup_storage_bytes: u64::from(limits.max_compute_workgroup_storage_size),
            max_dispatch_dimension: limits.max_compute_workgroups_per_dimension,
            max_storage_binding_bytes: limits
                .max_buffer_size
                .min(u64::from(limits.max_storage_buffer_binding_size)),
            max_rank: crate::backend::wgpu::params::BCAST_MAX_RANK,
        })
    }

    pub(crate) fn typed_reduction_candidates(
        &self,
        recipe: &ProviderReductionRecipe,
        host_metadata: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<[Option<ReductionStrategy>; 2]> {
        let mode = self.reduction_two_pass_mode;
        let (preferred, feasible) =
            self.typed_reduction_feasible_candidates(recipe, host_metadata)?;
        select_candidates(mode, preferred, &feasible)
    }

    pub(super) fn typed_reduction_feasible_candidates(
        &self,
        recipe: &ProviderReductionRecipe,
        host_metadata: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<(ReductionStrategy, [Option<ReductionStrategy>; 2])> {
        let d = recipe.descriptor();
        let config = self.typed_reduction_config(d.geometry.reduce_elements, d.geometry.slices)?;
        if d.config != config {
            return Err(typed_recipe::preflight::unsupported(
                "typed reduction recipe configuration differs from executing device",
            ));
        }
        crate::fusion::reduction_recipe::validate_lowered_sources(recipe, host_metadata)
            .map_err(ProviderExecutionError::from)?;
        let preferred = self.heuristic_reduction_tuning(
            d.geometry.reduce_elements as usize,
            d.geometry.slices as usize,
            d.config.workgroup_size,
        );
        let preferred = match preferred.mode {
            ReductionMode::SinglePass => ReductionStrategy::SinglePass,
            ReductionMode::TwoPass { .. } => ReductionStrategy::TwoPass,
        };
        let mut feasible = [None, None];
        for (index, strategy) in [ReductionStrategy::SinglePass, ReductionStrategy::TwoPass]
            .into_iter()
            .enumerate()
        {
            let Ok(selected) = ReductionSelection::new(recipe, strategy) else {
                continue;
            };
            match self.admit_reduction_selection(&selected, host_metadata) {
                Ok(()) => feasible[index] = Some(strategy),
                Err(error) if error.is_unsupported() => {}
                Err(error) => return Err(error),
            }
        }
        Ok((preferred, feasible))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn existing_mode_policy_orders_feasible_candidates_without_launching_trials() {
        assert_eq!(
            ordered_strategies(ReductionTwoPassMode::Auto, ReductionStrategy::TwoPass),
            [
                Some(ReductionStrategy::TwoPass),
                Some(ReductionStrategy::SinglePass)
            ]
        );
        assert_eq!(
            ordered_strategies(ReductionTwoPassMode::ForceOff, ReductionStrategy::TwoPass),
            [Some(ReductionStrategy::SinglePass), None]
        );
        assert_eq!(
            ordered_strategies(ReductionTwoPassMode::ForceOn, ReductionStrategy::SinglePass),
            [Some(ReductionStrategy::TwoPass), None]
        );
        let feasible = [Some(ReductionStrategy::SinglePass), None];
        assert_eq!(
            select_candidates(
                ReductionTwoPassMode::Auto,
                ReductionStrategy::TwoPass,
                &feasible
            )
            .unwrap(),
            feasible
        );
        assert!(select_candidates(
            ReductionTwoPassMode::ForceOn,
            ReductionStrategy::SinglePass,
            &feasible,
        )
        .unwrap_err()
        .is_unsupported());
    }
}
