//! Exact source-order coverage by the held selected image stage programs.
//! This is a prerequisite for arithmetic proof, not a provider capability.

use super::*;

fn moment_source_outputs(
    calls: &[runmat_accelerate_api::ProviderImageMomentPublication],
) -> impl Iterator<Item = u64> + '_ {
    calls.iter().flat_map(|call| {
        std::iter::once(call.raw_output)
            .chain(call.conversions.iter().map(|cast| cast.source_output))
    })
}

#[derive(Debug)]
enum SelectedStageCoverageFailure {
    Moments,
    Binary,
    Sigma,
    RealOutput,
    Powers,
    OutputProgram,
    OutputCallCount,
    OutputSub,
    OutputDiv,
    OutputPower,
    OutputReal,
    SourceOrder {
        selected: Option<u64>,
        source: Option<u64>,
    },
}

impl ImageSourceInvocation {
    /// Every selected graph operation must have exactly one staged owner, in
    /// source order. A matching pattern and final value alone do not prove
    /// that intermediate casts or a source-visible output were preserved.
    pub(crate) fn selected_stage_covers_source(&self) -> bool {
        match self.selected_stage_coverage() {
            Ok(()) => true,
            Err(reason) => {
                match reason {
                    SelectedStageCoverageFailure::SourceOrder { selected, source } => log::debug!(
                        target: "runmat_accelerate::fusion_exec",
                        "selected image source coverage declined: source order selected={selected:?}, source={source:?}"
                    ),
                    other => log::debug!(
                        target: "runmat_accelerate::fusion_exec",
                        "selected image source coverage declined: {other:?}"
                    ),
                }
                false
            }
        }
    }

    fn selected_stage_coverage(&self) -> Result<(), SelectedStageCoverageFailure> {
        let moments = self
            .moment_program
            .as_ref()
            .ok_or(SelectedStageCoverageFailure::Moments)?;
        let binary = self
            .selected_binary_calls()
            .ok_or(SelectedStageCoverageFailure::Binary)?;
        let sigma = self
            .selected_sigma_operations()
            .ok_or(SelectedStageCoverageFailure::Sigma)?;
        let powers = self
            .requested_staged_workload()
            .and_then(|workload| workload.selected_powers.as_ref())
            .ok_or(SelectedStageCoverageFailure::Powers)?;
        let real = self
            .selected_real_output_calls()
            .ok_or(SelectedStageCoverageFailure::RealOutput)?;
        let output_program = self
            .output_program
            .as_ref()
            .ok_or(SelectedStageCoverageFailure::OutputProgram)?;
        if output_program.calls.len() != real.calls().len() + 3 {
            return Err(SelectedStageCoverageFailure::OutputCallCount);
        }
        if output_program.calls.first().map(|call| call.source_output)
            != Some(binary.output_sub().output())
        {
            return Err(SelectedStageCoverageFailure::OutputSub);
        }
        if output_program.calls.get(1).map(|call| call.source_output)
            != Some(binary.output_div().output())
        {
            return Err(SelectedStageCoverageFailure::OutputDiv);
        }
        if output_program
            .calls
            .iter()
            .filter(|call| {
                matches!(
                    call.hook,
                    runmat_accelerate_api::ProviderImageOutputHook::ElemPow
                )
            })
            .map(|call| call.source_output)
            .ne([u64::from(powers.final_power.output)])
        {
            return Err(SelectedStageCoverageFailure::OutputPower);
        }
        if output_program
            .calls
            .iter()
            .filter(|call| {
                matches!(
                    call.hook,
                    runmat_accelerate_api::ProviderImageOutputHook::ScalarMul
                        | runmat_accelerate_api::ProviderImageOutputHook::ScalarAdd
                        | runmat_accelerate_api::ProviderImageOutputHook::ScalarMax
                        | runmat_accelerate_api::ProviderImageOutputHook::CastSingle
                        | runmat_accelerate_api::ProviderImageOutputHook::CastDouble
                )
            })
            .map(|call| call.source_output)
            .ne(real.calls().iter().map(ImageSelectedRealCall::output))
        {
            return Err(SelectedStageCoverageFailure::OutputReal);
        }
        let mut selected = moment_source_outputs(&moments.mu)
            .chain(std::iter::once(binary.sigma_center().output()))
            .chain(std::iter::once(u64::from(powers.square.output)))
            .chain(moment_source_outputs(&moments.variance))
            .chain(std::iter::once(sigma.source_ids().2))
            .chain(moment_source_outputs(std::slice::from_ref(&moments.sigma)))
            .chain(output_program.calls.iter().map(|call| call.source_output));
        let mut source = self.source.steps.iter();
        loop {
            match (selected.next(), source.next()) {
                (Some(expected), Some(step)) if expected == u64::from(step.output) => {}
                (None, None) => return Ok(()),
                (expected, actual) => {
                    return Err(SelectedStageCoverageFailure::SourceOrder {
                        selected: expected,
                        source: actual.map(|step| u64::from(step.output)),
                    });
                }
            }
        }
    }
}
