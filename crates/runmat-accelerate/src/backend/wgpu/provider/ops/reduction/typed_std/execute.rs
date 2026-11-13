use super::super::phase_context::{
    AllocationFault, BeforeBinaryFault, SelectedEmission, SquareRetirementFault, StdFormulaPhase,
};
use super::frame::StdFrame;
use super::*;
use crate::backend::wgpu::types::{BinaryOpCode, GlobalReduceOp};

pub(super) struct StdFaultSeams<'a> {
    pub(super) after_allocation: Option<&'a AllocationFault<'a>>,
    pub(super) before_binary: Option<&'a BeforeBinaryFault<'a>>,
    pub(super) after_square_retirement: Option<&'a SquareRetirementFault<'a>>,
}

impl WgpuProvider {
    pub(crate) async fn typed_std_exec(
        &self,
        input: &GpuTensorHandle,
        dim: Option<usize>,
        normalization: ProviderStdNormalization,
        nan_mode: ProviderNanMode,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.typed_std_with_fault(input, dim, normalization, nan_mode, None)
            .await
    }

    pub(super) async fn typed_std_with_fault(
        &self,
        input: &GpuTensorHandle,
        dim: Option<usize>,
        normalization: ProviderStdNormalization,
        nan_mode: ProviderNanMode,
        after_allocation: Option<&AllocationFault<'_>>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.typed_std_with_phase_fault(
            input,
            dim,
            normalization,
            nan_mode,
            StdFaultSeams {
                after_allocation,
                before_binary: None,
                after_square_retirement: None,
            },
        )
        .await
    }

    pub(super) async fn typed_std_with_phase_fault(
        &self,
        input: &GpuTensorHandle,
        dim: Option<usize>,
        normalization: ProviderStdNormalization,
        nan_mode: ProviderNanMode,
        seams: StdFaultSeams<'_>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let frame = StdFrame::admit(self, input, dim, nan_mode)?;
        let selected = StdSelected::prepare(self, &frame, input).await?;
        frame
            .revalidate(self, input)
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        let effect = Cell::new(false);
        let result = (|| -> ProviderExecutionResult<GpuTensorHandle> {
            if frame.sample_count == 0 {
                return self
                    .fill_std_selected_exec(
                        &frame.output_shape,
                        f64::NAN,
                        &selected.fill,
                        &effect,
                        seams.after_allocation,
                    )
                    .map_err(|error| terminal(error, effect.get()));
            }
            let sum_pair = selected.sum.as_ref().ok_or_else(|| {
                ProviderExecutionError::from(anyhow!("std selected sum pair absent"))
                    .with_unproven_effects()
            })?;
            let sum = if let Some(dim) = frame.dim {
                self.reduce_std_dim_sum_selected_exec(
                    input,
                    dim,
                    sum_pair,
                    &effect,
                    seams.after_allocation,
                )
            } else {
                self.reduce_global_with_selected_pipeline_exec(
                    input,
                    GlobalReduceOp::Sum,
                    SelectedEmission {
                        selected: Some(sum_pair),
                        effect: Some(&effect),
                        after_allocation: seams.after_allocation,
                    },
                )
            }
            .map_err(|error| terminal(error, effect.get()))?;
            selected.track(&sum);
            if let Some(before_binary) = seams.before_binary {
                before_binary(effect.get(), &sum).map_err(|error| terminal(error, effect.get()))?;
            }
            let squared = self
                .binary_std_selected_exec(
                    BinaryOpCode::Mul,
                    input,
                    input,
                    selected.binary.as_ref().ok_or_else(|| {
                        ProviderExecutionError::from(anyhow!("std selected binary pair absent"))
                            .with_unproven_effects()
                    })?,
                    &effect,
                )
                .map_err(|error| terminal(error, effect.get()))?;
            selected.track(&squared);
            let sum_sq = if let Some(dim) = frame.dim {
                self.reduce_std_dim_sum_selected_exec(
                    &squared,
                    dim,
                    sum_pair,
                    &effect,
                    seams.after_allocation,
                )
            } else {
                self.reduce_global_with_selected_pipeline_exec(
                    &squared,
                    GlobalReduceOp::Sum,
                    SelectedEmission {
                        selected: Some(sum_pair),
                        effect: Some(&effect),
                        after_allocation: seams.after_allocation,
                    },
                )
            }
            .map_err(|error| terminal(error, effect.get()))?;
            selected.track(&sum_sq);
            let square_wire = squared.wire_descriptor();
            if sum_sq.shares_allocation_with(&squared) {
                drop(squared);
            } else {
                selected.retire_last_use(squared);
                if let Some(after_square_retirement) = seams.after_square_retirement {
                    after_square_retirement(
                        self,
                        &square_wire,
                        selected.retained_temporary_evidence(),
                    )
                    .map_err(|error| terminal(error, effect.get()))?;
                }
            }
            let output = self
                .finish_std_from_sums_selected_exec(
                    sum,
                    sum_sq,
                    frame.sample_count,
                    normalization,
                    &frame.output_shape,
                    StdFormulaPhase {
                        selected: Some(&selected),
                        effect: Some(&effect),
                    },
                )
                .map_err(|error| terminal(error, effect.get()))?;
            if output.shape() == frame.output_shape.as_slice() {
                Ok(output)
            } else {
                // Keep the fresh physical owner until completion can observe a
                // failed view publication's retirement result.
                selected.track(&output);
                output
                    .with_view(output.view().clone().with_shape(frame.output_shape.clone()))
                    .map_err(|error| terminal(error.into(), effect.get()))
            }
        })();
        selected.complete(result, effect.get())
    }
}
