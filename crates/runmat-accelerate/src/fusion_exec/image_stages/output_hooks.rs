//! Held checkpoint-at-a-time output execution. The VM must acknowledge each
//! successful source store before another hook can run.

use super::*;
use runmat_accelerate_api::{ProviderImageOutputHook, ProviderImageOutputOperand};

enum HookResult {
    Ordinary(GpuTensorHandle),
    SharedReal(selected_real_output::SelectedRealOutput),
}

impl ImageNormalizeStageSession {
    /// Execute only through the next source-visible output checkpoint. All
    /// intermediate and previously published handles remain protected until
    /// the selected program completes; the physical owner must budget that
    /// exact lifetime before this route can be activated.
    pub(crate) async fn next_output_checkpoint(
        &mut self,
    ) -> ProviderExecutionResult<ImageOutputCheckpoint> {
        let Some(program) = self.program.output_program.as_ref() else {
            return Err(
                ProviderExecutionError::from(anyhow!("selected output hooks are absent"))
                    .after_effect(),
            );
        };
        let calls = &program.calls;
        // Allocate the bounded checkpoint index before moving the live mean
        // and sigma out of their state. A capacity failure cannot silently
        // retire those provider handles or turn this continuation replayable.
        let initial_layout = if matches!(&self.state, State::SigmaPublished { .. }) {
            match OutputProgress::layout(calls) {
                Ok(layout) => Some(layout),
                Err(error) => {
                    self.state = State::Failed;
                    return Err(error);
                }
            }
        } else {
            None
        };
        let prior = std::mem::replace(&mut self.state, State::InFlight);
        let mut progress = match prior {
            State::SigmaPublished { mean, sigma } => OutputProgress::new(
                mean,
                sigma,
                initial_layout.expect("layout prepared for sigma publication"),
            ),
            State::OutputReady(progress) => progress,
            other => {
                self.state = other;
                return Err(ProviderExecutionError::from(anyhow!(
                    "image output checkpoint is out of order"
                ))
                .after_effect());
            }
        };
        while let Some(call) = calls.get(progress.next_call) {
            let output = self.execute_output_hook(call, &progress).await;
            let output = match output {
                Ok(output) => output,
                Err(error) => {
                    self.state = State::Failed;
                    return Err(Self::abort_output_progress(error.after_effect(), progress));
                }
            };
            let expected = call
                .output_shape
                .map(|dimension| usize::try_from(dimension).expect("validated output shape"));
            let protected = progress
                .retained_values()
                .chain([&progress.mean, &progress.sigma])
                .collect::<Vec<_>>();
            let output = match output {
                HookResult::Ordinary(output) => self.admit_next(output, &expected, &[], &protected),
                HookResult::SharedReal(executed) => {
                    // Only the shared cast recipe may publish an exact source
                    // alias. Its owner has already authenticated live view,
                    // class, extent, borrowed leases, route, and observations.
                    if executed.route
                        == runmat_accelerate_api::ProviderRealElementwiseOutputRoute::SameClassFloatingCastAlias
                        && !matches!(
                            call.hook,
                            ProviderImageOutputHook::CastSingle | ProviderImageOutputHook::CastDouble
                        )
                    {
                        Err(ProviderExecutionError::from(anyhow!(
                            "selected non-cast output attempted an alias"
                        ))
                        .after_effect())
                    } else {
                        Ok(executed.output)
                    }
                }
            };
            let output = match output {
                Ok(output)
                    if runmat_accelerate_api::handle_precision(&output)
                        == Some(call.output_precision) =>
                {
                    output
                }
                Ok(output) => {
                    let failure = super::super::image_output::reject_untrusted_output(
                        output,
                        std::iter::once(&self.input).chain(protected.iter().copied()),
                        "selected image output has the wrong physical class",
                    );
                    self.state = State::Failed;
                    return Err(Self::abort_output_progress(
                        failure.after_effect(),
                        progress,
                    ));
                }
                Err(error) => {
                    self.state = State::Failed;
                    return Err(Self::abort_output_progress(error.after_effect(), progress));
                }
            };
            let output = if call.checkpoint {
                match self.provenance(output) {
                    Ok(output) => output,
                    Err(error) => {
                        self.state = State::Failed;
                        return Err(Self::abort_output_progress(error.after_effect(), progress));
                    }
                }
            } else {
                output
            };
            progress.values[progress.next_call] = Some(output.clone());
            progress.next_call += 1;
            if call.checkpoint {
                self.state = State::OutputAwaitAck {
                    progress,
                    source_output: call.source_output,
                };
                return Ok(ImageOutputCheckpoint {
                    source_output: call.source_output,
                    handle: output,
                });
            }
        }
        self.state = State::Failed;
        Err(ProviderExecutionError::from(anyhow!(
            "selected image program has no remaining checkpoint"
        ))
        .after_effect())
    }

    /// Call only after the VM has successfully published the yielded handle
    /// at the exact selected store. A wrong ID performs no work and leaves
    /// the same checkpoint pending; execution failure or dropped execution
    /// futures remain terminal and never replay earlier provider hooks.
    pub(crate) fn acknowledge_output_checkpoint(
        &mut self,
        source_output: u64,
    ) -> ProviderExecutionResult<()> {
        let prior = std::mem::replace(&mut self.state, State::InFlight);
        let State::OutputAwaitAck {
            progress,
            source_output: pending,
        } = prior
        else {
            self.state = prior;
            return Err(ProviderExecutionError::from(anyhow!(
                "image output has no pending publication"
            ))
            .after_effect());
        };
        if pending != source_output {
            self.state = State::OutputAwaitAck {
                progress,
                source_output: pending,
            };
            return Err(ProviderExecutionError::from(anyhow!(
                "image output publication is out of order"
            ))
            .after_effect());
        }
        let total = self
            .program
            .output_program
            .as_ref()
            .map_or(0, |program| program.calls.len());
        if progress.next_call < total {
            self.state = State::OutputReady(progress);
            return Ok(());
        }
        let cleanup = progress
            .into_values()
            .filter_map(|handle| handle.release().err())
            .collect::<Vec<_>>();
        self.state = State::Complete;
        if cleanup.is_empty() {
            Ok(())
        } else {
            Err(
                ProviderExecutionError::from_source(ProviderExecutionCompletionError::new(
                    None, cleanup,
                ))
                .after_effect(),
            )
        }
    }

    fn abort_output_progress(
        error: ProviderExecutionError,
        progress: OutputProgress,
    ) -> ProviderExecutionError {
        let cleanup = progress
            .into_values()
            .filter_map(|handle| handle.release().err())
            .collect::<Vec<_>>();
        if cleanup.is_empty() {
            error
        } else {
            ProviderExecutionError::from_source(ProviderExecutionCompletionError::new(
                Some(error),
                cleanup,
            ))
            .after_effect()
        }
    }

    async fn execute_output_hook(
        &self,
        call: &runmat_accelerate_api::ProviderImageOutputCall,
        progress: &OutputProgress,
    ) -> ProviderExecutionResult<HookResult> {
        if self.program.selected_powers.is_some()
            && matches!(
                call.hook,
                ProviderImageOutputHook::ElemMul | ProviderImageOutputHook::ElemAdd
            )
        {
            return Err(ProviderExecutionError::from(anyhow!(
                "selected output contains an unplanned binary hook"
            ))
            .after_effect());
        }
        let tensor =
            |operand: ProviderImageOutputOperand| -> ProviderExecutionResult<GpuTensorHandle> {
                match operand {
                    ProviderImageOutputOperand::Input => Ok(self.input.clone()),
                    ProviderImageOutputOperand::Mean => Ok(progress.mean.clone()),
                    ProviderImageOutputOperand::Sigma => Ok(progress.sigma.clone()),
                    ProviderImageOutputOperand::Produced(id) => progress
                        .get(id)
                        .cloned()
                        .ok_or_else(|| anyhow!("selected image hook input is absent").into()),
                    _ => Err(anyhow!("selected image hook requires a tensor operand").into()),
                }
            };
        let first = *call
            .operands
            .first()
            .ok_or_else(|| anyhow!("selected image hook has no operand"))?;
        let left = tensor(first)?;
        let right = call.operands.get(1).copied();
        let result = match call.hook {
            ProviderImageOutputHook::ElemSub
            | ProviderImageOutputHook::ElemDiv
            | ProviderImageOutputHook::ElemMul
            | ProviderImageOutputHook::ElemAdd => {
                let rhs = tensor(
                    right.ok_or_else(|| anyhow!("selected binary hook has no right operand"))?,
                )?;
                if self.program.selected_powers.is_some()
                    && matches!(
                        call.hook,
                        ProviderImageOutputHook::ElemSub | ProviderImageOutputHook::ElemDiv
                    )
                {
                    let stage = if call.hook == ProviderImageOutputHook::ElemSub {
                        selected_binary::BinaryStage::OutputSub
                    } else {
                        selected_binary::BinaryStage::OutputDiv
                    };
                    let selected = self.selected_binary_plans.as_ref().ok_or_else(|| {
                        ProviderExecutionError::from(anyhow!(
                            "selected output real binary recipe is absent"
                        ))
                        .after_effect()
                    })?;
                    if !selected.matches_output(stage, call) {
                        return Err(ProviderExecutionError::from(anyhow!(
                            "selected output real binary source call changed"
                        ))
                        .after_effect());
                    }
                    let mut protected = vec![&self.input, &progress.mean, &progress.sigma];
                    protected.extend(progress.retained_values());
                    return selected
                        .execute(
                            stage,
                            &left,
                            &rhs,
                            &protected,
                            self.provider_metadata_permit.as_ref().map(|permit| {
                                permit.as_ref() as &dyn runmat_execution::resource::ProviderHostMetadataAdmission
                            }),
                        )
                        .await
                        .map(HookResult::Ordinary);
                }
                match call.hook {
                    ProviderImageOutputHook::ElemSub => self.provider.elem_sub(&left, &rhs).await,
                    ProviderImageOutputHook::ElemDiv => self.provider.elem_div(&left, &rhs).await,
                    ProviderImageOutputHook::ElemMul => self.provider.elem_mul(&left, &rhs).await,
                    ProviderImageOutputHook::ElemAdd => self.provider.elem_add(&left, &rhs).await,
                    _ => unreachable!("selected elementwise hook"),
                }
            }
            ProviderImageOutputHook::ElemPow => {
                let exponent = right.ok_or_else(|| anyhow!("selected power has no exponent"))?;
                let selected = self
                    .program
                    .selected_powers
                    .as_ref()
                    .zip(self.selected_power_plans.as_ref())
                    .and_then(|(calls, plans)| {
                        (call.source_output == u64::from(calls.final_power.output))
                            .then_some((&calls.final_power, &plans.final_power))
                    });
                if self.program.selected_powers.is_some() && selected.is_none() {
                    return Err(anyhow!("selected POWER output changed after admission").into());
                }
                let legacy_bits = match exponent {
                    ProviderImageOutputOperand::ScalarF32Bits(bits) => bits,
                    ProviderImageOutputOperand::ScalarF64Bits(_) if selected.is_some() => 0,
                    _ => return Err(anyhow!("image POWER control lacks typed authority").into()),
                };
                let mut protected = vec![&self.input, &progress.mean, &progress.sigma];
                protected.extend(progress.retained_values());
                return self
                    .execute_stage_power(&left, selected, legacy_bits, &protected)
                    .await
                    .map(HookResult::Ordinary);
            }
            ProviderImageOutputHook::ScalarAdd
            | ProviderImageOutputHook::ScalarMul
            | ProviderImageOutputHook::ScalarMax => {
                if self.program.selected_powers.is_some() {
                    let selected = self.selected_real_plans.as_ref().ok_or_else(|| {
                        ProviderExecutionError::from(anyhow!(
                            "selected scalar output recipe is absent"
                        ))
                        .after_effect()
                    })?;
                    let mut protected = vec![&self.input, &progress.mean, &progress.sigma];
                    protected.extend(progress.retained_values());
                    return selected
                        .execute(call, &left, &protected, self.provider_metadata_permit.as_ref().map(|permit| permit.as_ref() as &dyn runmat_execution::resource::ProviderHostMetadataAdmission))
                        .await
                        .map(HookResult::SharedReal);
                }
                let scalar = match right
                    .ok_or_else(|| anyhow!("selected scalar hook has no control"))?
                {
                    ProviderImageOutputOperand::ScalarF32Bits(bits) => {
                        f64::from(f32::from_bits(bits))
                    }
                    ProviderImageOutputOperand::ScalarF64Bits(bits) => f64::from_bits(bits),
                    _ => return Err(anyhow!("selected scalar hook has no scalar control").into()),
                };
                match call.hook {
                    ProviderImageOutputHook::ScalarAdd => self.provider.scalar_add(&left, scalar),
                    ProviderImageOutputHook::ScalarMul => self.provider.scalar_mul(&left, scalar),
                    ProviderImageOutputHook::ScalarMax => self.provider.scalar_max(&left, scalar),
                    _ => unreachable!("selected scalar hook"),
                }
                .map_err(Into::into)
            }
            ProviderImageOutputHook::CastSingle | ProviderImageOutputHook::CastDouble
                if self.program.selected_powers.is_some() =>
            {
                let selected = self.selected_real_plans.as_ref().ok_or_else(|| {
                    ProviderExecutionError::from(anyhow!("selected cast output recipe is absent"))
                        .after_effect()
                })?;
                let mut protected = vec![&self.input, &progress.mean, &progress.sigma];
                protected.extend(progress.retained_values());
                return selected
                    .execute(
                        call,
                        &left,
                        &protected,
                        self.provider_metadata_permit.as_ref().map(|permit| {
                            permit.as_ref() as &dyn runmat_execution::resource::ProviderHostMetadataAdmission
                        }),
                    )
                    .await
                    .map(HookResult::SharedReal);
            }
            ProviderImageOutputHook::CastSingle => self.provider.unary_single(&left).await,
            ProviderImageOutputHook::CastDouble => self.provider.unary_double(&left).await,
        };
        result
            .map(HookResult::Ordinary)
            .map_err(ProviderExecutionError::after_effect)
    }
}
