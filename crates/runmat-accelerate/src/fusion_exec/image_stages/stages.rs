use super::*;

impl ImageNormalizeStageSession {
    /// Stage one produces exactly one resident [B,1,1] mean. A first-hook
    /// typed Unsupported remains pre-effect; after any returned allocation,
    /// all failures prohibit replay.
    pub(crate) async fn mean(&mut self) -> ProviderExecutionResult<ImageMean> {
        if !matches!(self.state, State::BeforeMean) {
            return Err(anyhow!("image mean checkpoint is out of order").into());
        }
        self.state = State::InFlight;
        let reserved = self.reserved_mu.take();
        let cold = self.cold_mu.take();
        let mut scratch = Vec::new();
        let result = async {
            let mean = if let Some(reserved) = reserved {
                let dims = first_nd_hook(&self.program.mu)
                    .ok_or_else(|| anyhow!("selected warm mean has no source N-D hook"))?;
                let resources = nd_resources::warm_mean(
                    self.provider,
                    &self.input,
                    &reserved,
                    &dims,
                    &self.moment_shape(),
                )?;
                let mean = reserved.into_mean();
                if resources.output_bytes != Some(mean.physical().requested_bytes()) {
                    return Err(anyhow!("selected warm N-D output bytes changed").into());
                }
                let mean = self.admit_next(mean, &self.moment_shape(), &scratch, &[])?;
                if let Some(publication) = self
                    .program
                    .moment_program
                    .as_ref()
                    .and_then(|program| program.mu.first())
                {
                    self.publish_moment(mean, publication, &mut scratch, &[], None)
                        .await?
                } else {
                    mean
                }
            } else if let Some(cold) = cold {
                let dims = first_nd_hook(&self.program.mu)
                    .ok_or_else(|| anyhow!("selected cold mean has no source N-D hook"))?;
                let resources = nd_resources::cold_mean(
                    self.provider,
                    &self.input,
                    &cold,
                    &dims,
                    &self.moment_shape(),
                )?;
                let requested_pair_bytes = cold.requested_pair_bytes().ok_or_else(|| {
                    ProviderExecutionError::from(anyhow!("selected cold N-D pair byte overflow"))
                })?;
                let requested_uniform_bytes = cold.requested_uniform_bytes();
                let executed = self
                    .provider
                    .execute_cold_mean_nd(
                        cold,
                        self.provider_metadata_permit.as_ref().map(|permit| {
                            permit.as_ref() as &dyn runmat_execution::resource::ProviderHostMetadataAdmission
                        }),
                    )
                    .map_err(|source| ProviderExecutionError::from(source).after_effect())?;
                nd_resources::executed_cold_mean(
                    self.provider,
                    &resources,
                    requested_pair_bytes,
                    requested_uniform_bytes,
                    &executed,
                    &self.moment_shape(),
                )?;
                let mean = self.admit_next(executed.mean, &self.moment_shape(), &scratch, &[])?;
                if let Some(publication) = self
                    .program
                    .moment_program
                    .as_ref()
                    .and_then(|program| program.mu.first())
                {
                    self.publish_moment(mean, publication, &mut scratch, &[], None)
                        .await?
                } else {
                    mean
                }
            } else {
                self.reduce_source_mean(&self.input, &self.program.mu, &mut scratch, &[], true)
                    .await?
            };
            self.provenance(mean)
        }
        .await;
        let result = Self::finish(result, scratch);
        match result {
            Ok(mean) => {
                self.state = State::MeanPublished(mean.clone());
                Ok(ImageMean(mean))
            }
            Err(error) => {
                self.state = State::Failed;
                Err(error)
            }
        }
    }

    /// Stage two produces exactly one resident [B,1,1] sigma. The returned
    /// mean from stage one is a protected borrowed output, not scratch.
    pub(crate) async fn sigma(&mut self) -> ProviderExecutionResult<ImageSigma> {
        let State::MeanPublished(mean) = &self.state else {
            return Err(anyhow!("image sigma checkpoint is out of order").into());
        };
        let mean = mean.clone();
        self.state = State::InFlight;
        let mut scratch = Vec::new();
        let result = async {
            let selected_sigma = self
                .selected_mean_plans
                .as_ref()
                .and_then(|plans| plans.sigma.as_ref());
            let selected_operations =
                selected_sigma.and_then(|selected| selected.operations.as_ref());
            if self.program.selected_powers.is_some()
                && (selected_operations.is_none()
                    || selected_operations
                        .is_some_and(|selected| !selected.matches_program(&self.program)))
            {
                return Err(ProviderExecutionError::from(anyhow!(
                    "selected Sigma source operation plan is missing or changed"
                ))
                .after_effect());
            }
            let centered = if self.program.selected_powers.is_some() {
                let selected = self.selected_binary_plans.as_ref().ok_or_else(|| {
                    ProviderExecutionError::from(anyhow!(
                        "selected Sigma centered subtraction recipe is absent"
                    ))
                    .after_effect()
                })?;
                if !selected.matches_sigma(&self.program) {
                    return Err(ProviderExecutionError::from(anyhow!(
                        "selected Sigma centered subtraction source changed"
                    ))
                    .after_effect());
                }
                selected
                    .execute(
                        selected_binary::BinaryStage::SigmaCenter,
                        &self.input,
                        &mean,
                        &[&self.input, &mean],
                        self.provider_metadata_permit.as_ref().map(|permit| {
                            permit.as_ref() as &dyn runmat_execution::resource::ProviderHostMetadataAdmission
                        }),
                    )
                    .await?
            } else {
                self.provider
                    .elem_sub(&self.input, &mean)
                    .await
                    .map_err(ProviderExecutionError::after_effect)?
            };
            let centered = self.admit_next(centered, self.input.shape(), &scratch, &[&mean])?;
            scratch.push(centered);
            let selected = self
                .program
                .selected_powers
                .as_ref()
                .zip(self.selected_power_plans.as_ref())
                .map(|(calls, plans)| (&calls.square, &plans.square));
            let squared = self
                .execute_stage_power(
                    &scratch[0],
                    selected,
                    self.program.square_exponent_f32_bits,
                    &[&self.input, &mean, &scratch[0]],
                )
                .await?;
            let squared = self.admit_next(squared, self.input.shape(), &scratch, &[&mean])?;
            scratch.push(squared);
            let squared_input = scratch[1].clone();
            let variance = self
                .reduce_source_mean(
                    &squared_input,
                    &self.program.variance,
                    &mut scratch,
                    &[&mean],
                    false,
                )
                .await?;
            scratch.push(variance);
            let shifted_input = scratch.len() - 1;
            let shifted = if let Some(selected) = selected_operations {
                let protected = std::iter::once(&self.input)
                    .chain(std::iter::once(&mean))
                    .chain(scratch.iter())
                    .collect::<Vec<_>>();
                selected
                    .execute_add(
                        &scratch[shifted_input],
                        &protected,
                        self.provider_metadata_permit.as_ref().map(|permit| {
                            permit.as_ref() as &dyn runmat_execution::resource::ProviderHostMetadataAdmission
                        }),
                    )
                    .await?
            } else {
                // Only the explicit unselected held fixture uses legacy hooks.
                let runmat_accelerate_api::scalar_program::ScalarLiteral::F32Bits(bits) =
                    self.program.epsilon
                else {
                    return Err(ProviderExecutionError::from(anyhow!(
                        "unselected image Sigma requires its admitted F32 control"
                    ))
                    .after_effect());
                };
                self.provider
                    .scalar_add(&scratch[shifted_input], f64::from(f32::from_bits(bits)))
                    .map_err(|error| ProviderExecutionError::from(error).after_effect())?
            };
            let shifted = self.admit_next(shifted, &self.moment_shape(), &scratch, &[&mean])?;
            if self.program.selected_powers.is_some() && selected_sigma.is_none() {
                return Err(ProviderExecutionError::from(anyhow!(
                    "selected Sigma physical proposal disappeared"
                ))
                .after_effect());
            }
            if let Some(selected) = selected_sigma {
                if !selected
                    .revalidate_source(self.provider, &shifted)
                    .map_err(ProviderExecutionError::after_effect)?
                {
                    return Err(ProviderExecutionError::from(anyhow!(
                        "selected Sigma source or allocator changed"
                    ))
                    .after_effect());
                }
            }
            scratch.push(shifted);
            let sqrt_input = scratch.len() - 1;
            let sigma = if let Some(selected) = selected_operations {
                if !selected.matches_program(&self.program) {
                    return Err(ProviderExecutionError::from(anyhow!(
                        "selected Sigma source changed before SQRT"
                    ))
                    .after_effect());
                }
                let protected = std::iter::once(&self.input)
                    .chain(std::iter::once(&mean))
                    .chain(scratch.iter())
                    .collect::<Vec<_>>();
                selected
                    .execute_sqrt(
                        &scratch[sqrt_input],
                        &protected,
                        self.provider_metadata_permit.as_ref().map(|permit| {
                            permit.as_ref() as &dyn runmat_execution::resource::ProviderHostMetadataAdmission
                        }),
                    )
                    .await?
            } else {
                self.provider
                    .unary_sqrt(&scratch[sqrt_input])
                    .await
                    .map_err(|error| ProviderExecutionError::from(error).after_effect())?
            };
            let sigma = self.admit_next(sigma, &self.moment_shape(), &scratch, &[&mean])?;
            if let Some(selected) = selected_sigma {
                if !selected
                    .revalidate_raw(self.provider, &sigma)
                    .map_err(ProviderExecutionError::after_effect)?
                {
                    return Err(super::super::outputs::reject_outputs(
                        vec![sigma],
                        "selected Sigma raw output extent or owner changed",
                    ));
                }
            }
            let sigma = if let Some(publication) = self
                .program
                .moment_program
                .as_ref()
                .map(|program| &program.sigma)
            {
                if selected_sigma.is_some_and(|selected| {
                    selected.raw_source_identity().0 != publication.source_input
                        || selected.raw_source_identity().1 != publication.raw_output
                }) {
                    return Err(ProviderExecutionError::from(anyhow!(
                        "selected Sigma source identities changed"
                    ))
                    .after_effect());
                }
                self.publish_moment(
                    sigma,
                    publication,
                    &mut scratch,
                    &[&mean],
                    selected_sigma.map(|selected| selected.casts.as_slice()),
                )
                .await?
            } else {
                sigma
            };
            self.provenance(sigma)
        }
        .await;
        let result = Self::finish(result, scratch);
        match result {
            Ok(sigma) => {
                self.state = State::SigmaPublished {
                    mean,
                    sigma: sigma.clone(),
                };
                Ok(ImageSigma(sigma))
            }
            Err(error) => {
                self.state = State::Failed;
                Err(error)
            }
        }
    }

    /// Stage three produces only the final resident [B,H,W] image. The live
    /// moment handles are protected from aliasing and never retired here.
    #[cfg(test)]
    pub(crate) async fn output(&mut self) -> ProviderExecutionResult<ImageResult> {
        if self.program.output_program.is_some() || self.program.selected_powers.is_some() {
            return Err(anyhow!("selected outputs require checkpoint acknowledgement").into());
        }
        let State::SigmaPublished { mean, sigma } = &self.state else {
            return Err(anyhow!("image output checkpoint is out of order").into());
        };
        let (mean, sigma) = (mean.clone(), sigma.clone());
        self.state = State::InFlight;
        let mut scratch = Vec::new();
        let result = async {
            let centered = self
                .provider
                .elem_sub(&self.input, &mean)
                .await
                .map_err(ProviderExecutionError::after_effect)?;
            let centered =
                self.admit_next(centered, self.input.shape(), &scratch, &[&mean, &sigma])?;
            scratch.push(centered);
            let output = self
                .provider
                .elem_div(&scratch[0], &sigma)
                .await
                .map_err(ProviderExecutionError::after_effect)?;
            let output = self.admit_next(output, self.input.shape(), &scratch, &[&mean, &sigma])?;
            scratch.push(output);
            let powered = self
                .execute_stage_power(
                    &scratch[1],
                    None,
                    self.program.final_exponent_f32_bits,
                    &[&self.input, &mean, &sigma, &scratch[0], &scratch[1]],
                )
                .await?;
            let powered =
                self.admit_next(powered, self.input.shape(), &scratch, &[&mean, &sigma])?;
            self.provenance(powered)
        }
        .await;
        let result = Self::finish(result, scratch);
        match result {
            Ok(output) => {
                self.state = State::Complete;
                Ok(ImageResult(output))
            }
            Err(error) => {
                self.state = State::Failed;
                Err(error)
            }
        }
    }
}
