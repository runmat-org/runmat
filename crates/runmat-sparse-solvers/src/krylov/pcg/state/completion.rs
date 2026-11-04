//! Terminal numerical outcomes retain the best iterate and its residual.
use super::*;

impl<T: Copy + PartialEq, A: Arithmetic<T>> RecurrenceState<'_, T, A> {
    pub(super) async fn complete_best<C: PcgOperations<T>>(
        &mut self,
        flag: u8,
        iteration: Iteration,
        residual_count: usize,
        checkpoint: &mut C,
    ) -> Result<(), KrylovFailure<C::Error>> {
        let Some(certificate) = self.best_certificate else {
            self.phase = Phase::BestProduct(flag, iteration, residual_count);
            return Ok(());
        };
        copy(&mut *self.work.x, &*self.work.best, &mut *checkpoint).await?;
        let relres =
            measured_relative_norm::<T, C>(certificate, iteration.rhs_squared, checkpoint).await?;
        self.phase = Phase::Complete(PcgOutcome {
            flag: if relres <= self.tolerance { 0 } else { flag },
            relres,
            iterations: iteration.best_iteration,
            residual_count,
        });
        Ok(())
    }

    pub(super) async fn complete_preconditioner_failure<C: PcgOperations<T>>(
        &mut self,
        checkpoint: &mut C,
    ) -> Result<(), KrylovFailure<C::Error>> {
        let stage = match self.phase {
            Phase::FirstPreconditioner(stage) | Phase::SecondPreconditioner(stage, _) => stage,
            _ => {
                return Err(invalid(
                    "PCG preconditioner failure outside a preconditioner phase",
                ))
            }
        };
        let (iteration, residual_count) = match stage {
            PreconditionerStage::Seed(seed) => (
                Iteration {
                    rhs_squared: seed.rhs_squared,
                    rho: seed.best_squared,
                    best_squared: seed.best_squared,
                    best_iteration: 0,
                    index: 0,
                },
                1,
            ),
            PreconditionerStage::Following { prior } | PreconditionerStage::Restart { prior } => {
                (prior, prior.index + 2)
            }
        };
        self.complete_best(2, iteration, residual_count, checkpoint)
            .await
    }

    pub(super) async fn true_residual<C: PcgOperations<T>>(
        &mut self,
        product: &[T],
        checkpoint: &mut C,
    ) -> Result<ScaledPositive, KrylovFailure<C::Error>> {
        for start in (0..self.rhs.len()).step_by(1024) {
            let end = (start + 1024).min(self.rhs.len());
            checkpoint
                .cooperate((end - start) as u64)
                .await
                .map_err(KrylovFailure::Operation)?;
            checkpoint
                .measured(
                    VectorWork::Residual {
                        elements: end - start,
                        domain: A::DOMAIN,
                    },
                    |_| {
                        for (offset, product) in product[start..end].iter().enumerate() {
                            let index = start + offset;
                            self.work.residual[index] = self.rhs[index];
                            A::subtract_scaled(&mut self.work.residual[index], 1.0, *product);
                        }
                        ((), BatchCompletion::Complete)
                    },
                )
                .await
                .map_err(KrylovFailure::Operation)?;
        }
        dot::<T, A, C>(&*self.work.residual, &*self.work.residual, checkpoint)
            .await?
            .positive()
            .ok_or_else(|| invalid("PCG true residual norm is invalid"))
    }

    pub(super) async fn after_candidate_product<C: PcgOperations<T>>(
        &mut self,
        mut iteration: Iteration,
        progressed: bool,
        product: &[T],
        checkpoint: &mut C,
    ) -> Result<(), KrylovFailure<C::Error>> {
        let actual = self.true_residual(product, &mut *checkpoint).await?;
        let relres =
            measured_relative_norm::<T, C>(actual, iteration.rhs_squared, checkpoint).await?;
        // A reliable residual replacement supersedes the optimistic recursive
        // value at this iteration, not the historical values of earlier steps.
        self.work.residual_norms[iteration.index + 1] =
            measured_norm::<T, C>(actual, checkpoint).await?;
        if actual.less_than(iteration.best_squared) {
            iteration.best_squared = actual;
            iteration.best_iteration = iteration.index + 1;
            copy(&mut *self.work.best, &*self.work.x, &mut *checkpoint).await?;
            self.best_certificate = Some(actual);
        }
        if relres <= self.tolerance {
            self.phase = Phase::Complete(PcgOutcome {
                flag: 0,
                relres,
                iterations: iteration.index + 1,
                residual_count: iteration.index + 2,
            });
            return Ok(());
        }
        if !progressed || iteration.index + 1 == self.maximum_iterations {
            return self
                .complete_best(
                    if progressed { 1 } else { 3 },
                    iteration,
                    iteration.index + 2,
                    checkpoint,
                )
                .await;
        }
        // The recursive residual was too optimistic. Restart from the actual
        // residual and request each preconditioner once, in its existing order.
        let stage = PreconditionerStage::Restart { prior: iteration };
        self.phase = if self.first_preconditioner {
            Phase::FirstPreconditioner(stage)
        } else if self.second_preconditioner {
            Phase::SecondPreconditioner(stage, SecondInput::Residual)
        } else {
            self.restart_coefficients();
            copy(&mut *self.work.direction, &*self.work.residual, checkpoint).await?;
            Phase::DirectionProduct(Iteration {
                rho: actual,
                index: iteration.index + 1,
                ..iteration
            })
        };
        Ok(())
    }
}
