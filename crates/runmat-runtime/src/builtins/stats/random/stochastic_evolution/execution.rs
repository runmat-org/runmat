//! Ordered host phase of an already selected stochastic source call.
//! Selection and complete-route resource admission belong to the parent
//! coordinator; this child never mints a candidate or replays a failed stage.

use super::host::{compute_state, HostSourceValue};
use super::host_readback::StochasticHostReadback;
use super::source_eligibility::{
    StochasticCheckpoint, StochasticPreparedSource, StochasticSourceExecution,
};
use crate::builtins::array::creation::randn::draw_source_normal;
use crate::builtins::common::random::{self, gpu_like::ExactHostRandomMode, RngAlgorithm};
use crate::context::host_presentation::HostSinkReceipt;
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry, RuntimeError};

fn invalid_prepared_class() -> RuntimeError {
    build_runtime_error("selected stochastic source lost its real noise class")
        .with_identifier("RunMat:StochasticSource:InvalidPlan")
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}

/// Executes only a Host route already chosen by the whole-call selector. A
/// managed completion receipt, when present, stays owned until every source
/// checkpoint and arithmetic stage has settled. An unbounded Host route has
/// no placement feedback receipt; neither case implies a provider grant.
pub(super) async fn execute_selected_host_source(
    context: &RuntimeContext,
    mut prepared: StochasticPreparedSource,
    mut readback: Option<StochasticHostReadback>,
    mode: ExactHostRandomMode,
    selected_algorithm: RngAlgorithm,
    mut completion: Option<HostSinkReceipt>,
    publish: &mut impl FnMut(StochasticCheckpoint) -> BuiltinResult<()>,
) -> BuiltinResult<StochasticSourceExecution> {
    StochasticPreparedSource::checkpoint_receipt(prepared.receipt(), context)?;
    let mut turn = random::acquire_turn(Some(context), "stochastic_evolution").await?;
    StochasticPreparedSource::checkpoint_receipt(prepared.receipt(), context)?;
    // A competing ordered source operation may have changed the algorithm
    // while this call waited for the FIFO turn. No RNG draw, source store, or
    // provider work has happened yet, so the original For may resume.
    if random::normal_source_algorithm_in_turn(&mut turn)? != selected_algorithm {
        return Ok(StochasticSourceExecution::Declined);
    }

    let (initial_state, pattern, preparation_receipt) = prepared.parts_mut();
    let noise_dtype = match pattern.noise_class() {
        runmat_types::NumericClass::Single => runmat_value::NumericDType::F32,
        runmat_types::NumericClass::Double => runmat_value::NumericDType::F64,
        _ => return Err(invalid_prepared_class()),
    };
    // Drop order is significant on cancellation: the borrowed initial
    // numeric backing (current) retires before its metadata reservation.
    let mut readback_retention = None;
    let mut current = if readback.is_some() {
        if !matches!(initial_state, runmat_value::Value::GpuTensor(_)) {
            return Err(invalid_prepared_class());
        }
        None
    } else {
        let retained =
            HostSourceValue::capture(initial_state, pattern.state_class(), pattern.shape())?;
        // Keep only the shared numeric backing after moving the original
        // Value to the VM's source publication owner.
        *initial_state = runmat_value::Value::Num(0.0);
        Some(retained)
    };
    let mut after_effect = false;
    let result = async {
        for step in 0..pattern.steps() {
            StochasticPreparedSource::checkpoint_receipt(preparation_receipt, context)?;
            publish(StochasticCheckpoint::IterationBinding(f64::from(step + 1)))?;
            after_effect = true;

            // Exactly one normal source block per logical iteration. The
            // caller owns the same turn for the full loop, including yields.
            let noise = draw_source_normal(&mut turn, pattern.shape(), noise_dtype, mode).await?;
            StochasticPreparedSource::checkpoint_receipt(preparation_receipt, context)?;
            let retained_noise =
                HostSourceValue::capture(&noise, pattern.noise_class(), pattern.shape())?;
            publish(StochasticCheckpoint::Noise(noise))?;

            if current.is_none() {
                let plan = readback.take().ok_or_else(invalid_prepared_class)?;
                let source = std::mem::replace(initial_state, runmat_value::Value::Num(0.0));
                let runmat_value::Value::GpuTensor(handle) = source else {
                    return Err(invalid_prepared_class());
                };
                // Transfer follows t and the complete logical Z store, so a
                // failure leaves the same visible prefix as the source loop.
                let (downloaded, receipt) = plan.download(context, &handle).await?;
                readback_retention = Some(receipt);
                current = Some(HostSourceValue::capture(
                    &downloaded,
                    pattern.state_class(),
                    pattern.shape(),
                )?);
            }

            let next_state = compute_state(
                context,
                current.as_ref().ok_or_else(invalid_prepared_class)?,
                &retained_noise,
                pattern,
                step,
            )
            .await?;
            StochasticPreparedSource::checkpoint_receipt(preparation_receipt, context)?;
            let retained_next =
                HostSourceValue::capture(&next_state, pattern.result_class(), pattern.shape())?;
            publish(StochasticCheckpoint::State(next_state))?;
            current = Some(retained_next);
            // The initial readback backing is no longer referenced after S
            // replaces it; only now may its metadata funding retire.
            drop(readback_retention.take());
        }
        StochasticPreparedSource::checkpoint_receipt(preparation_receipt, context)?;
        Ok(())
    }
    .await;
    drop(turn);
    match result {
        Ok(()) => {
            if let Some(receipt) = completion.take() {
                receipt.finish(true);
            }
            Ok(StochasticSourceExecution::Completed)
        }
        Err(error) => {
            if after_effect {
                if let Some(receipt) = completion.take() {
                    receipt.finish(false);
                }
            }
            Err(error)
        }
    }
}
