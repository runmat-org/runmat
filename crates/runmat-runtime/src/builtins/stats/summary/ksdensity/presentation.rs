use super::*;
#[cfg(feature = "plot-core")]
pub(super) type AxesTarget = crate::builtins::plotting::op_common::axes_target::AxesTarget;
#[cfg(not(feature = "plot-core"))]
pub(super) type AxesTarget = Option<()>;

pub(super) fn split(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<(AxesTarget, crate::call::BuiltinArguments)> {
    #[cfg(feature = "plot-core")]
    {
        // A real numeric singleton is legitimate sample data with an explicit
        // bandwidth, independent of any same-numbered legacy graphics handle.
        if matches!(
            args.first(),
            Some(Value::Num(_) | Value::Int(_) | Value::Tensor(_) | Value::GpuTensor(_))
        ) {
            return Ok((None, args));
        }
        crate::builtins::plotting::op_common::split_leading_axes_arguments(args, NAME)
    }
    #[cfg(not(feature = "plot-core"))]
    {
        Ok((None, args))
    }
}
pub(super) fn admit(count: usize, axes: &AxesTarget) -> BuiltinResult<()> {
    if count == 0 && !cfg!(feature = "plot-core") {
        return Err(invalid(
            NAME,
            "plotting requires plot-core; request numerical outputs instead",
        ));
    }
    if count > 0 && axes.is_some() {
        return Err(invalid(
            NAME,
            "the typed axes-target form is supported only for plotting without outputs",
        ));
    }
    Ok(())
}
pub(super) async fn show(
    axes: AxesTarget,
    points: Tensor,
    estimates: Tensor,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    #[cfg(feature = "plot-core")]
    {
        crate::builtins::plotting::op_common::apply_axes_target(axes, NAME)?;
        let values = vec![Value::Tensor(points), Value::Tensor(estimates)];
        let args = match runtime {
            Some(runtime) => crate::execution::RootedValues::in_context(values, runtime)?,
            None => crate::execution::RootedValues::new(values)
                .map_err(|error| error.into_runtime_error())?,
        };
        // Reuse the same exact, lazy domain continuation as ANOVA display.
        match runtime {
            Some(runtime) => {
                crate::execution::rooted_continuation::admit_domain_continuation_with(
                    runtime,
                    move || async move {
                        crate::builtins::plotting::plot::plot_builtin_registered(
                            Some(runtime.clone()),
                            args.into(),
                        )
                        .await?;
                        Ok(())
                    },
                )?
                .await?
            }
            None => {
                crate::execution::rooted_continuation::admit_optional_hostcall_with(
                    None,
                    |source| {
                        crate::context::host_allocation::host_allocation_runtime_error(
                            None,
                            "RunMat:ksdensity:Continuation",
                            "KDE plot continuation admission failed",
                            source,
                        )
                    },
                    move || async move {
                        crate::builtins::plotting::plot::plot_builtin_registered(None, args.into())
                            .await?;
                        Ok::<(), crate::RuntimeError>(())
                    },
                )?
                .await?
            }
        }
        Ok(())
    }
    #[cfg(not(feature = "plot-core"))]
    {
        let _ = (axes, points, estimates, runtime);
        Err(invalid(NAME, "plotting requires plot-core"))
    }
}
