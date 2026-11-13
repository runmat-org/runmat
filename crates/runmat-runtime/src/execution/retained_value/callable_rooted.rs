//! Retain an admitted callable header without registering its GC graph again.
use super::*;

impl RetainedValueReadLease {
    pub(crate) fn prepare_callable_rooted(
        rooted: crate::execution::RootedValue,
        context: Option<&RuntimeContext>,
    ) -> Result<Self, RuntimeError> {
        let (value, roots) = rooted.into_value_and_lease();
        let error = |source| {
            host_allocation_runtime_error(
                context,
                "RunMat:Callable:RepeatableSelection",
                "retained callable admission failed",
                source,
            )
        };
        runmat_value::validate_no_transient_sequence(&value).map_err(|source| {
            semantic_error_with_source(
                "RunMat:Callable:RepeatableSelection",
                "transient retained callable",
                source,
            )
        })?;
        let quote = if let Value::ScopedFunctionHandle(handle) = &value {
            crate::execution::scoped_callable::validate_retention(context, handle)?;
            // The scope's queue/capability metadata has no finite allocation
            // certificate. Preserve the original rooted header only after the
            // invocation owner has admitted the unmanaged path.
            None
        } else {
            SharedValueGeneration::callable_quote(&value).map_err(|e| error(e.into()))?
        };
        let (bytes, claims) = quote
            .map(|quote| (quote.requested_bytes, quote.retained_claims))
            .unwrap_or((SharedValueGeneration::requested_bytes(), 1));
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(bytes).map_err(|_| error(HostAllocationError::Overflow))?,
            claims,
        )
        .map_err(error)?;
        let receipt = HostAllocationReceipt::admit(
            context,
            HostAllocationPlan::new(0, retained).map_err(error)?,
        )
        .map_err(error)?;
        let generation = if let Some(receipt) = &receipt {
            let context = context.expect("managed callable has context");
            let permit = receipt.output_permit(context).map_err(error)?;
            if quote.is_some() {
                let mut input = CallableGenerationInput::new(value);
                let generation = SharedValueGeneration::prepare_callable(&mut input, &permit);
                if let Err(source) = receipt.checkpoint(context) {
                    drop(roots);
                    drop(generation);
                    drop(input);
                    return Err(error(source));
                }
                match generation {
                    Ok(generation) => {
                        drop(input);
                        generation
                    }
                    Err(source) => {
                        drop(roots);
                        drop(input);
                        return Err(callable_prepare_error(context, source));
                    }
                }
            } else {
                // Text/Object targets keep their original canonical payload;
                // this claim is solely for the newly retained Value header.
                let result = SharedValueGeneration::prepare(value, &permit);
                if let Err(source) = receipt.checkpoint(context) {
                    drop(roots);
                    drop(result);
                    return Err(error(source));
                }
                result.map_err(|source| error(source.into()))?
            }
        } else {
            SharedValueGeneration::unbounded(value)
        };
        Ok(Self {
            roots: roots.shared_handles(),
            generation,
        })
    }
}
