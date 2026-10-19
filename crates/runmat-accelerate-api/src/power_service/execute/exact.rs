//! Certified scalar POWER execution and bounded result ownership.
use super::*;

pub(super) async fn execute(
    prepared: &PreparedResidentScalarPower,
    admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
) -> Result<ExecutedResidentScalarPower, PowerServiceError> {
    let resident = &prepared.resident;
    let request = super::super::symbolic::exact_single_base_request(
        &prepared.symbolic.resident,
        prepared.control,
        prepared.resident_operand,
        resident.physical().requested_bytes(),
    )
    .ok_or_else(|| PowerServiceError::pre("certified POWER facts changed"))?;
    let recipe = crate::ProviderRealElementwiseRecipe::plan(prepared.provider, request)
        .map_err(PowerServiceError)?
        .ok_or_else(|| PowerServiceError::pre("certified POWER recipe unavailable"))?;
    if Some(recipe.physical()) != prepared.symbolic.exact_proposal.as_ref() {
        return Err(PowerServiceError::pre(
            "certified POWER physical recipe changed",
        ));
    }
    let protected = prepared.protected_with(&[resident]);
    let output = recipe
        .execute_protected_admitted(resident, &protected, admission)
        .await
        .map_err(PowerServiceError)?
        .output;
    let mut owned = ExactResultGuard::new(output, &protected);
    if owned.current().shape() != prepared.output_shape.as_slice() {
        if let Err(error) = prepared.validate_live() {
            return Err(owned.fail(error));
        }
        let reshaped = match prepared
            .provider
            .reshape(owned.current(), &prepared.output_shape)
        {
            Ok(reshaped) => reshaped,
            Err(error) => return Err(owned.fail(PowerServiceError::after(error))),
        };
        let same_allocation = reshaped.shares_allocation_with(owned.current())
            && reshaped.physical() == owned.current().physical();
        owned.push(reshaped);
        if !same_allocation
            || !valid_output(
                owned.current(),
                prepared.provider,
                &prepared.output_shape,
                ProviderPrecision::F32,
                GpuTensorStorage::Real,
                &protected,
            )
        {
            return Err(owned.fail(PowerServiceError::pre("invalid certified POWER reshape")));
        }
    }
    if !valid_output(
        owned.current(),
        prepared.provider,
        &prepared.output_shape,
        ProviderPrecision::F32,
        GpuTensorStorage::Real,
        &protected,
    ) {
        return Err(owned.fail(PowerServiceError::pre("invalid certified POWER result")));
    }
    if let Err(error) = prepared.validate_live() {
        return Err(owned.fail(error));
    }
    let output = owned.transfer()?;
    // The shared publication helper consumes final ownership and explicitly
    // releases that reference, preserving cleanup errors, if provenance fails.
    Ok(ExecutedResidentScalarPower {
        handle: publish_provenance(output, resident)?,
        domain: PowerResultDomain::Real,
        scalar_evidence: None,
    })
}

/// One owner for every independently retained result reference. Operational
/// GpuTensorHandle aliases share an AllocationLease, never only numeric buffer
/// IDs (allocation/tensor.rs); releasing one view cannot revoke another lease.
struct ExactResultGuard<'a> {
    owned: [Option<GpuTensorHandle>; 2],
    borrowed: &'a [&'a GpuTensorHandle],
}

impl<'a> ExactResultGuard<'a> {
    fn new(output: GpuTensorHandle, borrowed: &'a [&'a GpuTensorHandle]) -> Self {
        Self {
            owned: [Some(output), None],
            borrowed,
        }
    }
    fn current(&self) -> &GpuTensorHandle {
        self.owned[1]
            .as_ref()
            .or(self.owned[0].as_ref())
            .expect("owned POWER output")
    }
    fn push(&mut self, output: GpuTensorHandle) {
        assert!(self.owned[1].is_none(), "at most one POWER reshape");
        self.owned[1] = Some(output);
    }
    fn fail(mut self, primary: PowerServiceError) -> PowerServiceError {
        exact_failure(primary, std::mem::take(&mut self.owned), self.borrowed)
    }
    fn transfer(mut self) -> Result<GpuTensorHandle, PowerServiceError> {
        let current = usize::from(self.owned[1].is_some());
        let output = self.owned[current].take().expect("owned POWER output");
        let cleanup = retire_exact(std::mem::take(&mut self.owned), self.borrowed);
        if cleanup.is_empty() {
            Ok(output)
        } else {
            Err(exact_failure(
                PowerServiceError(ProviderExecutionError::from_source(
                    ProviderExecutionCompletionError::new(None, cleanup),
                )),
                [Some(output), None],
                self.borrowed,
            ))
        }
    }
}

impl Drop for ExactResultGuard<'_> {
    fn drop(&mut self) {
        // No await follows acquisition of this guard; ordinary errors drain
        // it explicitly above. Unwind still relinquishes only owned leases.
        let _ = retire_exact(std::mem::take(&mut self.owned), self.borrowed);
    }
}

/// Only privately produced results are retireable. Borrowed aliases returned
/// by a faulty hook are dropped as references, never explicitly released.
fn exact_failure(
    primary: PowerServiceError,
    owned: [Option<GpuTensorHandle>; 2],
    borrowed: &[&GpuTensorHandle],
) -> PowerServiceError {
    let cleanup = retire_exact(owned, borrowed);
    let primary = primary.into_execution_error().after_effect();
    if cleanup.is_empty() {
        PowerServiceError(primary)
    } else {
        PowerServiceError(
            ProviderExecutionError::from_source(ProviderExecutionCompletionError::new(
                Some(primary),
                cleanup,
            ))
            .after_effect(),
        )
    }
}

fn retire_exact(
    owned: [Option<GpuTensorHandle>; 2],
    borrowed: &[&GpuTensorHandle],
) -> Vec<crate::allocation::GpuHandleReleaseError> {
    owned
        .into_iter()
        .rev()
        .flatten()
        .filter_map(|output| {
            if borrowed
                .iter()
                .any(|input| output.shares_allocation_with(input))
            {
                drop(output);
                None
            } else {
                output.release().err()
            }
        })
        .collect()
}
