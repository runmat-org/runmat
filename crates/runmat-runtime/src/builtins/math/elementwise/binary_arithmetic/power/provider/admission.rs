//! Side-effect-free standalone execution admission, separate from fusion's
//! classification-only hook. Unadvertised legacy hooks keep their old route.
use runmat_accelerate_api::*;

pub(in super::super) fn representation(
    handle: &GpuTensorHandle,
) -> anyhow::Result<ProviderRepresentation> {
    let precision = handle_precision(handle)
        .ok_or_else(|| anyhow::anyhow!("power operand precision metadata is missing"))?;
    Ok(ProviderRepresentation {
        element_type: precision.into(),
        storage: ProviderStorage::DenseReal,
        layout: if handle_is_transposed(handle) {
            ProviderLayout::Strided
        } else {
            ProviderLayout::ColumnMajorContiguous
        },
        shape: handle
            .shape()
            .iter()
            .map(|extent| u64::try_from(*extent))
            .collect::<Result<_, _>>()?,
        residency: ProviderResidency::Device,
    })
}

pub(super) fn may_execute(
    provider: &dyn AccelProvider,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
) -> anyhow::Result<bool> {
    let operation =
        ProviderOperationIdentity::new(runmat_builtins::POWER_CATALOG_ENTRY.identity.name);
    if !provider
        .capability_snapshot()
        .operations
        .iter()
        .any(|candidate| {
            candidate.identity == operation
                && candidate.family == ProviderOperationFamily::Elementwise
        })
    {
        return Ok(true);
    }
    let mut query = ProviderFeasibilityQuery {
        operation,
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![representation(lhs)?, representation(rhs)?],
        outputs: Vec::new(),
        workload: ProviderWorkload {
            real_power_mode: match handle_precision(lhs) {
                Some(ProviderPrecision::F32) => Some(RealPowerArithmeticMode::NativeSingle),
                Some(ProviderPrecision::F64) => Some(RealPowerArithmeticMode::NativeDouble),
                None => None,
            },
            ..ProviderWorkload::default()
        },
    };
    Ok(any_domain_supported(provider, &mut query))
}

fn any_domain_supported(
    provider: &dyn AccelProvider,
    query: &mut ProviderFeasibilityQuery,
) -> bool {
    // Hypothetical outcomes are capability queries, not evidence about values.
    // If either outcome executes, classification remains necessary.
    for domain in [PowerResultDomain::Real, PowerResultDomain::Complex] {
        query.workload.numerical_domain = NumericalDomainEvidence::RealPower(domain);
        if provider.query_feasibility(query).is_supported() {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests;
