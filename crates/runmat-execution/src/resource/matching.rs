use std::collections::BTreeSet;

use super::{
    AcceleratorAllocationDomainId, AcceleratorDevice, AcceleratorDeviceLease, AcceleratorRequest,
};

const MAX_ACCELERATOR_DEVICES_PER_REQUEST: usize = 16;

/// Selects a deterministic set of distinct devices that jointly satisfies all
/// accelerator requirements. Requirements may overlap; each selected device is
/// assigned to exactly one requested slot.
pub fn select_accelerator_devices(
    devices: &[AcceleratorDevice],
    requests: &[AcceleratorRequest],
    unavailable: &BTreeSet<AcceleratorAllocationDomainId>,
) -> Option<Vec<AcceleratorDevice>> {
    let needs = expanded_requirements(requests)?;
    let mut candidates = devices
        .iter()
        .filter(|device| !unavailable.contains(&device.allocation_domain))
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.id.cmp(&right.id));
    let selected = exact_matching_with_selection(
        &needs,
        candidates.len(),
        |need, candidate| need.matches(candidates[candidate]),
        |candidate, selected| {
            selected.iter().all(|selected| {
                candidates[*selected].allocation_domain != candidates[candidate].allocation_domain
            })
        },
    )?;
    let mut devices = selected
        .into_iter()
        .map(|index| candidates[index].clone())
        .collect::<Vec<_>>();
    devices.sort_by(|left, right| left.id.cmp(&right.id));
    Some(devices)
}

/// Returns whether the supplied distinct devices are an exact realization of
/// the accelerator requirements, including overlapping requirements.
pub fn accelerator_devices_exactly_satisfy(
    devices: &[AcceleratorDevice],
    requests: &[AcceleratorRequest],
) -> bool {
    if devices.windows(2).any(|pair| pair[0].id >= pair[1].id)
        || devices
            .iter()
            .map(|device| &device.allocation_domain)
            .collect::<BTreeSet<_>>()
            .len()
            != devices.len()
    {
        return false;
    }
    let Some(needs) = expanded_requirements(requests) else {
        return false;
    };
    needs.len() == devices.len()
        && exact_matching(&needs, devices.len(), |need, candidate| {
            need.matches(&devices[candidate])
        })
        .is_some()
}

/// Returns whether scheduler-facing requests are at least as restrictive as
/// immutable program requirements. This compares contracts, not a particular
/// inventory: every device eligible for a matched request must also satisfy
/// the program requirement.
pub fn accelerator_requests_satisfy_requirements(
    requests: &[AcceleratorRequest],
    requirements: &[AcceleratorRequest],
) -> bool {
    let Some(request_slots) = expanded_requirements(requests) else {
        return false;
    };
    let Some(required_slots) = expanded_requirements(requirements) else {
        return false;
    };
    if required_slots.len() > request_slots.len() {
        return false;
    }
    exact_matching(
        &required_slots,
        request_slots.len(),
        |required, candidate| request_implies_requirement(request_slots[candidate], required),
    )
    .is_some()
}

/// Returns whether a concrete reservation request stays within an enclosing
/// accelerator budget. It may request fewer slots or strengthen a slot's
/// provider, memory, or feature constraint, but cannot broaden one.
pub fn accelerator_request_is_within(
    request: &[AcceleratorRequest],
    limit: &[AcceleratorRequest],
) -> bool {
    let Some(request_slots) = expanded_requirements(request) else {
        return false;
    };
    let Some(limit_slots) = expanded_requirements(limit) else {
        return false;
    };
    if request_slots.len() > limit_slots.len() {
        return false;
    }
    exact_matching(&request_slots, limit_slots.len(), |requested, candidate| {
        request_implies_requirement(requested, limit_slots[candidate])
    })
    .is_some()
}

pub(crate) fn accelerator_leases_exactly_satisfy(
    leases: &[AcceleratorDeviceLease],
    requests: &[AcceleratorRequest],
) -> bool {
    let Some(needs) = expanded_requirements(requests) else {
        return false;
    };
    needs.len() == leases.len()
        && exact_matching(&needs, leases.len(), |need, candidate| {
            let lease = &leases[candidate];
            need.class == lease.class
                && need.minimum_allocation_bytes <= lease.max_allocation_bytes
                && need
                    .provider
                    .as_ref()
                    .is_none_or(|provider| provider == &lease.provider)
                && need.required_features.is_subset(&lease.features)
        })
        .is_some()
}

fn expanded_requirements(requests: &[AcceleratorRequest]) -> Option<Vec<&AcceleratorRequest>> {
    if requests.iter().any(|request| request.validate().is_err()) {
        return None;
    }
    let count = requests.iter().try_fold(0usize, |total, request| {
        total.checked_add(usize::from(request.count))
    })?;
    if count > MAX_ACCELERATOR_DEVICES_PER_REQUEST {
        return None;
    }
    Some(
        requests
            .iter()
            .flat_map(|request| std::iter::repeat_n(request, usize::from(request.count)))
            .collect(),
    )
}

fn request_implies_requirement(
    request: &AcceleratorRequest,
    requirement: &AcceleratorRequest,
) -> bool {
    request.class == requirement.class
        && request.minimum_allocation_bytes >= requirement.minimum_allocation_bytes
        && match &requirement.provider {
            Some(required) => request.provider.as_ref() == Some(required),
            None => true,
        }
        && requirement
            .required_features
            .is_subset(&request.required_features)
}

fn exact_matching<T>(
    needs: &[T],
    candidate_count: usize,
    matches: impl Fn(&T, usize) -> bool,
) -> Option<Vec<usize>> {
    exact_matching_with_selection(needs, candidate_count, matches, |_, _| true)
}

fn exact_matching_with_selection<T>(
    needs: &[T],
    candidate_count: usize,
    matches: impl Fn(&T, usize) -> bool,
    can_select: impl Fn(usize, &[usize]) -> bool,
) -> Option<Vec<usize>> {
    let remaining = (0..needs.len()).collect::<Vec<_>>();
    let mut used = vec![false; candidate_count];
    let mut selected = Vec::with_capacity(needs.len());
    match_remaining(
        needs,
        &remaining,
        &mut used,
        &mut selected,
        &matches,
        &can_select,
    )
    .then_some(selected)
}

fn match_remaining<T>(
    needs: &[T],
    remaining: &[usize],
    used: &mut [bool],
    selected: &mut Vec<usize>,
    matches: &impl Fn(&T, usize) -> bool,
    can_select: &impl Fn(usize, &[usize]) -> bool,
) -> bool {
    if remaining.is_empty() {
        return true;
    }
    let (remaining_index, &need_index) = remaining
        .iter()
        .enumerate()
        .min_by_key(|(_, need_index)| {
            used.iter()
                .enumerate()
                .filter(|(candidate, in_use)| {
                    !**in_use && matches(&needs[**need_index], *candidate)
                })
                .count()
        })
        .expect("non-empty accelerator requirement set");
    let mut next = remaining.to_vec();
    next.remove(remaining_index);
    for candidate in 0..used.len() {
        if used[candidate]
            || !matches(&needs[need_index], candidate)
            || !can_select(candidate, selected)
        {
            continue;
        }
        used[candidate] = true;
        selected.push(candidate);
        if match_remaining(needs, &next, used, selected, matches, can_select) {
            return true;
        }
        selected.pop();
        used[candidate] = false;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{
        AcceleratorClass, AcceleratorFeature, AcceleratorProvider, AcceleratorProviderId,
        AcceleratorProviderVersion,
    };
    use crate::Digest;

    fn request(
        provider: Option<AcceleratorProvider>,
        minimum_allocation_bytes: u64,
        features: impl IntoIterator<Item = AcceleratorFeature>,
    ) -> AcceleratorRequest {
        AcceleratorRequest {
            class: AcceleratorClass::new("gpu").unwrap(),
            count: 1,
            minimum_allocation_bytes,
            provider,
            required_features: features.into_iter().collect(),
        }
    }

    fn provider() -> AcceleratorProvider {
        AcceleratorProvider {
            id: AcceleratorProviderId::new("runmat.test").unwrap(),
            version: AcceleratorProviderVersion::new("1.0.0").unwrap(),
            abi_fingerprint: Digest::sha256(b"stable-provider-abi"),
        }
    }

    #[test]
    fn scheduler_requests_may_strengthen_but_not_weaken_program_requirements() {
        let required = request(Some(provider()), 1024, [AcceleratorFeature::Compute]);
        let broad = request(None, 1024, [AcceleratorFeature::Compute]);
        assert!(!accelerator_requests_satisfy_requirements(
            &[broad],
            std::slice::from_ref(&required)
        ));

        let exact = request(
            Some(provider()),
            2048,
            [
                AcceleratorFeature::Compute,
                AcceleratorFeature::UnifiedMemory,
            ],
        );
        assert!(accelerator_requests_satisfy_requirements(
            std::slice::from_ref(&exact),
            std::slice::from_ref(&required)
        ));

        let mut extra = exact;
        extra.count = 2;
        assert!(accelerator_requests_satisfy_requirements(
            &[extra],
            &[required]
        ));
    }

    #[test]
    fn reservation_may_use_fewer_stricter_slots_without_broadening_its_limit() {
        let broad = request(None, 1024, [AcceleratorFeature::Compute]);
        let exact = request(Some(provider()), 2048, [AcceleratorFeature::Compute]);
        let mut two_broad = broad.clone();
        two_broad.count = 2;
        assert!(accelerator_request_is_within(
            std::slice::from_ref(&exact),
            std::slice::from_ref(&two_broad)
        ));
        assert!(!accelerator_request_is_within(&[two_broad], &[exact]));
    }
}
