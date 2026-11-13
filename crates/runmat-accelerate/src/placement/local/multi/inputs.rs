//! Source and checked-route validation before feasibility, cache or effects.
use super::*;
use runmat_runtime::builtins::common::gpu_helpers::exact_provider_for_handle;

pub(super) struct SourcePreflight {
    pub(super) digest: Digest,
    pub(super) routes: Vec<Option<ProviderPreparationRoute>>,
}

pub(super) fn validate_set(
    provider: &dyn AccelProvider,
    operation: &str,
    physical: &[LocalPhysicalCandidate<'_>],
) -> Result<SourcePreflight, Box<runmat_runtime::RuntimeError>> {
    if physical.is_empty() {
        return Err(Box::new(
            runmat_runtime::build_runtime_error("placement requires a physical candidate record")
                .with_identifier("RunMat:Placement:InvalidRequest")
                .build(),
        ));
    }
    let canonical_inputs = physical[0].data_inputs;
    if physical.iter().any(|candidate| {
        candidate.data_inputs.len() != canonical_inputs.len()
            || candidate
                .data_inputs
                .iter()
                .zip(canonical_inputs)
                .any(|(left, right)| !std::ptr::eq(*left, *right))
            || (candidate.physical_identity.is_some()
                && (candidate.source_inputs.is_none() || candidate.prepared_inputs.is_none()))
            || candidate
                .prepared_inputs
                .as_ref()
                .is_some_and(|prepared| prepared.len() != candidate.data_inputs.len())
            || candidate
                .workload
                .typed_reduction
                .as_ref()
                .is_some_and(|typed| {
                    candidate.physical_identity != Some(typed.selection_identity)
                        || candidate.prepared_inputs.as_ref() != Some(&typed.inputs)
                        || candidate.outputs.as_slice() != std::slice::from_ref(&typed.output)
                })
            || matches!(candidate.preparation_program, Some(
                ProviderPreparationProgram::TypedReduction { selection, .. }
            ) if candidate.physical_identity != Some(selection.identity())
                || candidate.workload.typed_reduction.as_ref().is_none_or(|typed|
                    !typed.matches_selection(selection)))
    }) {
        return Err(Box::new(runmat_runtime::build_runtime_error(
            "physical candidates must share exact ordered data leaves and explicit typed transfer facts",
        ).with_identifier("RunMat:Placement:InvalidRequest").build()));
    }
    if physical.iter().any(|candidate| {
        candidate.authority.is_some_and(|authority| match authority {
            super::PreparedPhysicalAuthority::Compound(plan) => {
                candidate.physical_identity != Some(plan.identity())
                    || !plan.matches(provider, candidate.data_inputs)
            }
            super::PreparedPhysicalAuthority::ImageStages { plan, .. } => {
                candidate.physical_identity != Some(plan.semantic_identity())
                    || candidate.data_inputs.len() != 1
                    || !matches!(candidate.data_inputs[0], Value::GpuTensor(handle) if handle.view_identity() == plan.input_view_identity())
                    || !matches!(plan.matches_pre_effect_snapshot(provider), Ok(true))
            }
            super::PreparedPhysicalAuthority::ImageHostStages(plan) => {
                candidate.physical_identity != Some(plan.semantic_identity())
                    || candidate.data_inputs.len() != 1
                    || !matches!(
                        plan.matches_source_and_provider(provider, candidate.data_inputs[0]),
                        Ok(true)
                    )
                    || candidate.source_inputs.as_deref()
                        != Some(std::slice::from_ref(&plan.source_representation()))
                    || candidate.prepared_inputs.as_deref()
                        != Some(std::slice::from_ref(plan.prepared_representation()))
                    || candidate.input_adaptation != Some(plan.preparation_route())
                    || candidate.required_upload_bytes != Some(plan.upload_bytes())
            }
        })
    }) {
        return Err(Box::new(
            runmat_runtime::build_runtime_error(
                "compound candidate lost its admitted source/recipe",
            )
            .with_identifier("RunMat:Placement:InvalidRequest")
            .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
            .build(),
        ));
    }
    let duplicate_name = if physical.len() == 1 {
        false
    } else {
        let mut names = std::collections::BTreeSet::new();
        physical.iter().any(|candidate| {
            !names.insert(candidate_name(
                provider,
                operation,
                candidate.physical_identity,
            ))
        })
    };
    if duplicate_name {
        return Err(Box::new(
            runmat_runtime::build_runtime_error("duplicate physical candidate identity")
                .with_identifier("RunMat:Placement:InvalidRequest")
                .build(),
        ));
    }
    // Validate the entire set before any provider feasibility, preparation
    // snapshot, cost query or service graph can observe a partial set.
    let mut routes = Vec::new();
    routes.try_reserve_exact(physical.len()).map_err(|_| {
        Box::new(
            runmat_runtime::build_runtime_error("physical route allocation failed")
                .with_identifier("RunMat:Placement:ResourceOverflow")
                .build(),
        )
    })?;
    for candidate in physical {
        routes.push(validate_candidate(provider, candidate)?);
    }
    let mut source_facts = Vec::new();
    source_facts
        .try_reserve_exact(physical.len())
        .map_err(|_| {
            Box::new(
                runmat_runtime::build_runtime_error("physical source fact allocation failed")
                    .with_identifier("RunMat:Placement:ResourceOverflow")
                    .build(),
            )
        })?;
    source_facts.extend(physical.iter().map(|candidate| {
        (
            candidate.physical_identity,
            candidate.source_inputs.as_ref(),
            candidate.prepared_inputs.as_ref(),
            candidate.input_adaptation,
            candidate.required_upload_bytes,
        )
    }));
    source_facts.sort_by_key(|record| record.0);
    let source_digest = Digest::sha256_serialized(&("local-physical-sources-v1", source_facts))
        .expect("physical source facts serialize");
    Ok(SourcePreflight {
        digest: source_digest,
        routes,
    })
}

pub(super) fn validate_candidate(
    provider: &dyn AccelProvider,
    candidate: &LocalPhysicalCandidate<'_>,
) -> Result<Option<ProviderPreparationRoute>, Box<runmat_runtime::RuntimeError>> {
    if let Some(source) = candidate.source_inputs.as_ref() {
        let actual: Option<Vec<_>> = candidate
            .data_inputs
            .iter()
            .map(|value| tensor_representation(value, provider.precision()))
            .collect();
        if actual.as_ref() != Some(source) {
            return Err(Box::new(
                runmat_runtime::build_runtime_error(
                    "physical candidate source representation differs from checked leaf",
                )
                .with_identifier("RunMat:Placement:InvalidRequest")
                .build(),
            ));
        }
    }
    let typed_route = candidate.workload.typed_reduction.is_some()
        || matches!(
            candidate.preparation_program,
            Some(ProviderPreparationProgram::TypedReduction { .. })
        );
    if typed_route {
        for value in candidate.data_inputs {
            let Value::GpuTensor(handle) = *value else {
                continue;
            };
            // Representation projection deliberately omits device identity.
            // Check the live lease and exact routing owner before any probe;
            // the backend still authenticates the physical allocation.
            if handle.device_id() != provider.device_id()
                || handle.allocation_lease().validate_live().is_err()
                || handle.view().validate(handle.physical()).is_err()
                || !exact_provider_for_handle(handle).is_some_and(|owner| {
                    runmat_accelerate_api::same_provider_instance(owner, provider)
                })
            {
                return Err(Box::new(
                    runmat_runtime::build_runtime_error(
                        "typed reduction resident input has no live selected provider owner",
                    )
                    .with_identifier("RunMat:Placement:InvalidRequest")
                    .build(),
                ));
            }
        }
    }
    let has_adaptation = candidate.source_inputs != candidate.prepared_inputs;
    let typed_inventory = candidate
        .workload
        .typed_reduction
        .as_ref()
        .and_then(|typed| typed.input_preparation.as_ref());
    if let Some(inputs) = typed_inventory {
        let resources =
            crate::placement::typed_reduction_input_resources(provider.device_id(), inputs)
                .ok_or_else(|| invalid_request("typed reduction input resources are unbounded"))?;
        let sources: Vec<_> = inputs
            .entries()
            .iter()
            .map(|entry| entry.source.clone())
            .collect();
        let prepared: Vec<_> = inputs
            .entries()
            .iter()
            .map(|entry| entry.prepared.clone())
            .collect();
        if candidate.source_inputs.as_ref() != Some(&sources)
            || candidate.prepared_inputs.as_ref() != Some(&prepared)
            || candidate.input_adaptation != Some(resources.route)
            || candidate.required_upload_bytes != Some(resources.required_upload_bytes)
        {
            return Err(invalid_request(
                "typed reduction input route differs from checked inventory",
            ));
        }
    } else if has_adaptation && typed_route {
        return Err(invalid_request(
            "typed reduction input adaptation has no admitted inventory",
        ));
    }
    if has_adaptation
        && (candidate.source_inputs.is_none()
            || candidate.prepared_inputs.is_none()
            || candidate.input_adaptation.is_none()
            || candidate.required_upload_bytes.is_none())
    {
        return Err(Box::new(
                runmat_runtime::build_runtime_error(
                    "changed provider inputs require an admitted preparation route and exact transfer bytes",
                )
                .with_identifier("RunMat:Placement:InvalidRequest")
                .build(),
            ));
    }
    if candidate.input_adaptation.is_some() && !has_adaptation && typed_inventory.is_none() {
        return Err(Box::new(
            runmat_runtime::build_runtime_error(
                "input preparation route has no source-to-physical change",
            )
            .with_identifier("RunMat:Placement:InvalidRequest")
            .build(),
        ));
    }
    let program_route = candidate
        .preparation_program
        .and_then(|program| program.route());
    if candidate.input_adaptation.is_some()
        && program_route.is_some()
        && candidate.input_adaptation != program_route
    {
        return Err(Box::new(
            runmat_runtime::build_runtime_error("conflicting provider preparation routes")
                .with_identifier("RunMat:Placement:InvalidRequest")
                .build(),
        ));
    }
    Ok(candidate.input_adaptation.or(program_route))
}

fn invalid_request(message: &'static str) -> Box<runmat_runtime::RuntimeError> {
    Box::new(
        runmat_runtime::build_runtime_error(message)
            .with_identifier("RunMat:Placement:InvalidRequest")
            .build(),
    )
}
