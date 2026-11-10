use super::*;

pub(in crate::analysis_kernel) fn execute_observation_partition(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let source_request = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::ObservationEvaluationRequest,
    )?;
    let request = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::ObservationPartitionEvaluationRequest,
    )?;
    let partition_plan = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::ObservationPartitionPlan,
    )?;
    let fields = unique_input(&invocation, AnalysisNumericalObjectKind::FieldPartition)?;
    let field_catalog = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::ObservationFieldCatalog,
    )?;
    let nonlinear_workload = optional_unique_input(
        &invocation,
        AnalysisNumericalObjectKind::NonlinearSolidWorkload,
    )?;
    let linear_workload =
        optional_unique_input(&invocation, AnalysisNumericalObjectKind::SolverInvocation)?;
    let frequency_workload = optional_unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralFrequencyWorkload,
    )?;
    let thermal_workload =
        optional_unique_input(&invocation, AnalysisNumericalObjectKind::ThermalWorkload)?;
    let magnetic_workload = optional_unique_input(
        &invocation,
        AnalysisNumericalObjectKind::MagnetostaticWorkload,
    )?;
    let maxwell_workload = optional_unique_input(
        &invocation,
        AnalysisNumericalObjectKind::MaxwellHarmonicWorkload,
    )?;
    let acoustic_response_workload = optional_unique_input(
        &invocation,
        AnalysisNumericalObjectKind::AcousticResponseWorkload,
    )?;
    if usize::from(nonlinear_workload.is_some())
        + usize::from(linear_workload.is_some())
        + usize::from(frequency_workload.is_some())
        + usize::from(thermal_workload.is_some())
        + usize::from(magnetic_workload.is_some())
        + usize::from(maxwell_workload.is_some())
        + usize::from(acoustic_response_workload.is_some())
        > 1
    {
        return Err(AnalysisBridgeError::Invalid(
            "observation partition requires exactly one typed workload source".into(),
        ));
    }
    let resource_plan_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    let mesh = optional_unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    if invocation.inputs.len() != 7 + usize::from(mesh.is_some())
        || invocation.host.workload.outputs
            != vec![AnalysisNumericalObjectKind::ObservationPartitionContribution]
    {
        return Err(AnalysisBridgeError::Invalid(
            "observation partition inputs or contribution output differ from the workload contract"
                .into(),
        ));
    }
    let source_request_identity = source_request.manifest.logical_identity;
    let request_identity = request.manifest.logical_identity;
    let partition_plan_identity = partition_plan.manifest.logical_identity;
    let field_set_identity = fields.manifest.logical_identity;
    let resource_plan = runmat_analysis_execution::decode_admitted_resource_plan(
        resource_plan_input,
        invocation.output_policy.inventory_limits(),
    )?;
    runmat_analysis_execution::require_observation_partition_source_decode_budget(
        invocation.inputs,
        &resource_plan,
        invocation.output_policy.inventory_limits(),
    )?;
    let source_request =
        ObservationEvaluationRequest::from_canonical_bytes(&source_request.payload())?;
    let catalog =
        runmat_analysis_fea::results::observation::ObservationFieldCatalog::from_canonical_bytes(
            &field_catalog.payload(),
        )?;
    // Native node/element fields of any family carry their producing step in the catalog;
    // the producing workload input is provenance of a family-specific kind.
    let volume_mesh = matches!(
        catalog.source(),
        runmat_analysis_fea::results::observation::ObservationFieldCatalogSource::VolumeMeshRecovery { .. }
    );
    let typed_workloads = [
        nonlinear_workload.is_some(),
        linear_workload.is_some(),
        frequency_workload.is_some(),
        thermal_workload.is_some(),
        magnetic_workload.is_some(),
        maxwell_workload.is_some(),
        acoustic_response_workload.is_some(),
    ]
    .into_iter()
    .filter(|present| *present)
    .count();
    if (volume_mesh && typed_workloads != usize::from(acoustic_response_workload.is_some()))
        || (!volume_mesh && (typed_workloads != 1 || acoustic_response_workload.is_some()))
    {
        return Err(AnalysisBridgeError::Invalid(
            "observation partition requires exactly one workload source of its catalog kind".into(),
        ));
    }
    if catalog.identity() != field_catalog.manifest.logical_identity {
        return Err(AnalysisBridgeError::Identity(
            "observation source catalog identity differs from its input",
        ));
    }
    let request = ObservationPartitionEvaluationRequest::from_canonical_bytes(&request.payload())?;
    let partition_plan = ObservationPartitionPlan::from_canonical_bytes(&partition_plan.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let fields = RecoveredFieldPartitionSet::from_canonical_bytes(&fields.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let partition = partition_plan
        .partitions
        .iter()
        .find(|partition| partition.identity == request.partition)
        .ok_or(AnalysisBridgeError::Identity(
            "observation partition request names no canonical assignment",
        ))?;
    request.validate_for(&source_request, &partition_plan, partition, &fields)?;
    catalog.validate_canonical_selection(&source_request.plan, &source_request.selection)?;
    if source_request.identity().digest() != source_request_identity
        || request.identity().digest() != request_identity
        || partition_plan.identity.digest() != partition_plan_identity
        || fields.identity.digest() != field_set_identity
        || resource_plan.identity.digest() != resource_plan_input.manifest.logical_identity
        || request.producer != invocation.host.workload.step
    {
        return Err(AnalysisBridgeError::Identity(
            "observation partition inputs differ from their object identities",
        ));
    }
    let mesh = mesh
        .map(|mesh| {
            SolverMeshArtifact::canonical_decode(&mesh.payload())
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
        })
        .transpose()?;
    let mut solid_projection = None;
    let mut sampled_topology = None;
    let mut source_workspace = ObservationPartitionSourceWorkspace::InputEnvelope;
    let (model, backend) = if let Some(workload_objects) = magnetic_workload.or(maxwell_workload) {
        let (model, sampling) = sampled::admit_source(
            workload_objects,
            &catalog,
            &source_request,
            &partition_plan,
            mesh.as_ref(),
        )?;
        sampled_topology = Some(sampling);
        (Some(model), runmat_analysis_fea::ComputeBackend::Cpu)
    } else if volume_mesh {
        let mesh = mesh.as_ref().ok_or(AnalysisBridgeError::Identity(
            "volume-mesh observation requires its exact mesh",
        ))?;
        let step = catalog.volume_mesh_step()?;
        if let Some(objects) = acoustic_response_workload {
            let workload =
                runmat_analysis_fea::execution::AcousticResponseWorkload::from_canonical_bytes(
                    &objects.payload(),
                )?;
            if workload.identity()?.digest() != objects.manifest.logical_identity
                || workload.producer_identity()? != request.producer
                || workload
                    .model
                    .sequence
                    .steps
                    .iter()
                    .find(|candidate| candidate.step_id == workload.step_id)
                    != Some(step)
            {
                return Err(AnalysisBridgeError::Identity(
                    "acoustic response observation workload, producer or authored step differs",
                ));
            }
        }
        if request.event_attribution.is_some()
            || !resource_plan.request.accelerators.is_empty()
            || step.step_id != source_request.plan.step_id
        {
            return Err(AnalysisBridgeError::Identity(
                "volume-mesh observation step or serial envelope differs",
            ));
        }
        catalog.validate_volume_mesh_selection(&source_request.selection, mesh)?;
        catalog.validate_volume_mesh_plan(&source_request.plan, mesh)?;
        let expected_partitions = ObservationPartitionPlan::derive(
            &source_request.plan,
            &partition_plan.observation_id,
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::serial(),
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        if partition_plan != expected_partitions {
            return Err(AnalysisBridgeError::Identity(
                "volume-mesh observation partition policy differs",
            ));
        }
        solid_projection = Some(
            runmat_analysis_fea::topology::AnalysisProjection::derive(mesh)
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        );
        (None, runmat_analysis_fea::ComputeBackend::Cpu)
    } else if let Some(workload_objects) = thermal_workload {
        let workload = runmat_analysis_fea::execution::ThermalWorkload::from_canonical_bytes(
            &workload_objects.payload(),
        )?;
        if workload.identity()?.digest() != workload_objects.manifest.logical_identity
            || workload.backend != runmat_analysis_fea::ComputeBackend::Cpu
        {
            return Err(AnalysisBridgeError::Identity(
                "thermal observation workload identity or CPU envelope differs",
            ));
        }
        let mesh = mesh.as_ref().ok_or(AnalysisBridgeError::Identity(
            "thermal observation requires its exact mesh",
        ))?;
        let runmat_analysis_fea::results::observation::ObservationFieldCatalogSource::Thermal {
            manifest,
        } = catalog.source()
        else {
            return Err(AnalysisBridgeError::Identity(
                "thermal observation requires a thermal catalog",
            ));
        };
        // Admission read the same source law before validation allocates.
        source_workspace = ObservationPartitionSourceWorkspace::thermal(
            &workload,
            manifest,
            &source_request.plan,
            mesh,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        catalog.validate_thermal_selection(&source_request.selection, &workload, mesh)?;
        catalog.validate_thermal_plan(&source_request.plan, &workload, mesh)?;
        let expected_partitions = ObservationPartitionPlan::derive(
            &source_request.plan,
            &partition_plan.observation_id,
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::serial(),
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        if partition_plan != expected_partitions {
            return Err(AnalysisBridgeError::Identity(
                "thermal observation partition policy differs from its workload",
            ));
        }
        solid_projection = Some(
            runmat_analysis_fea::topology::AnalysisProjection::derive(mesh)
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        );
        (Some(workload.model), workload.backend)
    } else if let Some(workload_objects) = nonlinear_workload {
        let workload =
            runmat_analysis_fea::execution::NonlinearSolidWorkload::from_canonical_bytes(
                &workload_objects.payload(),
            )?;
        let source =
            runmat_analysis_fea::execution::StructuralObservationSource::NonlinearSolid(&workload);
        source.validate()?;
        if workload.identity()?.digest() != workload_objects.manifest.logical_identity
            || workload.producer_identity()? != request.producer
            || mesh.is_none()
            || request.event_attribution.is_some()
            || !resource_plan.request.accelerators.is_empty()
        {
            return Err(AnalysisBridgeError::Identity(
                "nonlinear final observation source or CPU envelope differs",
            ));
        }
        catalog.validate_structural_selection(&source_request.selection)?;
        catalog.validate_structural_plan(&source_request.plan, source.model(), mesh.as_ref())?;
        let expected = ObservationPartitionPlan::derive(
            &source_request.plan,
            &partition_plan.observation_id,
            source.observation_execution(),
        )
        .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?;
        if expected != partition_plan {
            return Err(AnalysisBridgeError::Identity(
                "nonlinear observation partition policy differs",
            ));
        }
        (
            Some(workload.model),
            runmat_analysis_fea::ComputeBackend::Cpu,
        )
    } else if let Some(workload) = linear_workload {
        catalog.validate_structural_selection(&source_request.selection)?;
        let workload = StructuralLinearWorkload::from_canonical_bytes(&workload.payload())?;
        (Some(workload.model), workload.backend)
    } else {
        catalog.validate_structural_selection(&source_request.selection)?;
        let workload = StructuralFrequencyWorkload::from_canonical_bytes(
            &frequency_workload
                .expect("exclusive source checked")
                .payload(),
        )?;
        (Some(workload.model().clone()), workload.backend())
    };
    // Only structural evaluation reads the authored model; the other sources carry their
    // own topology. A missing model outside those sources is rejected here.
    if model.is_none() && solid_projection.is_none() && sampled_topology.is_none() {
        return Err(AnalysisBridgeError::Identity(
            "structural observation evaluation requires its authored model",
        ));
    }
    let structural_model = || {
        model
            .as_ref()
            .expect("structural sources carry their model")
    };
    let FeaResourceDemand::StructuralObservationPartition(_) = &resource_plan.input.demand else {
        return Err(AnalysisBridgeError::Identity(
            "observation partition requires its exact phase resource demand",
        ));
    };
    let accelerator = match resource_plan.request.accelerators.as_slice() {
        [] => None,
        [allocation] => Some(allocation),
        _ => {
            return Err(AnalysisBridgeError::Identity(
                "observation partition has multiple admitted accelerator allocations",
            ))
        }
    };
    if sampled_topology.is_some() && (accelerator.is_some() || request.event_attribution.is_some())
    {
        return Err(AnalysisBridgeError::Identity(
            "sampled observations require serial final evaluation",
        ));
    }
    let input_facts = prepared_input_resource_facts(&inputs_except_resource_plan(&invocation))?;
    let expected_resource_plan =
        build_observation_partition_resource_plan(ObservationPartitionResourcePlanning {
            source: &source_request,
            request: &request,
            partition_plan: &partition_plan,
            partition,
            source_fields: &fields,
            inputs: input_facts.clone(),
            output: projected_output_resource_facts(
                observation_partition_payload_bytes_upper_bound(
                    &source_request,
                    &request,
                    &partition_plan,
                    partition,
                    &fields,
                    accelerator.is_some(),
                )
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            )?,
            output_set: projected_output_set_resource_facts(),
            accelerator,
            source_workspace,
        })
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if resource_plan.input.phase
        != FeaResourcePhase::StructuralObservationPartition(request.identity())
        || resource_plan != expected_resource_plan
    {
        return Err(AnalysisBridgeError::Identity(
            "observation partition resource plan does not converge from authoritative inputs",
        ));
    }
    let contribution = match resource_plan.request.accelerators.as_slice() {
        [] if sampled_topology.is_some() => {
            runmat_analysis_fea::results::observation::evaluate_sampled_observation_partition(
                &source_request.plan,
                &partition_plan,
                partition,
                sampled_topology
                    .as_ref()
                    .ok_or(AnalysisBridgeError::Identity("sampled topology absent"))?,
                mesh.as_ref()
                    .ok_or(AnalysisBridgeError::Identity("sampled mesh absent"))?,
                &fields,
            )
        }
        [] if solid_projection.is_some() => {
            runmat_analysis_fea::results::observation::evaluate_solid_observation_partition(
                &source_request.plan,
                &partition_plan,
                partition,
                solid_projection.as_ref().expect("solid-projection branch"),
                &fields,
                request.event_attribution,
            )
        }
        [] => evaluate_structural_observation_partition(
            &source_request.plan,
            &partition_plan,
            partition,
            structural_model(),
            mesh.as_ref(),
            &fields,
            request.event_attribution,
        ),
        [allocation] => {
            if backend != runmat_analysis_fea::ComputeBackend::Gpu
                || !invocation.host.workload.required_capabilities.contains(
                    &runmat_analysis_execution::AnalysisCapabilityRequirement::Accelerator {
                        class: allocation.class.clone(),
                    },
                )
            {
                return Err(AnalysisBridgeError::Identity(
                    "observation accelerator target differs from its workload placement",
                ));
            }
            evaluate_structural_observation_partition_accelerated(
                &source_request.plan,
                &partition_plan,
                partition,
                StructuralObservationAccelerationInput {
                    model: structural_model(),
                    mesh: mesh.as_ref(),
                    field: &fields,
                    event_attribution: request.event_attribution,
                    allocation,
                    resource_plan: resource_plan.identity,
                },
            )
        }
        _ => Err(
            runmat_analysis_fea::results::observation::ObservationError::Invalid(
                "observation partition has multiple admitted accelerator allocations".into(),
            ),
        ),
    }
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    Ok(vec![AnalysisKernelOutput::whole(
        AnalysisNumericalObjectKind::ObservationPartitionContribution,
        contribution.identity.digest(),
        contribution
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    )])
}
