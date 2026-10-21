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
    let fluid_workload = optional_unique_input(
        &invocation,
        AnalysisNumericalObjectKind::FluidSteadyWorkload,
    )?;
    let acoustic_workload = optional_unique_input(
        &invocation,
        AnalysisNumericalObjectKind::AcousticHarmonicWorkload,
    )?;
    if usize::from(nonlinear_workload.is_some())
        + usize::from(linear_workload.is_some())
        + usize::from(frequency_workload.is_some())
        + usize::from(thermal_workload.is_some())
        + usize::from(magnetic_workload.is_some())
        + usize::from(maxwell_workload.is_some())
        + usize::from(fluid_workload.is_some())
        + usize::from(acoustic_workload.is_some())
        != 1
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
    let mut finite_volume_topology = None;
    let (model, backend) = if let Some(workload_objects) = fluid_workload {
        let (model, projection, geometry) = finite_volume::admit_source(
            workload_objects,
            &catalog,
            &source_request,
            &partition_plan,
            mesh.as_ref(),
        )?;
        finite_volume_topology = Some((projection, geometry));
        (model, runmat_analysis_fea::ComputeBackend::Cpu)
    } else if let Some(workload_objects) = magnetic_workload.or(maxwell_workload) {
        let (model, sampling) = sampled::admit_source(
            workload_objects,
            &catalog,
            &source_request,
            &partition_plan,
            mesh.as_ref(),
        )?;
        sampled_topology = Some(sampling);
        (model, runmat_analysis_fea::ComputeBackend::Cpu)
    } else if let Some(workload_objects) = acoustic_workload {
        let workload = runmat_analysis_fea::execution::AcousticHarmonicWorkload::from_canonical_bytes(
            &workload_objects.payload(),
        )?;
        let mesh = mesh.as_ref().ok_or(AnalysisBridgeError::Identity(
            "acoustic observation requires its exact mesh",
        ))?;
        if workload.identity()?.digest() != workload_objects.manifest.logical_identity
            || workload.producer_identity()? != request.producer
            || request.event_attribution.is_some()
            || !resource_plan.request.accelerators.is_empty()
        {
            return Err(AnalysisBridgeError::Identity(
                "acoustic observation workload identity or serial envelope differs",
            ));
        }
        let step = workload
            .model
            .sequence
            .steps
            .iter()
            .find(|step| step.step_id == workload.step_id)
            .ok_or(AnalysisBridgeError::Identity(
                "acoustic observation step is absent",
            ))?;
        catalog.validate_volume_mesh_selection(&source_request.selection, mesh)?;
        catalog.validate_volume_mesh_plan(&source_request.plan, step, mesh)?;
        let expected_partitions = ObservationPartitionPlan::derive(
            &source_request.plan,
            &partition_plan.observation_id,
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::serial(),
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        if partition_plan != expected_partitions {
            return Err(AnalysisBridgeError::Identity(
                "acoustic observation partition policy differs from its workload",
            ));
        }
        solid_projection = Some(
            runmat_analysis_fea::topology::AnalysisProjection::derive(mesh)
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        );
        (workload.model, runmat_analysis_fea::ComputeBackend::Cpu)
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
        (workload.model, workload.backend)
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
        (workload.model, runmat_analysis_fea::ComputeBackend::Cpu)
    } else if let Some(workload) = linear_workload {
        catalog.validate_structural_selection(&source_request.selection)?;
        let workload = StructuralLinearWorkload::from_canonical_bytes(&workload.payload())?;
        (workload.model, workload.backend)
    } else {
        catalog.validate_structural_selection(&source_request.selection)?;
        let workload = StructuralFrequencyWorkload::from_canonical_bytes(
            &frequency_workload
                .expect("exclusive source checked")
                .payload(),
        )?;
        (workload.model().clone(), workload.backend())
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
    if (sampled_topology.is_some() || finite_volume_topology.is_some())
        && (accelerator.is_some() || request.event_attribution.is_some())
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
                    &input_facts,
                    &fields,
                    accelerator.is_some(),
                )
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            )?,
            output_set: projected_output_set_resource_facts(),
            accelerator,
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
        [] if finite_volume_topology.is_some() => {
            let runmat_analysis_fea::results::observation::ObservationFieldCatalogSource::FiniteVolumeRecovery { layout, .. } = catalog.source() else {
                return Err(AnalysisBridgeError::Identity("finite-volume catalog layout absent"));
            };
            let (projection, geometry) = finite_volume_topology.as_ref().expect("finite-volume branch");
            runmat_analysis_fea::results::observation::evaluate_finite_volume_observation_partition(
                &source_request.plan, &partition_plan, partition,
                runmat_analysis_fea::results::observation::FiniteVolumeObservationSource { layout, projection, geometry },
                &fields, runmat_analysis_fea::results::observation::ObservationLimits::default())
        }
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
            &model,
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
                    model: &model,
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
