use std::collections::BTreeSet;

use runmat_analysis_core::StateOutput;
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, CanonicalSemanticState,
    SemanticStatePublicationRequest, SemanticStatePublicationSource, SolverResultIdentity,
    StatePublicationProfiles,
};
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::value::ValueRefKind;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ObjectNamespace, ProgramExecutionResponse};

use crate::{
    import_analysis_objects, import_analysis_output_set, prepare_analysis_objects,
    prepare_analysis_program, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, AnalysisHostWorkload, AnalysisWorkloadInput,
    AnalysisWorkloadRequest, PreparedAnalysisObjects, PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};

pub struct SemanticStatePublicationPreparation {
    pub source_host: AnalysisHostWorkload,
    pub source_result: PreparedAnalysisObjects,
    pub source_profiles: PreparedAnalysisObjects,
    pub source: SemanticStatePublicationSource,
    pub declaration: StateOutput,
    pub topology_digest: Digest,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedSemanticStatePublication {
    pub request: SemanticStatePublicationRequest,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedSemanticState {
    pub state: CanonicalSemanticState,
    pub objects: PreparedAnalysisObjects,
    pub response: ProgramExecutionResponse,
}

pub fn prepare_semantic_state_publication(
    preparation: SemanticStatePublicationPreparation,
) -> AnalysisBridgeResult<PreparedSemanticStatePublication> {
    preparation.source_host.validate()?;
    let source_kind_matches = matches!(
        (preparation.source, preparation.source_host.workload.kind),
        (
            SemanticStatePublicationSource::AcousticNativeFields,
            AnalysisWorkKind::SolveAcousticHarmonic
        ) | (
            SemanticStatePublicationSource::AcousticResponseFields,
            AnalysisWorkKind::SolveAcousticResponse
        ) | (
            SemanticStatePublicationSource::StaticElectricalNativeFields,
            AnalysisWorkKind::SolveElectricalStatic
        ) | (
            SemanticStatePublicationSource::MagnetostaticNativeFields,
            AnalysisWorkKind::SolveMagnetostatic
        ) | (
            SemanticStatePublicationSource::MaxwellNativeFields,
            AnalysisWorkKind::SolveMaxwellHarmonic
        ) | (
            SemanticStatePublicationSource::ThermalPrimaryField,
            AnalysisWorkKind::SolveThermal
        ) | (
            SemanticStatePublicationSource::StructuralPrimaryFields,
            AnalysisWorkKind::Solve
        ) | (
            SemanticStatePublicationSource::StructuralLinearization,
            AnalysisWorkKind::Solve
        ) | (
            SemanticStatePublicationSource::ModalBasis,
            AnalysisWorkKind::ExtractStructuralModes
        ) | (
            SemanticStatePublicationSource::StructuralResponseEnvelope,
            AnalysisWorkKind::AssembleStructuralModalResponseSpectrumResult
        ) | (
            SemanticStatePublicationSource::StructuralRandomStatistics,
            AnalysisWorkKind::AssembleStructuralRandomResponseResult
        )
    );
    if !source_kind_matches
        || preparation.source_result.manifest.kind != AnalysisNumericalObjectKind::SolverResult
        || preparation.chunk_bytes == 0
    {
        return Err(AnalysisBridgeError::Invalid(
            "semantic-state publication requires one completed solver result and bounded chunks"
                .into(),
        ));
    }
    let acoustic_source = (preparation.source
        == SemanticStatePublicationSource::AcousticNativeFields)
        .then(|| crate::acoustic_state_profiles::decode_acoustic_source(&preparation.source_result))
        .transpose()?;
    let response_source = (preparation.source
        == SemanticStatePublicationSource::AcousticResponseFields)
        .then(|| {
            crate::acoustic_response_state_profiles::decode_acoustic_response_source(
                &preparation.source_result,
            )
        })
        .transpose()?;
    let electrical_source = (preparation.source
        == SemanticStatePublicationSource::StaticElectricalNativeFields)
        .then(|| {
            crate::electrical_state_profiles::decode_electrical_source(&preparation.source_result)
        })
        .transpose()?;
    let maxwell_source = (preparation.source
        == SemanticStatePublicationSource::MaxwellNativeFields)
        .then(|| crate::maxwell_state_profiles::decode_maxwell_source(&preparation.source_result))
        .transpose()?;
    let magnetic_source = (preparation.source
        == SemanticStatePublicationSource::MagnetostaticNativeFields)
        .then(|| crate::magnetic_state_profiles::decode_magnetic_source(&preparation.source_result))
        .transpose()?;
    let source_result = if preparation.source == SemanticStatePublicationSource::ThermalPrimaryField
    {
        let manifest = crate::decode_prepared_thermal_manifest(&preparation.source_result)?;
        if manifest.producer != preparation.source_host.workload.step
            || manifest.topology_digest != preparation.topology_digest
            || !preparation.source_host.workload.inputs.iter().any(|input| {
                input.kind == AnalysisNumericalObjectKind::ThermalWorkload
                    && input.logical_identity == manifest.source_workload.digest()
            })
        {
            return Err(AnalysisBridgeError::Identity(
                "thermal state source differs from its workload",
            ));
        }
        manifest.identity()
    } else if let Some(result) = &maxwell_source {
        let topology = Digest::from_bytes(*result.response().mesh_digest.bytes());
        if result.producer_identity() != preparation.source_host.workload.step
            || result
                .sampling_topology()
                .identity()
                .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?
                .discretization_digest()
                != preparation.topology_digest
            || !preparation.source_host.workload.inputs.iter().any(|input| {
                input.kind == AnalysisNumericalObjectKind::MaxwellHarmonicWorkload
                    && input.logical_identity == result.source_workload().digest()
            })
            || !preparation.source_host.workload.inputs.iter().any(|input| {
                input.kind == AnalysisNumericalObjectKind::Mesh
                    && input.logical_identity == topology
            })
        {
            return Err(AnalysisBridgeError::Identity(
                "Maxwell state source differs from original workload/mesh/sample topology",
            ));
        }
        result.identity()?
    } else if let Some(result) = &magnetic_source {
        if result.producer() != preparation.source_host.workload.step
            || result
                .sampling_topology()
                .identity()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?
                .discretization_digest()
                != preparation.topology_digest
            || !preparation.source_host.workload.inputs.iter().any(|input| {
                input.kind == AnalysisNumericalObjectKind::MagnetostaticWorkload
                    && input.logical_identity == result.source_workload().digest()
            })
            || !preparation.source_host.workload.inputs.iter().any(|input| {
                input.kind == AnalysisNumericalObjectKind::Mesh
                    && input.logical_identity == result.mesh_digest()
            })
        {
            return Err(AnalysisBridgeError::Identity(
                "magnetic state source differs from workload, mesh or sampled topology",
            ));
        }
        result.source_result_identity()?
    } else if let Some(result) = &response_source {
        if result.producer() != preparation.source_host.workload.step
            || Digest::from_bytes(*result.mesh_digest().bytes()) != preparation.topology_digest
            || !preparation.source_host.workload.inputs.iter().any(|input| {
                input.kind == AnalysisNumericalObjectKind::AcousticResponseWorkload
                    && input.logical_identity == result.source_workload().digest()
            })
            || !preparation.source_host.workload.inputs.iter().any(|input| {
                input.kind == AnalysisNumericalObjectKind::Mesh
                    && input.logical_identity == preparation.topology_digest
            })
        {
            return Err(AnalysisBridgeError::Identity(
                "acoustic response state source differs from workload or topology",
            ));
        }
        result.source_result_identity()?
    } else if let Some(result) = &electrical_source {
        if result.producer() != preparation.source_host.workload.step
            || Digest::from_bytes(*result.mesh_digest().bytes()) != preparation.topology_digest
            || !preparation.source_host.workload.inputs.iter().any(|input| {
                input.kind == AnalysisNumericalObjectKind::ElectricalStaticWorkload
                    && input.logical_identity == result.source_workload().digest()
            })
            || !preparation.source_host.workload.inputs.iter().any(|input| {
                input.kind == AnalysisNumericalObjectKind::Mesh
                    && input.logical_identity == preparation.topology_digest
            })
        {
            return Err(AnalysisBridgeError::Identity(
                "electrical state source differs from workload or topology",
            ));
        }
        result.source_result_identity()?
    } else if let Some(result) = &acoustic_source {
        if result.producer() != preparation.source_host.workload.step
            || Digest::from_bytes(*result.mesh_digest().bytes()) != preparation.topology_digest
            || !preparation.source_host.workload.inputs.iter().any(|input| {
                input.kind == AnalysisNumericalObjectKind::AcousticHarmonicWorkload
                    && input.logical_identity == result.source_workload().digest()
            })
            || !preparation.source_host.workload.inputs.iter().any(|input| {
                input.kind == AnalysisNumericalObjectKind::Mesh
                    && input.logical_identity == preparation.topology_digest
            })
        {
            return Err(AnalysisBridgeError::Identity(
                "acoustic state source differs from workload or topology",
            ));
        }
        result.source_result_identity()?
    } else if matches!(
        preparation.source,
        SemanticStatePublicationSource::StructuralPrimaryFields
            | SemanticStatePublicationSource::StructuralLinearization
    ) {
        crate::structural::result_source::decode_structural_result_source(
            &preparation.source_host,
            &preparation.source_result,
        )?
        .0
    } else {
        SolverResultIdentity::new(
            preparation.source_host.workload.identity,
            preparation.source_result.manifest.payload_digest,
        )?
    };
    if preparation.source_result.manifest.logical_identity != source_result.digest() {
        return Err(AnalysisBridgeError::Identity(
            "solver result identity does not match its source workload and content",
        ));
    }
    let request = SemanticStatePublicationRequest::new(
        preparation.source,
        preparation.source_host.workload.step,
        preparation.declaration,
        preparation.topology_digest,
        source_result,
    )?;
    let request_bytes = request.canonical_bytes()?;
    if let Some(result) = &maxwell_source {
        result.validate_publication_request(&request)?;
        let catalog =
            StatePublicationProfiles::from_canonical_bytes(&preparation.source_profiles.payload())?;
        if preparation.source_profiles.manifest.kind
            != AnalysisNumericalObjectKind::StatePublicationProfiles
            || preparation.source_profiles.manifest.logical_identity != catalog.identity()
            || catalog
                != result
                    .state_publication_profiles(preparation.source_result.manifest.payload_digest)?
        {
            return Err(AnalysisBridgeError::Identity(
                "Maxwell publication profiles differ from original completed source",
            ));
        }
    }
    if let Some(result) = &magnetic_source {
        result.validate_publication_request(&request)?;
        let catalog =
            StatePublicationProfiles::from_canonical_bytes(&preparation.source_profiles.payload())?;
        if preparation.source_profiles.manifest.kind
            != AnalysisNumericalObjectKind::StatePublicationProfiles
            || preparation.source_profiles.manifest.logical_identity != catalog.identity()
            || catalog
                != result
                    .state_publication_profiles(preparation.source_result.manifest.payload_digest)?
        {
            return Err(AnalysisBridgeError::Identity(
                "magnetic publication profiles differ from completed source",
            ));
        }
    }
    if let Some(result) = &response_source {
        result.validate_publication_request(&request)?;
        let catalog =
            StatePublicationProfiles::from_canonical_bytes(&preparation.source_profiles.payload())?;
        if preparation.source_profiles.manifest.kind
            != AnalysisNumericalObjectKind::StatePublicationProfiles
            || preparation.source_profiles.manifest.logical_identity != catalog.identity()
            || catalog
                != result
                    .state_publication_profiles(preparation.source_result.manifest.payload_digest)?
        {
            return Err(AnalysisBridgeError::Identity(
                "acoustic response publication profiles differ from completed source",
            ));
        }
    }
    if let Some(result) = &electrical_source {
        result.validate_publication_request(&request)?;
        let catalog =
            StatePublicationProfiles::from_canonical_bytes(&preparation.source_profiles.payload())?;
        if preparation.source_profiles.manifest.kind
            != AnalysisNumericalObjectKind::StatePublicationProfiles
            || preparation.source_profiles.manifest.logical_identity != catalog.identity()
            || catalog
                != result
                    .state_publication_profiles(preparation.source_result.manifest.payload_digest)?
        {
            return Err(AnalysisBridgeError::Identity(
                "electrical publication profiles differ from completed source",
            ));
        }
    }
    if let Some(result) = &acoustic_source {
        result.validate_publication_request(&request)?;
        let catalog =
            StatePublicationProfiles::from_canonical_bytes(&preparation.source_profiles.payload())?;
        if preparation.source_profiles.manifest.kind
            != AnalysisNumericalObjectKind::StatePublicationProfiles
            || preparation.source_profiles.manifest.logical_identity != catalog.identity()
            || catalog
                != result
                    .state_publication_profiles(preparation.source_result.manifest.payload_digest)?
        {
            return Err(AnalysisBridgeError::Identity(
                "acoustic publication profiles differ from completed source",
            ));
        }
    }
    let request_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::StatePublicationRequest,
        Digest::sha256(&request_bytes),
        &request_bytes,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let inputs = vec![
        (request_objects, ValueRefKind::DriverObject),
        (preparation.source_result, ValueRefKind::ResultObject),
        (preparation.source_profiles, ValueRefKind::ResultObject),
    ];
    let mut workload_inputs: Vec<AnalysisWorkloadInput> = inputs
        .iter()
        .map(|(input, reference_kind)| AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: *reference_kind,
        })
        .collect();
    let mut inputs = inputs
        .into_iter()
        .map(|(input, _)| input)
        .collect::<Vec<_>>();
    let catalog = StatePublicationProfiles::from_canonical_bytes(&inputs[2].payload())?;
    let resource_plan =
        crate::build_prepared_state_publication_resource_plan(&request, &catalog, &inputs)?;
    let plan_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        resource_plan.identity.digest(),
        &resource_plan.canonical_bytes()?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    workload_inputs.push(AnalysisWorkloadInput {
        kind: plan_objects.manifest.kind,
        logical_identity: plan_objects.manifest.logical_identity,
        payload_digest: plan_objects.manifest.payload_digest,
        manifest_digest: plan_objects.root.digest,
        manifest_encoded_length: plan_objects.root.encoded_length,
        reference_kind: ValueRefKind::DriverObject,
    });
    inputs.push(plan_objects);
    let required_capabilities = BTreeSet::from([AnalysisCapabilityRequirement::Host {
        abi: ANALYSIS_HOST_ABI.into(),
    }]);
    let workload = AnalysisWorkloadRequest::new(
        preparation.source_host.workload.study,
        preparation.source_host.workload.step,
        preparation.source_host.workload.execution_plan,
        AnalysisWorkKind::PublishState,
        workload_inputs,
        vec![AnalysisNumericalObjectKind::SemanticState],
        required_capabilities,
    )?;
    let host = AnalysisHostWorkload::new(
        workload,
        preparation.source_host.artifact_access,
        crate::AnalysisOutputPolicy::new(preparation.chunk_bytes, preparation.inventory_limits)?,
    )?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedSemanticStatePublication {
        request,
        host,
        program,
        inputs,
        resource_plan,
    })
}

pub fn import_semantic_state(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedSemanticStatePublication,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<CanonicalSemanticState> {
    Ok(import_semantic_state_completion(source, response, prepared, limits)?.state)
}

pub fn import_semantic_state_completion(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedSemanticStatePublication,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedSemanticState> {
    response.validate_against(&prepared.program.request)?;
    let retained_response = response.clone();
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [output] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "semantic-state publication must return exactly one state root".into(),
        ));
    };
    let imported = import_analysis_objects(
        source,
        &output.root,
        AnalysisNumericalObjectKind::SemanticState,
        output.logical_identity,
        output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let mut expected =
        imported.unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    expected.push(output_set.root);
    if output_set.result_objects != expected {
        return Err(AnalysisBridgeError::Identity(
            "semantic-state result inventory differs from its canonical closure",
        ));
    }
    // Import admission precedes joining output chunks or decoding the owned semantic
    // payload. A matching response/request alone does not bound output materialization.
    prepared.resource_plan.validate_identity()?;
    let runmat_analysis_fea::planning::resource::FeaResourceDemand::SemanticStatePublication(
        demand,
    ) = &prepared.resource_plan.input.demand
    else {
        return Err(AnalysisBridgeError::Identity(
            "semantic import lacks publication resource authority",
        ));
    };
    if demand.request != Digest::sha256(prepared.request.canonical_bytes()?)
        || demand.source != prepared.request.source
        || demand.source_result != prepared.request.source_result
        || demand.topology_digest != prepared.request.topology_digest
    {
        return Err(AnalysisBridgeError::Identity(
            "semantic import resource request differs",
        ));
    }
    crate::object::validate_prepared_output_resource_facts(&imported, &demand.output)?;
    let payload = imported.payload();
    let materialization =
        runmat_canonical_codec::canonical_payload_typed_resident_bytes_estimate(&payload)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if materialization > demand.state_materialization_resident_bytes_estimate.0 {
        return Err(AnalysisBridgeError::Identity(
            "semantic output actual codec profile exceeds publication admission",
        ));
    }
    let state = CanonicalSemanticState::from_canonical_bytes(&payload)?;
    if output.logical_identity != state.identity.digest()
        || state.producer != prepared.request.producer
        || state.state_id != prepared.request.declaration.state_id
        || state.kind != prepared.request.declaration.kind
        || state.completeness != prepared.request.declaration.completeness
        || state.persistence != prepared.request.declaration.persistence
        || state.topology_digest != prepared.request.topology_digest
        || state.source_result != prepared.request.source_result
    {
        return Err(AnalysisBridgeError::Identity(
            "published semantic state differs from its admitted request",
        ));
    }
    Ok(ImportedSemanticState {
        state,
        objects: imported,
        response: retained_response,
    })
}
