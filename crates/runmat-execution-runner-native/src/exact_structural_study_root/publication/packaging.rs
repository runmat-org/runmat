use super::*;
mod frequency;
mod modal;
mod prestress;
mod transient;
use frequency::*;
use modal::*;
use prestress::*;
use transient::*;

#[cfg(test)]
mod thermal_tests;

pub(crate) fn package_exact_structural_study_result(
    request: &ProgramExecutionRequest,
    decoded: &DecodedExactStructuralStudyRoot,
    meshing: &ExactMeshingDagRunResult,
    region_resolution: &runmat_analysis_execution::PreparedAnalysisObjects,
    analysis: ExactStructuralAnalysisPublication<'_>,
) -> NativeExecutionResult<(ProgramExecutionResponse, Vec<ProgramExecutionObject>)> {
    let thermal_root = match &analysis {
        ExactStructuralAnalysisPublication::ThermalPrestress { completion, .. } => Some(
            runmat_analysis_execution::prepare_thermal_study_root(&completion.thermal)
                .map_err(protocol)?,
        ),
        _ => None,
    };
    if let Some(compiled) = &decoded.compiled_study {
        compiled
            .validate_against(&meshing.artifact)
            .map_err(protocol)?;
        if decoded.workloads.authored_model().map_err(protocol)? != &compiled.model {
            return Err(protocol(
                "compiled study model differs from its executed structural workloads",
            ));
        }
    }
    let region_resolution_objects = region_resolution
        .references(&decoded.artifact_access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let region_resolution_root = region_resolution_objects
        .first()
        .ok_or_else(|| protocol("topology-region resolution closure is empty"))?
        .clone();
    let input_closures = analysis
        .inputs()
        .iter()
        .map(|input| {
            input
                .objects
                .references(&decoded.artifact_access, ValueRefKind::ResultObject)
                .map_err(protocol)
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let input_roots = input_closures
        .iter()
        .map(|objects| {
            objects
                .first()
                .cloned()
                .ok_or_else(|| protocol("input materialization closure is empty"))
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let input_objects = unique_references(input_closures.into_iter().flatten())?;
    let unresolved_workloads_digest = analysis.unresolved_workloads_digest();
    let analysis = match analysis {
        ExactStructuralAnalysisPublication::ThermalPrestress { completion, .. } => {
            use crate::ThermalPrestressStructuralCompletion;
            match &completion.structural {
                ThermalPrestressStructuralCompletion::Buckling(value) => {
                    buckling_closure(value, &decoded.artifact_access)?
                }
                ThermalPrestressStructuralCompletion::Modal(value) => {
                    prestressed_modal_closure(value, &decoded.artifact_access)?
                }
                ThermalPrestressStructuralCompletion::Frequency(value) => {
                    prestressed_frequency_closure(value, &decoded.artifact_access)?
                }
                ThermalPrestressStructuralCompletion::ModalFrequency(value) => {
                    prestressed_modal_frequency_closure(value, &decoded.artifact_access)?
                }
            }
        }
        ExactStructuralAnalysisPublication::LinearSequence { sequence, .. } => {
            let closures = sequence.result_closures().map_err(protocol)?;
            let mut closure_objects = Vec::new();
            for closure in &closures {
                closure_objects.extend(response_objects(&closure.response)?);
            }
            let objects = unique_references(closure_objects)?;
            ExactStructuralStudyAnalysisClosure::LinearSequence { closures, objects }
        }
        ExactStructuralAnalysisPublication::StructuralTransient { completion, .. } => {
            transient_closure(
                &completion.result.result_objects,
                &completion.result.receipt_objects,
                &decoded.artifact_access,
            )?
        }
        ExactStructuralAnalysisPublication::StructuralModal { completion, .. } => {
            modal_closure(completion, &decoded.artifact_access)?
        }
        ExactStructuralAnalysisPublication::StructuralModalTransient { completion, .. } => {
            modal_transient_closure(completion, &decoded.artifact_access)?
        }
        ExactStructuralAnalysisPublication::StructuralModalResponseSpectrum {
            completion, ..
        } => modal_response_spectrum_closure(completion, &decoded.artifact_access)?,
        ExactStructuralAnalysisPublication::StructuralRandomResponse { completion, .. } => {
            random_response_closure(completion, &decoded.artifact_access)?
        }
        ExactStructuralAnalysisPublication::StructuralBuckling { completion, .. } => {
            buckling_closure(completion, &decoded.artifact_access)?
        }
        ExactStructuralAnalysisPublication::StructuralPrestressedModal { completion, .. } => {
            prestressed_modal_closure(completion, &decoded.artifact_access)?
        }
        ExactStructuralAnalysisPublication::StructuralPrestressedFrequency {
            completion, ..
        } => prestressed_frequency_closure(completion, &decoded.artifact_access)?,
        ExactStructuralAnalysisPublication::StructuralPrestressedModalFrequency {
            completion,
            ..
        } => prestressed_modal_frequency_closure(completion, &decoded.artifact_access)?,
        ExactStructuralAnalysisPublication::StructuralFrequency { completion, .. } => {
            frequency_closure(completion, &decoded.artifact_access)?
        }
    };
    let publication = ExactStructuralStudyPublication {
        schema_version: 4,
        thermal: thermal_root
            .as_ref()
            .map(|prepared| thermal_prefix::ThermalPrefixClosure {
                root: prepared.root.clone(),
                catalog: prepared.catalog.clone(),
            }),
        program_revision: request.recipe.program_revision.clone(),
        authored_study_digest: decoded.authored_study_digest,
        compiled_study: decoded.compiled_study.clone(),
        compiled_geometry_output: decoded.compiled_geometry_output.clone(),
        unresolved_workloads_digest,
        meshing_request_digest: Digest::from_bytes(
            *decoded
                .meshing_request
                .canonical_digest()
                .map_err(protocol)?
                .bytes(),
        ),
        meshing_root: meshing.root.clone(),
        meshing_objects: meshing.result_objects.clone(),
        region_resolution_root,
        region_resolution_objects: region_resolution_objects.clone(),
        input_roots,
        input_objects: input_objects.clone(),
        analysis,
    };
    let mut new_objects = region_resolution
        .objects
        .iter()
        .zip(&region_resolution_objects)
        .map(|(object, reference)| ProgramExecutionObject {
            reference: reference.clone(),
            bytes: object.bytes.clone(),
        })
        .collect::<Vec<_>>();
    let publication_object = LogicalObject::new(
        ObjectNamespace::ResultValue,
        "analysis/exact-structural-study-publications",
        PUBLICATION_MEDIA_TYPE,
        publication.canonical_bytes(&decoded.artifact_access)?,
    )
    .map_err(protocol)?;
    let publication_root = decoded
        .artifact_access
        .reference(
            &publication_object,
            ValueRefKind::ResultObject,
            PUBLICATION_VALUE_SCHEMA,
        )
        .map_err(protocol)?;
    let inventory = unique_references(
        meshing
            .result_objects
            .iter()
            .chain(&region_resolution_objects)
            .chain(&input_objects)
            .chain(publication.analysis.objects())
            .chain(
                publication
                    .thermal
                    .iter()
                    .flat_map(|thermal| thermal.catalog.objects.iter()),
            )
            .chain(std::iter::once(&publication_root))
            .cloned(),
    )?;
    reference_inventory(&inventory)?;
    let response = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![ValuePayload::Object(Box::new(publication_root.clone()))],
        result_objects: inventory,
    };
    response.validate_against(request).map_err(protocol)?;
    new_objects.push(ProgramExecutionObject {
        reference: publication_root,
        bytes: publication_object.bytes,
    });
    if let Some(thermal) = thermal_root {
        for object in thermal.objects {
            let reference = thermal
                .catalog
                .objects
                .iter()
                .find(|reference| reference.logical_digest == object.descriptor.digest)
                .ok_or_else(|| protocol("thermal prefix object missing from catalog"))?;
            new_objects.push(ProgramExecutionObject {
                reference: reference.clone(),
                bytes: object.bytes,
            });
        }
    }
    for object in &new_objects {
        object.validate().map_err(protocol)?;
    }
    Ok((response, new_objects))
}

fn numerical_closure_roots<const N: usize>(
    components: [&runmat_analysis_execution::PreparedAnalysisObjects; N],
    access: &ArtifactAccess,
) -> NativeExecutionResult<([ValueRef; N], Vec<ValueRef>)> {
    let roots = components
        .iter()
        .map(|component| {
            component
                .root_reference(access, ValueRefKind::ResultObject)
                .map_err(protocol)
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?
        .try_into()
        .map_err(|_| protocol("numerical closure has an invalid root inventory"))?;
    let objects = unique_references(
        components
            .into_iter()
            .map(|component| {
                component
                    .unique_references(access, ValueRefKind::ResultObject)
                    .map_err(protocol)
            })
            .collect::<NativeExecutionResult<Vec<_>>>()?
            .into_iter()
            .flatten(),
    )?;
    Ok((roots, objects))
}
