//! A line-parameter study on a `cross_section` geometry: meshed and solved in place on the
//! section's triangle mesh, with no exact geometry, volume mesh or prepared worker root.
use runmat_analysis_core::AuthoredStudyInputBinding;
use runmat_analysis_fea::physics::electrical::cross_section::{
    cross_section_digest, cross_section_meshing_evidence, solve_cross_section_with_mesh,
    CrossSection,
};
use runmat_geometry_core::GeometryDefinitionSource;
use runmat_runtime::analysis::{AnalysisLineParametersRunOptions, LineParametersStudyCompletion};

use super::{CompletedFeaStudyResult, CompletedLocalFeaCase, LocalFeaExecutionOptions};

/// A solved section with the mesh it was solved on and that mesh's evidence.
#[derive(Debug)]
pub struct CompletedLineParametersStudy {
    pub completion: LineParametersStudyCompletion,
    pub mesh: runmat_meshing_core::SolverMeshArtifact,
    pub meshing_evidence: runmat_meshing_core::MeshingEvidence,
    pub options: AnalysisLineParametersRunOptions,
}

/// Whether `study` solves a cross-section; such studies take [`execute_case`].
pub(super) fn is_cross_section(study: &runmat_analysis_language::ResolvedAnalysisStudy) -> bool {
    matches!(
        study.geometry.source,
        GeometryDefinitionSource::CrossSection(_)
    )
}

pub(super) async fn execute_case(
    source: &runmat_core::CheckedFeaSessionSource,
    study: runmat_analysis_language::ResolvedAnalysisStudy,
    inputs: Vec<AuthoredStudyInputBinding>,
    options: &LocalFeaExecutionOptions,
) -> Result<CompletedLocalFeaCase, String> {
    let case_id = study.study.study_id.clone();
    let realized = study
        .study
        .realize_with_models(&study.models, &study.geometry, &inputs)
        .map_err(|error| error.to_string())?;
    let family = runmat_analysis_fea::admission::admit_study(&realized, &study.material_catalog)
        .map_err(|issues| {
            issues
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n")
        })?;
    if family != runmat_analysis_fea::admission::AdmittedFamily::LineParameters {
        return Err("a cross_section geometry is solved only by a line_parameters analysis".into());
    }
    if options.backend != runmat_analysis_fea::ComputeBackend::Cpu
        || !options.accelerators.is_empty()
    {
        return Err("line-parameter execution has no admitted accelerator envelope".into());
    }
    let GeometryDefinitionSource::CrossSection(definition) = &realized.geometry.source else {
        return Err("a line_parameters analysis needs a cross_section geometry".into());
    };
    let settings = study.mesh.settings.clone();
    let controls = settings.cross_section;
    let section = CrossSection::from_authored(definition, controls.as_ref());
    let digest =
        cross_section_digest(definition, controls.as_ref()).map_err(|error| error.to_string())?;
    let request = crate::resolve_native_meshing_request(
        &runmat_geometry_io::ExactCadImportOptions::default(),
        settings,
    )
    .map_err(|error| error.to_string())?;
    let ceilings = crate::machine_capacity::meshing_execution_ceilings(&request.limits);
    let solved =
        solve_cross_section_with_mesh(&section, digest, request).map_err(|e| e.to_string())?;
    if let Some(observer) = &options.mesh_observer {
        observer(&solved.mesh);
    }
    let meshing_evidence = cross_section_meshing_evidence(
        &solved.mesh,
        ceilings,
        crate::meshing_platform::native_meshing_platform(None, None),
        solved.meshing_time_ms,
    )
    .map_err(|error| error.to_string())?;
    let compiled =
        runmat_analysis_fea::study::compile_cross_section_study(&realized, digest, &solved.mesh)
            .map_err(|error| error.to_string())?;
    let [step] = compiled.model.sequence.steps.as_slice() else {
        return Err("a line-parameter study has exactly one analysis".into());
    };
    let observation_results =
        runmat_analysis_fea::results::observation::evaluate_line_parameter_observations(
            step,
            &solved.mesh,
            &solved.parameters,
        )
        .map_err(|error| error.to_string())?;
    let frozen_project_digest = source
        .frozen_project()
        .map(|project| runmat_execution::Digest::from_bytes(*project.graph_digest().bytes()));
    let result = CompletedFeaStudyResult::LineParameters(Box::new(CompletedLineParametersStudy {
        completion: LineParametersStudyCompletion {
            compiled,
            section_digest: digest,
            parameters: solved.parameters,
            observation_results,
            frozen_project_digest,
        },
        mesh: solved.mesh,
        meshing_evidence,
        options: AnalysisLineParametersRunOptions::default(),
    }));
    let mut publication = super::prepare_study_publication(&result, "typed-local")?;
    publication
        .admit_accelerator_placement(&options.accelerator_placement)
        .map_err(|error| error.to_string())?;
    let publication = super::assessment::complete(source, publication)
        .await
        .map_err(|error| error.to_string())?;
    Ok(CompletedLocalFeaCase {
        case_id,
        result,
        publication,
    })
}
