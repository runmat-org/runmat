use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{
    AuthoredInputValue, AuthoredModel, AuthoredModelArgumentValue, AuthoredStudy, StepObjectRef,
};
use runmat_analysis_language::{FeaSymbolKind, LogicalSourcePath};
use runmat_geometry_core::{GeometryDefinitionSource, GeometryRegionRef, GeometryRegionSelector};

use super::{
    model::present_models, FeaOutputPresentation, FeaProcedurePresentation,
    FeaStudyInputPresentation, FeaStudyObjectKind, FeaStudyObjectPresentation,
    FeaStudyPresentation, FeaStudySectionKind, FeaStudySectionPresentation, PresentationSourceMap,
};
use crate::analysis::source::CheckedFeaSource;

impl CheckedFeaSource {
    pub fn study_presentations(&self) -> Vec<FeaStudyPresentation> {
        let sources = PresentationSourceMap::from_declarations(self.program.declarations.iter());
        self.program
            .studies
            .values()
            .filter(|study| study.module_path == self.root_module.as_str())
            .map(|study| self.present_study(study, &sources))
            .collect()
    }

    fn present_study(
        &self,
        study: &AuthoredStudy,
        sources: &PresentationSourceMap,
    ) -> FeaStudyPresentation {
        let models = study_models(study, self.program.models.values());
        let region_bindings = model_region_bindings(study);
        let region_selectors = geometry_region_selectors(self.program.geometries.values());
        let active_steps = active_object_steps(study);
        let all_step_ids = study
            .sequence
            .steps
            .iter()
            .map(|step| step.step_id.0.clone())
            .collect::<Vec<_>>();
        let geometry_units = self
            .program
            .geometries
            .values()
            .find(|geometry| {
                geometry.module_path == study.geometry.definition.module_path
                    && geometry.geometry_id == study.geometry.definition.geometry_id
            })
            .and_then(|geometry| match &geometry.source {
                GeometryDefinitionSource::Imported(source) => Some(source.units),
                _ => None,
            });
        let study_source = LogicalSourcePath::new(&study.module_path)
            .ok()
            .and_then(|module| sources.declaration(&module, FeaSymbolKind::Study, &study.study_id));
        let objects = present_models(
            &models,
            &region_bindings,
            &region_selectors,
            &active_steps,
            &all_step_ids,
            sources,
        );
        let available_sections = available_sections(study, &objects);

        FeaStudyPresentation {
            study_id: study.study_id.clone(),
            geometry_id: study.geometry.definition.geometry_id.clone(),
            mesh_id: study.mesh.mesh_id.clone(),
            model_id: study.model.model.model_id.clone(),
            geometry_units,
            inputs: study
                .inputs
                .iter()
                .map(|input| FeaStudyInputPresentation {
                    input_id: input.input_id.clone(),
                    value_type: input.value_type,
                    default: input.default.clone(),
                    minimum: input.minimum.clone(),
                    maximum: input.maximum.clone(),
                    required: input.is_required(),
                    source: study_source.clone(),
                })
                .collect(),
            objects,
            procedures: study
                .sequence
                .steps
                .iter()
                .map(|step| FeaProcedurePresentation {
                    step_id: step.step_id.0.clone(),
                    physics_family: step.family,
                    analysis_procedure: step.procedure,
                    depends_on: study
                        .sequence
                        .dependencies
                        .iter()
                        .filter(|dependency| dependency.successor == step.step_id)
                        .map(|dependency| dependency.predecessor.0.clone())
                        .collect(),
                    active_object_ids: step
                        .activations
                        .iter()
                        .map(|activation| step_object_id(&activation.object).to_string())
                        .collect(),
                    source: study_source.clone(),
                })
                .collect(),
            outputs: study
                .outputs
                .iter()
                .map(|output| FeaOutputPresentation {
                    output_id: output.output_id.clone(),
                    attributed: true,
                    source: study_source.clone(),
                })
                .collect(),
            available_sections,
        }
    }
}

fn available_sections(
    study: &AuthoredStudy,
    objects: &[FeaStudyObjectPresentation],
) -> Vec<FeaStudySectionPresentation> {
    use FeaStudySectionKind as Section;
    let count = |section| {
        objects
            .iter()
            .filter(|object| object_section(&object.kind) == section)
            .count()
    };
    [
        (Section::Inputs, study.inputs.len()),
        (
            Section::MaterialAssignments,
            count(Section::MaterialAssignments),
        ),
        (
            Section::PropertyAssignments,
            count(Section::PropertyAssignments),
        ),
        (Section::Constraints, count(Section::Constraints)),
        (Section::Boundaries, count(Section::Boundaries)),
        (Section::Connectors, count(Section::Connectors)),
        (Section::Interfaces, count(Section::Interfaces)),
        (Section::Signals, count(Section::Signals)),
        (Section::ExcitationCases, count(Section::ExcitationCases)),
        (Section::Excitations, count(Section::Excitations)),
        (
            Section::InitialConditions,
            count(Section::InitialConditions),
        ),
        (Section::ResponseSpectra, count(Section::ResponseSpectra)),
        (
            Section::RandomExcitations,
            count(Section::RandomExcitations),
        ),
        (Section::Outputs, study.outputs.len()),
    ]
    .into_iter()
    .map(|(section, configured_count)| FeaStudySectionPresentation {
        section,
        configured_count,
    })
    .collect()
}

fn object_section(kind: &FeaStudyObjectKind) -> FeaStudySectionKind {
    match kind {
        FeaStudyObjectKind::MaterialAssignment => FeaStudySectionKind::MaterialAssignments,
        FeaStudyObjectKind::PropertyAssignment => FeaStudySectionKind::PropertyAssignments,
        FeaStudyObjectKind::Constraint(_) => FeaStudySectionKind::Constraints,
        FeaStudyObjectKind::Boundary(_) => FeaStudySectionKind::Boundaries,
        FeaStudyObjectKind::Connector(_) => FeaStudySectionKind::Connectors,
        FeaStudyObjectKind::Interface(_) => FeaStudySectionKind::Interfaces,
        FeaStudyObjectKind::Signal => FeaStudySectionKind::Signals,
        FeaStudyObjectKind::Excitation(_) => FeaStudySectionKind::Excitations,
        FeaStudyObjectKind::ExcitationCase => FeaStudySectionKind::ExcitationCases,
        FeaStudyObjectKind::InitialCondition(_) => FeaStudySectionKind::InitialConditions,
        FeaStudyObjectKind::ResponseSpectrum => FeaStudySectionKind::ResponseSpectra,
        FeaStudyObjectKind::RandomExcitation => FeaStudySectionKind::RandomExcitations,
    }
}

fn geometry_region_selectors<'a>(
    geometries: impl Iterator<Item = &'a runmat_geometry_core::GeometryDefinition>,
) -> BTreeMap<GeometryRegionRef, GeometryRegionSelector> {
    geometries
        .flat_map(|geometry| {
            geometry.regions.iter().map(move |region| {
                (
                    GeometryRegionRef {
                        module_path: geometry.module_path.clone(),
                        geometry_id: geometry.geometry_id.clone(),
                        region_id: region.region_id.clone(),
                        role: region.role,
                    },
                    region.selector.clone(),
                )
            })
        })
        .collect()
}

fn model_region_bindings(study: &AuthoredStudy) -> BTreeMap<String, GeometryRegionRef> {
    let input_defaults = study
        .inputs
        .iter()
        .filter_map(|input| match input.default.as_ref() {
            Some(AuthoredInputValue::Region(region)) => Some((input.input_id.as_str(), region)),
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    study
        .model
        .arguments
        .iter()
        .filter_map(|argument| {
            let region = match &argument.value {
                AuthoredModelArgumentValue::Constant(AuthoredInputValue::Region(region)) => region,
                AuthoredModelArgumentValue::StudyInput(input) => {
                    input_defaults.get(input.input_id.as_str())?
                }
                _ => return None,
            };
            Some((argument.input_id.clone(), region.clone()))
        })
        .collect()
}

fn step_object_id(object: &StepObjectRef) -> &str {
    match object {
        StepObjectRef::Constraint(id)
        | StepObjectRef::Boundary(id)
        | StepObjectRef::Connector(id)
        | StepObjectRef::Interface(id)
        | StepObjectRef::Excitation(id)
        | StepObjectRef::InitialCondition(id)
        | StepObjectRef::MaterialParameter(id) => id,
    }
}

fn active_object_steps(study: &AuthoredStudy) -> BTreeMap<String, Vec<String>> {
    let mut result = BTreeMap::<String, Vec<String>>::new();
    for step in &study.sequence.steps {
        for activation in &step.activations {
            result
                .entry(step_object_id(&activation.object).to_string())
                .or_default()
                .push(step.step_id.0.clone());
        }
    }
    result
}

fn study_models<'a>(
    study: &AuthoredStudy,
    models: impl Iterator<Item = &'a AuthoredModel>,
) -> Vec<&'a AuthoredModel> {
    let catalog = models.collect::<Vec<_>>();
    let mut pending = vec![study.model.model.clone()];
    let mut visited = BTreeSet::new();
    let mut selected = Vec::new();
    while let Some(reference) = pending.pop() {
        if !visited.insert(reference.clone()) {
            continue;
        }
        let Some(model) = catalog
            .iter()
            .copied()
            .find(|model| model.reference() == reference)
        else {
            continue;
        };
        pending.extend(model.includes.iter().map(|include| include.model.clone()));
        selected.push(model);
    }
    selected
}
