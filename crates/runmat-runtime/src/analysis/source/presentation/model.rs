use std::collections::BTreeMap;

use runmat_analysis_core::{
    AuthoredInitialConditionTarget, AuthoredModel, AuthoredPropertyAssignment,
    AuthoredPropertyTarget, AuthoredRegionSource,
};
use runmat_geometry_core::GeometryRegionRef;

use super::{
    kinds::{
        boundary_kind, connector_kind, constraint_kind, initial_condition_kind, interface_kind,
    },
    signal::{present_parameterized_signal, present_signal},
    FeaPresentationTarget, FeaRandomExcitationPresentation, FeaResponseSpectrumPresentation,
    FeaStudyObjectDetails, FeaStudyObjectKind, FeaStudyObjectPresentation, PresentationSourceMap,
};

pub(super) fn present_models(
    models: &[&AuthoredModel],
    region_bindings: &BTreeMap<String, GeometryRegionRef>,
    active_steps: &BTreeMap<String, Vec<String>>,
    all_step_ids: &[String],
    sources: &PresentationSourceMap,
) -> Vec<FeaStudyObjectPresentation> {
    let mut objects = Vec::new();
    for model in models {
        let signals = model
            .signals
            .iter()
            .map(|signal| (signal.signal_id().0.as_str(), signal))
            .collect::<BTreeMap<_, _>>();
        let parameterized_signals = model
            .parameterized_signals
            .iter()
            .map(|signal| (signal.signal_id.0.as_str(), signal))
            .collect::<BTreeMap<_, _>>();
        let case_membership = excitation_case_membership(model);

        for (index, assignment) in model.material_assignments.iter().enumerate() {
            objects.push(object(
                format!("material_assignment.{index}"),
                model,
                FeaStudyObjectKind::MaterialAssignment,
                FeaStudyObjectDetails::MaterialAssignment(assignment.clone()),
                property_target(&assignment.target, region_bindings),
                None,
            ));
        }
        for (index, assignment) in model.property_assignments.iter().enumerate() {
            objects.push(object(
                format!("property_assignment.{index}"),
                model,
                FeaStudyObjectKind::PropertyAssignment,
                FeaStudyObjectDetails::PropertyAssignment(assignment.clone()),
                property_assignment_targets(assignment, region_bindings),
                None,
            ));
        }
        for constraint in &model.constraints {
            let mut targets = Vec::new();
            constraint.kind.visit_targets(|target| {
                if let Some(target) = region_target(&target.region, region_bindings) {
                    targets.push(target);
                }
            });
            objects.push(object(
                constraint.constraint_id.clone(),
                model,
                FeaStudyObjectKind::Constraint(constraint_kind(&constraint.kind)),
                FeaStudyObjectDetails::Constraint(constraint.kind.clone()),
                targets,
                sources.model_object(
                    &model.module_path,
                    &model.model_id,
                    "constraint",
                    local_id(&constraint.constraint_id),
                ),
            ));
        }
        for boundary in &model.boundaries {
            objects.push(object(
                boundary.boundary_id.clone(),
                model,
                FeaStudyObjectKind::Boundary(boundary_kind(&boundary.kind)),
                FeaStudyObjectDetails::Boundary(boundary.kind.clone()),
                region_target(&boundary.target, region_bindings)
                    .into_iter()
                    .collect(),
                sources.model_object(
                    &model.module_path,
                    &model.model_id,
                    "load",
                    local_id(&boundary.boundary_id),
                ),
            ));
        }
        for connector in &model.connectors {
            let mut targets = region_target(&connector.primary, region_bindings)
                .into_iter()
                .collect::<Vec<_>>();
            targets.extend(
                connector
                    .secondary
                    .as_ref()
                    .and_then(|target| region_target(target, region_bindings)),
            );
            objects.push(object(
                connector.connector_id.clone(),
                model,
                FeaStudyObjectKind::Connector(connector_kind(&connector.kind)),
                FeaStudyObjectDetails::Connector(connector.kind.clone()),
                targets,
                sources.model_object(
                    &model.module_path,
                    &model.model_id,
                    "connector",
                    local_id(&connector.connector_id),
                ),
            ));
        }
        for interface in &model.interfaces {
            let targets = [&interface.primary, &interface.secondary]
                .into_iter()
                .filter_map(|target| region_target(target, region_bindings))
                .collect();
            objects.push(object(
                interface.interface_id.clone(),
                model,
                FeaStudyObjectKind::Interface(interface_kind(&interface.kind)),
                FeaStudyObjectDetails::Interface(interface.kind.clone()),
                targets,
                sources.model_object(
                    &model.module_path,
                    &model.model_id,
                    "interface",
                    local_id(&interface.interface_id),
                ),
            ));
        }
        for signal in &model.signals {
            let presentation = present_signal(signal);
            let mut item = object(
                format!("signal.{}", signal.signal_id().0),
                model,
                FeaStudyObjectKind::Signal,
                FeaStudyObjectDetails::Signal(presentation.clone()),
                Vec::new(),
                sources.model_object(
                    &model.module_path,
                    &model.model_id,
                    "signal",
                    &signal.signal_id().0,
                ),
            );
            item.frame = signal.frame().cloned();
            item.signal = Some(presentation);
            objects.push(item);
        }
        for signal in &model.parameterized_signals {
            let presentation = present_parameterized_signal(signal);
            let mut item = object(
                format!("signal.{}", signal.signal_id.0),
                model,
                FeaStudyObjectKind::Signal,
                FeaStudyObjectDetails::Signal(presentation.clone()),
                Vec::new(),
                sources.model_object(
                    &model.module_path,
                    &model.model_id,
                    "signal",
                    &signal.signal_id.0,
                ),
            );
            item.frame = signal.frame.clone();
            item.signal = Some(presentation);
            objects.push(item);
        }
        for excitation in &model.excitations {
            let targets = match &excitation.target {
                runmat_analysis_core::AuthoredExcitationTarget::Region(target) => {
                    region_target(target, region_bindings).into_iter().collect()
                }
                runmat_analysis_core::AuthoredExcitationTarget::WholeModel => {
                    vec![FeaPresentationTarget::WholeModel]
                }
            };
            let mut item = object(
                excitation.excitation_id.clone(),
                model,
                FeaStudyObjectKind::Excitation(excitation.kind),
                FeaStudyObjectDetails::Excitation(excitation.clone()),
                targets,
                sources.model_object(
                    &model.module_path,
                    &model.model_id,
                    if excitation.excitation_id.starts_with("load.") {
                        "load"
                    } else {
                        "excitation"
                    },
                    local_id(&excitation.excitation_id),
                ),
            );
            item.frame = Some(excitation.frame.clone());
            item.direction = excitation.direction.map(|direction| direction.vector);
            item.distribution = Some(excitation.distribution);
            item.kinematics = Some(excitation.kinematics);
            item.signal = signals
                .get(excitation.signal_id.0.as_str())
                .map(|signal| present_signal(signal))
                .or_else(|| {
                    parameterized_signals
                        .get(excitation.signal_id.0.as_str())
                        .map(|signal| present_parameterized_signal(signal))
                });
            item.active_step_ids = excitation.active_step_ids.clone();
            item.active_case_ids = case_membership
                .get(excitation.excitation_id.as_str())
                .cloned()
                .unwrap_or_default();
            objects.push(item);
        }
        for condition in &model.initial_conditions {
            let targets = match &condition.target {
                AuthoredInitialConditionTarget::Region(target) => {
                    region_target(target, region_bindings).into_iter().collect()
                }
                AuthoredInitialConditionTarget::WholeModel => {
                    vec![FeaPresentationTarget::WholeModel]
                }
            };
            let mut item = object(
                condition.initial_condition_id.clone(),
                model,
                FeaStudyObjectKind::InitialCondition(initial_condition_kind(&condition.kind)),
                FeaStudyObjectDetails::InitialCondition(condition.kind.clone()),
                targets,
                sources.model_object(
                    &model.module_path,
                    &model.model_id,
                    "initial",
                    local_id(&condition.initial_condition_id),
                ),
            );
            item.frame = Some(condition.frame.clone());
            objects.push(item);
        }
        for case in &model.excitation_cases {
            let mut item = object(
                case.case_id.clone(),
                model,
                FeaStudyObjectKind::ExcitationCase,
                FeaStudyObjectDetails::ExcitationCase(case.clone()),
                Vec::new(),
                sources.model_object(
                    &model.module_path,
                    &model.model_id,
                    "load_case",
                    local_id(&case.case_id),
                ),
            );
            item.active_step_ids = activation_steps(&case.activation, all_step_ids);
            objects.push(item);
        }
        for spectrum in &model.response_spectra {
            let sample_count = match &spectrum.abscissa {
                runmat_analysis_core::SpectrumAbscissa::FrequencyHertz(values)
                | runmat_analysis_core::SpectrumAbscissa::PeriodSeconds(values) => values.len(),
            };
            objects.push(object(
                spectrum.spectrum_id.0.clone(),
                model,
                FeaStudyObjectKind::ResponseSpectrum,
                FeaStudyObjectDetails::ResponseSpectrum(FeaResponseSpectrumPresentation {
                    spectrum_id: spectrum.spectrum_id.0.clone(),
                    sample_count,
                    frame: spectrum.direction.frame.clone(),
                    direction: spectrum.direction.vector,
                }),
                Vec::new(),
                sources.model_object(
                    &model.module_path,
                    &model.model_id,
                    "response_spectrum",
                    local_id(&spectrum.spectrum_id.0),
                ),
            ));
        }
        for excitation in &model.random_excitations {
            objects.push(object(
                excitation.excitation_id.0.clone(),
                model,
                FeaStudyObjectKind::RandomExcitation,
                FeaStudyObjectDetails::RandomExcitation(FeaRandomExcitationPresentation {
                    excitation_id: excitation.excitation_id.0.clone(),
                    frequency_sample_count: excitation.frequencies_hz.len(),
                    channel_count: excitation.channels.len(),
                }),
                Vec::new(),
                sources.model_object(
                    &model.module_path,
                    &model.model_id,
                    "random_excitation",
                    local_id(&excitation.excitation_id.0),
                ),
            ));
        }
    }
    for item in &mut objects {
        if item.active_step_ids.is_empty() {
            item.active_step_ids = active_steps
                .get(&item.object_id)
                .cloned()
                .unwrap_or_default();
        }
    }
    objects
}

fn object(
    object_id: String,
    model: &AuthoredModel,
    kind: FeaStudyObjectKind,
    details: FeaStudyObjectDetails,
    targets: Vec<FeaPresentationTarget>,
    source: Option<runmat_analysis_language::FeaDiagnosticLocation>,
) -> FeaStudyObjectPresentation {
    FeaStudyObjectPresentation {
        object_id,
        model_id: model.model_id.clone(),
        kind,
        details,
        targets,
        frame: None,
        direction: None,
        distribution: None,
        kinematics: None,
        signal: None,
        active_step_ids: Vec::new(),
        active_case_ids: Vec::new(),
        source,
    }
}

fn region_target(
    source: &AuthoredRegionSource,
    bindings: &BTreeMap<String, GeometryRegionRef>,
) -> Option<FeaPresentationTarget> {
    match source {
        AuthoredRegionSource::Geometry(region) => {
            Some(FeaPresentationTarget::GeometryRegion(region.clone()))
        }
        AuthoredRegionSource::ModelInput(input) => bindings
            .get(&input.input_id)
            .cloned()
            .map(FeaPresentationTarget::GeometryRegion)
            .or_else(|| match input.value_type {
                runmat_analysis_core::AuthoredInputType::Region(role) => {
                    Some(FeaPresentationTarget::ModelInput {
                        input_id: input.input_id.clone(),
                        role,
                    })
                }
                _ => None,
            }),
    }
}

fn activation_steps(
    activation: &runmat_analysis_core::ExcitationCaseActivation,
    all_step_ids: &[String],
) -> Vec<String> {
    match activation {
        runmat_analysis_core::ExcitationCaseActivation::Always
        | runmat_analysis_core::ExcitationCaseActivation::MutuallyExclusiveGroup { .. } => {
            all_step_ids.to_vec()
        }
        runmat_analysis_core::ExcitationCaseActivation::Steps(steps) => steps.clone(),
    }
}

fn property_target(
    target: &AuthoredPropertyTarget,
    bindings: &BTreeMap<String, GeometryRegionRef>,
) -> Vec<FeaPresentationTarget> {
    let source = match target {
        AuthoredPropertyTarget::Beam(source)
        | AuthoredPropertyTarget::Shell(source)
        | AuthoredPropertyTarget::Solid(source)
        | AuthoredPropertyTarget::Cell(source) => source,
    };
    region_target(source, bindings).into_iter().collect()
}

fn property_assignment_targets(
    assignment: &AuthoredPropertyAssignment,
    bindings: &BTreeMap<String, GeometryRegionRef>,
) -> Vec<FeaPresentationTarget> {
    match assignment {
        AuthoredPropertyAssignment::Section { target, .. }
        | AuthoredPropertyAssignment::Orientation { target, .. } => {
            property_target(target, bindings)
        }
        AuthoredPropertyAssignment::Layup { target, .. } => {
            region_target(target, bindings).into_iter().collect()
        }
    }
}

fn excitation_case_membership(model: &AuthoredModel) -> BTreeMap<&str, Vec<String>> {
    let mut membership = BTreeMap::<&str, Vec<String>>::new();
    for case in &model.excitation_cases {
        for excitation in &case.excitation_ids {
            membership
                .entry(excitation)
                .or_default()
                .push(case.case_id.clone());
        }
    }
    membership
}

fn local_id(id: &str) -> &str {
    id.rsplit_once('.').map_or(id, |(_, local)| local)
}
