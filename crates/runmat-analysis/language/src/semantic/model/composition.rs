use super::diagnostic;
use crate::{CheckedFeaDeclaration, FeaDiagnostic, FeaSymbolKey, LogicalSourcePath};
use runmat_analysis_core::{
    AuthoredBoundary, AuthoredConnector, AuthoredConstraint, AuthoredExcitation,
    AuthoredExcitationCase, AuthoredInitialCondition, AuthoredInterface, AuthoredLayupSource,
    AuthoredMaterialSource, AuthoredModel, AuthoredModelRef, AuthoredOrientationSource,
    AuthoredPropertyAssignment, AuthoredPropertyTarget, AuthoredRegionSource,
    AuthoredSectionSource, PhysicalSignal, RandomExcitationSpectrum, ResponseSpectrum,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn validate(
    models: &mut BTreeMap<(LogicalSourcePath, FeaSymbolKey), AuthoredModel>,
    declarations: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    let inventory = models
        .iter()
        .map(|(key, model)| {
            (
                AuthoredModelRef {
                    module_path: model.module_path.clone(),
                    model_id: model.model_id.clone(),
                },
                key.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let keys = models.keys().cloned().collect::<Vec<_>>();
    let mut invalid = BTreeSet::new();
    for key in keys {
        let mut stack = BTreeSet::new();
        let mut assignments = CompositionAssignments::default();
        if let Err(message) = collect(&key, models, &inventory, &mut stack, &mut assignments) {
            invalid.insert(key.clone());
            if let Some(declaration) = declarations.get(&key) {
                diagnostic(declaration, declaration.span, message, diagnostics);
            }
        }
    }
    models.retain(|key, _| !invalid.contains(key));
}

fn collect(
    key: &(LogicalSourcePath, FeaSymbolKey),
    models: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), AuthoredModel>,
    inventory: &BTreeMap<AuthoredModelRef, (LogicalSourcePath, FeaSymbolKey)>,
    stack: &mut BTreeSet<(LogicalSourcePath, FeaSymbolKey)>,
    assignments: &mut CompositionAssignments,
) -> Result<(), String> {
    if !stack.insert(key.clone()) {
        return Err("model composition contains an include cycle".into());
    }
    let model = models
        .get(key)
        .ok_or_else(|| "included model is not available for composition".to_string())?;
    for include in &model.includes {
        let target = inventory.get(&include.model).ok_or_else(|| {
            format!(
                "included model `{}::{}` is not available",
                include.model.module_path, include.model.model_id
            )
        })?;
        let included = models
            .get(target)
            .ok_or_else(|| "included model is not available for composition".to_string())?;
        if !included.inputs.is_empty() {
            return Err(format!(
                "included model `{}` requires explicit input binding",
                included.model_id
            ));
        }
        collect(target, models, inventory, stack, assignments)?;
    }
    collect_materials(model, assignments)?;
    collect_properties(model, assignments)?;
    for constraint in &model.constraints {
        if let Some(existing) = assignments.constraints.get(&constraint.constraint_id) {
            if existing != constraint {
                return Err(format!(
                    "model composition defines conflicting constraint `{}`",
                    constraint.constraint_id
                ));
            }
        } else {
            assignments
                .constraints
                .insert(constraint.constraint_id.clone(), constraint.clone());
        }
    }
    collect_named(
        &model.boundaries,
        &mut assignments.boundaries,
        |value| &value.boundary_id,
        "boundary",
    )?;
    collect_named(
        &model.connectors,
        &mut assignments.connectors,
        |value| &value.connector_id,
        "connector",
    )?;
    collect_named(
        &model.interfaces,
        &mut assignments.interfaces,
        |value| &value.interface_id,
        "interface",
    )?;
    collect_named(
        &model.signals,
        &mut assignments.signals,
        |value| value.signal_id().0.as_str(),
        "signal",
    )?;
    collect_named(
        &model.excitations,
        &mut assignments.excitations,
        |value| value.excitation_id.as_str(),
        "excitation",
    )?;
    collect_named(
        &model.excitation_cases,
        &mut assignments.excitation_cases,
        |value| value.case_id.as_str(),
        "excitation case",
    )?;
    collect_named(
        &model.response_spectra,
        &mut assignments.response_spectra,
        |value| value.spectrum_id.0.as_str(),
        "response spectrum",
    )?;
    collect_named(
        &model.random_excitations,
        &mut assignments.random_excitations,
        |value| value.excitation_id.0.as_str(),
        "random excitation",
    )?;
    collect_named(
        &model.initial_conditions,
        &mut assignments.initial_conditions,
        |value| value.initial_condition_id.as_str(),
        "initial condition",
    )?;
    stack.remove(key);
    Ok(())
}

fn collect_materials(
    model: &AuthoredModel,
    assignments: &mut CompositionAssignments,
) -> Result<(), String> {
    for assignment in &model.material_assignments {
        insert_property(
            &mut assignments.materials,
            &assignment.target,
            &assignment.material,
            "material",
        )?;
        if let AuthoredPropertyTarget::Shell(source) = &assignment.target {
            if assignments.layups.contains_key(source) {
                return Err(
                    "model composition gives a shell both homogeneous and laminate ownership"
                        .into(),
                );
            }
        }
    }
    Ok(())
}

fn collect_properties(
    model: &AuthoredModel,
    assignments: &mut CompositionAssignments,
) -> Result<(), String> {
    for assignment in &model.property_assignments {
        match assignment {
            AuthoredPropertyAssignment::Section { section, target } => {
                insert_property(&mut assignments.sections, target, section, "section")?;
            }
            AuthoredPropertyAssignment::Layup { layup, target } => {
                insert_property(&mut assignments.layups, target, layup, "layup")?;
                if assignments
                    .materials
                    .contains_key(&AuthoredPropertyTarget::Shell(target.clone()))
                {
                    return Err(
                        "model composition gives a shell both homogeneous and laminate ownership"
                            .into(),
                    );
                }
            }
            AuthoredPropertyAssignment::Orientation {
                orientation,
                target,
            } => insert_property(
                &mut assignments.orientations,
                target,
                orientation,
                "orientation",
            )?,
        }
    }
    Ok(())
}

#[derive(Default)]
struct CompositionAssignments {
    materials: BTreeMap<AuthoredPropertyTarget, AuthoredMaterialSource>,
    sections: BTreeMap<AuthoredPropertyTarget, AuthoredSectionSource>,
    layups: BTreeMap<AuthoredRegionSource, AuthoredLayupSource>,
    orientations: BTreeMap<AuthoredPropertyTarget, AuthoredOrientationSource>,
    constraints: BTreeMap<String, AuthoredConstraint>,
    boundaries: BTreeMap<String, AuthoredBoundary>,
    connectors: BTreeMap<String, AuthoredConnector>,
    interfaces: BTreeMap<String, AuthoredInterface>,
    signals: BTreeMap<String, PhysicalSignal>,
    excitations: BTreeMap<String, AuthoredExcitation>,
    excitation_cases: BTreeMap<String, AuthoredExcitationCase>,
    response_spectra: BTreeMap<String, ResponseSpectrum>,
    random_excitations: BTreeMap<String, RandomExcitationSpectrum>,
    initial_conditions: BTreeMap<String, AuthoredInitialCondition>,
}

fn collect_named<T: Clone + PartialEq>(
    values: &[T],
    collected: &mut BTreeMap<String, T>,
    identity: impl Fn(&T) -> &str,
    label: &str,
) -> Result<(), String> {
    for value in values {
        let id = identity(value);
        if let Some(existing) = collected.get(id) {
            if existing != value {
                return Err(format!(
                    "model composition defines conflicting {label} `{id}`"
                ));
            }
        } else {
            collected.insert(id.to_string(), value.clone());
        }
    }
    Ok(())
}

fn insert_property<K, V>(
    assignments: &mut BTreeMap<K, V>,
    target: &K,
    value: &V,
    label: &str,
) -> Result<(), String>
where
    K: Ord + Clone,
    V: PartialEq + Clone,
{
    if let Some(existing) = assignments.get(target) {
        if existing != value {
            return Err(format!(
                "model composition assigns conflicting {label} definitions to one target"
            ));
        }
    } else {
        assignments.insert(target.clone(), value.clone());
    }
    Ok(())
}
