//! Shared, typed static facts for the FEA object constructors `fea.study`,
//! `fea.sweep`, `fea.model`, and `fea.material`. Runtime decoding remains the
//! authority for nested component payloads and geometry binding.

use super::identities::{
    FEA_EXECUTION_PLAN_CLASS, FEA_MATERIAL_CLASS, FEA_MODEL_CLASS,
    FEA_PROPERTY_REGION_TARGET_CLASS, FEA_STUDY_CLASS, GEOMETRY_ASSET_CLASS, MESHING_MESH_CLASS,
};
use super::options::{FeaModelOption, FeaStudyOption, FeaSweepOption};
use crate::{
    catalog::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{CallInference, CallRequest, InferenceDiagnostic};
use std::collections::BTreeSet;

mod facts;
mod kind;

pub(crate) use facts::{
    dynamic_object_fact, host_text, known_class, object_fact, possible_component_list,
    possible_exact_scalar, possible_logical_scalar, possible_object, possible_object_or_cell,
    resident,
};
pub use kind::FeaConstructorKind;

pub fn infer_constructor(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    kind: FeaConstructorKind,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let arguments = &request.arguments;
    let positional = kind.positional();
    if arguments.len() < positional || !(arguments.len() - positional).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FEA-ARITY",
            format!(
                "{} requires {positional} positional argument(s) followed by complete Name, Value pairs",
                kind.name()
            ),
            arguments.len(),
        ));
    }
    if kind != FeaConstructorKind::MaterialAssignment
        && arguments.first().is_some_and(|id| !host_text(id))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-FEA-TEXT",
            format!("{} id must be host text", kind.name()),
            0,
        ));
    }
    match kind {
        FeaConstructorKind::Study | FeaConstructorKind::Model => {
            if arguments
                .get(1)
                .is_some_and(|geometry| !possible_object(geometry, &[GEOMETRY_ASSET_CLASS]))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-GEOMETRY",
                    "FEA model and study geometry must be a host geometry.Asset",
                    1,
                ));
            }
        }
        FeaConstructorKind::Sweep => {
            if arguments
                .get(1)
                .is_some_and(|studies| !possible_object_or_cell(studies, &[FEA_STUDY_CLASS]))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-STUDIES",
                    "fea.sweep requires a host fea.Study or cell array of studies",
                    1,
                ));
            }
        }
        FeaConstructorKind::MaterialAssignment => {
            if arguments
                .first()
                .is_some_and(|target| !possible_object(target, &[FEA_PROPERTY_REGION_TARGET_CLASS]))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-TARGET",
                    "fea.materialAssignment requires a fea.PropertyRegionTarget; region text is not accepted",
                    0,
                ));
            }
            if arguments
                .get(1)
                .is_some_and(|material| !possible_object(material, &[FEA_MATERIAL_CLASS]))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-MATERIAL",
                    "fea.materialAssignment requires a fea.Material object; material ids are not accepted",
                    1,
                ));
            }
        }
        FeaConstructorKind::Material => {}
    }
    if kind == FeaConstructorKind::Study
        && arguments
            .get(2)
            .is_some_and(|model| !possible_object(model, &[FEA_MODEL_CLASS]))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-FEA-MODEL",
            "fea.study requires an explicit host fea.Model",
            2,
        ));
    }
    infer_options(request, kind, positional, &mut diagnostics);
    finish_fixed(entry, request, object_fact(kind.class()), diagnostics)
}

fn infer_options(
    request: &CallRequest,
    kind: FeaConstructorKind,
    positional: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let arguments = &request.arguments;
    let mut seen = BTreeSet::new();
    let mut all_names_known = true;
    let mut has_steps = false;
    let mut has_sequence = false;
    for index in (positional..arguments.len()).step_by(2) {
        if !host_text(&arguments[index]) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEA-OPTION",
                format!("{} option names must be host text", kind.name()),
                index,
            ));
        }
        let value = arguments.get(index + 1);
        if value.is_some_and(resident) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEA-RESIDENT",
                "FEA constructor option values are not implicitly gathered from a device",
                index + 1,
            ));
        }
        let Some(name) = request.literals.literal_string_at(index) else {
            all_names_known = false;
            continue;
        };
        match kind {
            FeaConstructorKind::Study | FeaConstructorKind::Sweep => {
                let (allowed, class) = match kind {
                    FeaConstructorKind::Study => match FeaStudyOption::decode(&name) {
                        Some(FeaStudyOption::Execution) => (true, Some(FEA_EXECUTION_PLAN_CLASS)),
                        Some(FeaStudyOption::Mesh) => (true, Some(MESHING_MESH_CLASS)),
                        None => (false, None),
                    },
                    _ => (FeaSweepOption::decode(&name).is_some(), None),
                };
                if !allowed {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-OPTION",
                        format!("unsupported {} option", kind.name()),
                        index,
                    ));
                    continue;
                }
                if !seen.insert(super::options::normalize_option_key(&name)) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-DUPLICATE",
                        format!("duplicate {} option", kind.name()),
                        index,
                    ));
                }
                let valid = match (class, value) {
                    (Some(class), Some(value)) => possible_object(value, &[class]),
                    (None, Some(value)) => possible_logical_scalar(value),
                    (_, None) => true,
                };
                if !valid {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-OPTION-VALUE",
                        match class {
                            Some(_) => "option value has the wrong FEA object class",
                            None => "FailFast must be a host logical scalar",
                        },
                        index + 1,
                    ));
                }
            }
            FeaConstructorKind::Model => {
                let Some(option) = FeaModelOption::decode(&name) else {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-OPTION",
                        "unsupported fea.model option",
                        index,
                    ));
                    continue;
                };
                has_steps |= option == FeaModelOption::Steps;
                has_sequence |= option == FeaModelOption::Sequence;
                let valid = value.is_none_or(|value| match option {
                    FeaModelOption::Frame => host_text(value),
                    _ => possible_component_list(value),
                });
                if !valid {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-OPTION-VALUE",
                        "fea.model Frame is text; component options are typed FEA objects or cells of them",
                        index + 1,
                    ));
                }
            }
            FeaConstructorKind::MaterialAssignment => {
                let normalized = super::options::normalize_option_key(&name);
                let valid = match normalized.as_str() {
                    "expectedmaterial" | "expectedmaterialid" => {
                        value.is_none_or(|value| possible_object(value, &[FEA_MATERIAL_CLASS]))
                    }
                    "confidence" => {
                        value.is_none_or(host_text)
                            && request
                                .literals
                                .literal_string_at(index + 1)
                                .is_none_or(|text| {
                                    matches!(
                                        text.to_ascii_lowercase().as_str(),
                                        "verified" | "probable" | "inferred"
                                    )
                                })
                    }
                    _ => {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-FEA-OPTION",
                            "unsupported fea.materialAssignment option",
                            index,
                        ));
                        continue;
                    }
                };
                if !valid {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-OPTION-VALUE",
                        "ExpectedMaterial is a fea.Material; Confidence is verified, probable, or inferred",
                        index + 1,
                    ));
                }
            }
            FeaConstructorKind::Material => {}
        }
    }
    if kind == FeaConstructorKind::Model {
        if has_steps && has_sequence {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEA-SEQUENCE",
                "fea.model Sequence and Steps are mutually exclusive",
                positional,
            ));
        } else if all_names_known && !has_steps && !has_sequence {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEA-SEQUENCE",
                "fea.model requires exactly one of Sequence or Steps",
                positional,
            ));
        }
    }
    if kind == FeaConstructorKind::Material && arguments.len() == positional {
        diagnostics.push(argument_error(
            "RM-CATALOG-FEA-MATERIAL-EMPTY",
            "fea.material requires at least one explicit constitutive capability",
            positional,
        ));
    }
}

#[cfg(test)]
mod tests;
