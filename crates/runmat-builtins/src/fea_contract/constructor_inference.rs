//! Shared, typed static facts for the FEA object-constructor family.

use super::identities::{
    FEA_BOUNDARY_CONDITION_CLASS, FEA_DOMAIN_CLASS, FEA_INTERFACE_CLASS, FEA_LOAD_CASE_CLASS,
    FEA_MATERIAL_ASSIGNMENT_CLASS, FEA_MATERIAL_CLASS, FEA_MODEL_CLASS, FEA_RUN_OPTIONS_CLASS,
    FEA_STEP_CLASS, FEA_STUDY_CLASS, FEA_SWEEP_CLASS,
};

use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CertaintyFact, DimensionFact, DynamicReason, ObjectFact,
    ResidencyFact, ShapeFact, StaticClassIdentity, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeaConstructorKind {
    AuthorStudy,
    Study,
    Sweep,
    Step,
    RunOptions,
    Model,
    Material,
    MaterialAssignment,
    BoundaryCondition,
    LoadCase,
    Domain,
    Interface,
    LoadDocument,
}

impl FeaConstructorKind {
    const fn positional(self) -> usize {
        match self {
            Self::AuthorStudy => 3,
            Self::Study | Self::Sweep | Self::Step | Self::Model | Self::MaterialAssignment => 2,
            Self::BoundaryCondition | Self::LoadCase | Self::Interface => 3,
            Self::RunOptions | Self::Material | Self::Domain | Self::LoadDocument => 1,
        }
    }

    const fn class(self) -> Option<StaticClassIdentity> {
        Some(match self {
            Self::AuthorStudy | Self::Study => FEA_STUDY_CLASS,
            Self::Sweep => FEA_SWEEP_CLASS,
            Self::Step => FEA_STEP_CLASS,
            Self::RunOptions => FEA_RUN_OPTIONS_CLASS,
            Self::Model => FEA_MODEL_CLASS,
            Self::Material => FEA_MATERIAL_CLASS,
            Self::MaterialAssignment => FEA_MATERIAL_ASSIGNMENT_CLASS,
            Self::BoundaryCondition => FEA_BOUNDARY_CONDITION_CLASS,
            Self::LoadCase => FEA_LOAD_CASE_CLASS,
            Self::Domain => FEA_DOMAIN_CLASS,
            Self::Interface => FEA_INTERFACE_CLASS,
            Self::LoadDocument => return None,
        })
    }

    const fn is_document_load(self) -> bool {
        matches!(self, Self::LoadDocument)
    }
}

pub(crate) fn host_text(fact: &ValueFact) -> bool {
    if matches!(fact.residency, ResidencyFact::Device { .. }) {
        return false;
    }
    match fact.kind {
        // Unknown kind with shape [1, N] may still be a character row. Do not
        // reject it merely because a string scalar with that shape is invalid.
        ValueKindFact::Unknown => true,
        ValueKindFact::String => !matches!(fact.shape.element_count(), Some(count) if count != 1),
        ValueKindFact::Character => match &fact.shape {
            ShapeFact::Shaped { dims } => {
                !matches!(dims.first(), Some(DimensionFact::Known(rows)) if *rows != 1)
            }
            ShapeFact::Scalar | ShapeFact::Ranked { rank: 2 } | ShapeFact::Unknown => true,
            _ => false,
        },
        _ => false,
    }
}

fn possible_host_study(fact: &ValueFact) -> bool {
    !matches!(fact.residency, ResidencyFact::Device { .. })
        && match &fact.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => {
                !matches!(fact.shape.element_count(), Some(count) if count != 1)
                    && object
                        .runtime_class
                        .as_ref()
                        .is_none_or(|class| class.is(FEA_STUDY_CLASS))
            }
            _ => false,
        }
}

fn object_result(kind: FeaConstructorKind, dynamic_document: bool) -> ValueFact {
    let mut fact = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: if dynamic_document {
                None
            } else {
                kind.class().map(StaticClassIdentity::owned)
            },
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    if dynamic_document {
        // The document may contain a Study or Sweep; path shape cannot tell.
        fact.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    fact
}

pub fn infer_constructor(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    kind: FeaConstructorKind,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let path_study = kind == FeaConstructorKind::Study && request.arguments.len() == 1;
    let positional = if path_study { 1 } else { kind.positional() };
    let arity_valid = if kind.is_document_load() || path_study {
        request.arguments.len() == 1
    } else if kind == FeaConstructorKind::Step {
        request.arguments.len() == 2
    } else {
        request.arguments.len() >= positional
            && (request.arguments.len() - positional).is_multiple_of(2)
    };
    if !arity_valid {
        diagnostics.push(argument_error(
            "RM-CATALOG-FEA-ARITY",
            "FEA constructor requires its positional values and complete name/value pairs",
            request.arguments.len(),
        ));
    }
    for index in 0..request.arguments.len().min(positional) {
        if matches!(
            kind,
            FeaConstructorKind::Model | FeaConstructorKind::Study | FeaConstructorKind::AuthorStudy
        ) && index == 1
            && !path_study
        {
            let input = &request.arguments[index];
            let valid = match &input.kind {
                ValueKindFact::Unknown => true,
                ValueKindFact::Object(object) => object
                    .runtime_class
                    .as_ref()
                    .is_none_or(|class| class.is(StaticClassIdentity::new("geometry.Asset"))),
                _ => false,
            } && !matches!(input.residency, ResidencyFact::Device { .. })
                && !matches!(input.shape.element_count(), Some(count) if count != 1);
            if !valid {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-GEOMETRY",
                    "FEA model/study authoring geometry must be a host geometry.Asset",
                    index,
                ));
            }
        } else if kind == FeaConstructorKind::AuthorStudy && index == 2 {
            let summary = &request.arguments[index];
            if matches!(summary.residency, ResidencyFact::Device { .. })
                || !matches!(
                    &summary.kind,
                    ValueKindFact::Unknown
                        | ValueKindFact::Struct(_)
                        | ValueKindFact::Object(_)
                        | ValueKindFact::String
                        | ValueKindFact::Character
                )
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-AUTHORING-SUMMARY",
                    "fea.authorStudy requires a host mesh-authoring summary object or JSON text",
                    index,
                ));
            }
        } else if kind == FeaConstructorKind::Sweep && index == 1 {
            let input = &request.arguments[index];
            let valid = match &input.kind {
                ValueKindFact::Cell(cell) => {
                    possible_host_study(&cell.element)
                        && cell.elements.iter().all(possible_host_study)
                        && !matches!(input.residency, ResidencyFact::Device { .. })
                }
                _ => possible_host_study(input),
            };
            if !valid {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-STUDIES",
                    "fea.sweep requires a host fea.Study or cell of studies",
                    index,
                ));
            }
        } else if !host_text(&request.arguments[index]) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEA-TEXT",
                "FEA structural identifiers, kinds, and paths must be host text scalars",
                index,
            ));
        }
    }
    if !kind.is_document_load() && !path_study && kind != FeaConstructorKind::Step {
        let mut all_option_names_known = true;
        let mut has_profile = false;
        for index in (positional..request.arguments.len()).step_by(2) {
            match request.literals.literal_string_at(index) {
                Some(name) => {
                    if name.eq_ignore_ascii_case("profile") {
                        has_profile = true;
                    }
                }
                None => all_option_names_known = false,
            }
            if !host_text(&request.arguments[index]) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-OPTION",
                    "FEA option names must be host text scalars",
                    index,
                ));
            }
            if let Some(value) = request.arguments.get(index + 1) {
                if matches!(value.residency, ResidencyFact::Device { .. }) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-RESIDENT",
                        "FEA constructor option values do not implicitly gather resident data",
                        index + 1,
                    ));
                }
                if kind == FeaConstructorKind::Sweep
                    && request.literals.literal_string_at(index).as_deref() == Some("failfast")
                    && (!matches!(value.kind, ValueKindFact::Logical | ValueKindFact::Unknown)
                        || matches!(value.shape.element_count(), Some(count) if count != 1))
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-FAILFAST",
                        "fea.sweep FailFast must be a host logical scalar",
                        index + 1,
                    ));
                }
            }
        }
        if matches!(
            kind,
            FeaConstructorKind::Model | FeaConstructorKind::Study | FeaConstructorKind::AuthorStudy
        ) && all_option_names_known
            && !has_profile
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEA-PROFILE",
                "FEA model and study authoring constructors require a Profile option",
                positional,
            ));
        }
    }
    // Literal facts here do not certify nested typed component payloads or geometry.
    finish_fixed(
        entry,
        request,
        object_result(kind, kind.is_document_load() || path_study),
        diagnostics,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fea_contract::{
        catalog_entry::{self, ConstructorEntryDefinition},
        descriptors::FEA_MODEL_DESCRIPTOR,
        integer::FEA_STRUCTURAL_INTEGER_AUDIT,
    };
    use crate::{BuiltinCatalogProvenance, BuiltinDocumentation};
    use runmat_types::{
        CellFact, LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount,
    };

    fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        }
    }

    fn entry() -> BuiltinCatalogEntry {
        catalog_entry::entry(ConstructorEntryDefinition {
            provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
            name: "fea.model",
            documentation: BuiltinDocumentation::EMPTY,
            descriptor: &FEA_MODEL_DESCRIPTOR,
            inference: |request, entry| {
                infer_constructor(request, entry, FeaConstructorKind::Model)
            },
            integer_capabilities: &[],
            integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
            filesystem_read: false,
            distributed: crate::BuiltinDistributedPolicy::Unsupported,
        })
    }

    #[test]
    fn path_capability_is_declared_without_changing_pure_constructor_admission() {
        let pure = entry();
        assert!(pure.contract.capabilities.is_empty());
        let path = catalog_entry::entry(ConstructorEntryDefinition {
            provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
            name: "fea.load",
            documentation: BuiltinDocumentation::EMPTY,
            descriptor: &FEA_MODEL_DESCRIPTOR,
            inference: |request, entry| {
                infer_constructor(request, entry, FeaConstructorKind::LoadDocument)
            },
            integer_capabilities: &[],
            integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
            filesystem_read: true,
            distributed: crate::BuiltinDistributedPolicy::Unsupported,
        });
        assert_eq!(
            path.contract.capabilities,
            &[runmat_types::CapabilityRequirement::Filesystem]
        );
    }

    #[test]
    fn model_requires_literal_profile_and_host_geometry_class() {
        assert_eq!(
            entry().contract.compatibility,
            crate::BuiltinCompatibility::RunMat
        );
        let text = ValueFact::scalar(ValueKindFact::String);
        let geometry = ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(StaticClassIdentity::new("geometry.Asset").owned()),
                properties: BTreeMap::new(),
                properties_complete: false,
                handle_semantics: Some(false),
            }),
            ShapeFact::Scalar,
            StorageFact::Opaque,
        );
        let missing = infer_constructor(
            &request(
                vec![text.clone(), geometry.clone()],
                vec![LiteralValue::Unknown; 2],
            ),
            &entry(),
            FeaConstructorKind::Model,
        );
        assert!(missing
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-FEA-PROFILE"));
        let valid = infer_constructor(
            &request(
                vec![text.clone(), geometry, text.clone(), text],
                vec![
                    LiteralValue::Unknown,
                    LiteralValue::Unknown,
                    LiteralValue::String("Profile".into()),
                    LiteralValue::String("linear_static_structural".into()),
                ],
            ),
            &entry(),
            FeaConstructorKind::Model,
        );
        assert!(valid.diagnostics.is_empty());
        assert_eq!(valid.outputs[0].residency, ResidencyFact::Host);
        assert!(
            matches!(&valid.outputs[0].kind, ValueKindFact::Object(object) if object.runtime_class.as_ref().is_some_and(|class| class.is(FEA_MODEL_CLASS)))
        );
    }

    #[test]
    fn document_class_is_dynamic_and_text_arrays_are_rejected() {
        let mut text_array = ValueFact::scalar(ValueKindFact::String);
        text_array.shape = ShapeFact::from(vec![Some(2), Some(1)]);
        let loaded = infer_constructor(
            &request(vec![text_array], vec![LiteralValue::Unknown]),
            &entry(),
            FeaConstructorKind::LoadDocument,
        );
        assert!(loaded
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-FEA-TEXT"));
        assert!(matches!(
            loaded.outputs[0].certainty,
            CertaintyFact::Dynamic(DynamicReason::RuntimeValue)
        ));

        // A partially typed [1, N] value could be a valid character row.
        let mut possible_char_row = ValueFact::unknown(DynamicReason::RuntimeValue);
        possible_char_row.shape = ShapeFact::from(vec![Some(1), Some(7)]);
        let uncertain = infer_constructor(
            &request(vec![possible_char_row], vec![LiteralValue::Unknown]),
            &entry(),
            FeaConstructorKind::LoadDocument,
        );
        assert!(!uncertain
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-FEA-TEXT"));

        // fea.study(path) currently delegates to the same loader and may
        // return a sweep. The path spelling cannot prove Study class.
        let path_form = infer_constructor(
            &request(
                vec![ValueFact::scalar(ValueKindFact::String)],
                vec![LiteralValue::Unknown],
            ),
            &entry(),
            FeaConstructorKind::Study,
        );
        assert!(matches!(
            path_form.outputs[0].certainty,
            CertaintyFact::Dynamic(DynamicReason::RuntimeValue)
        ));
        assert!(
            matches!(&path_form.outputs[0].kind, ValueKindFact::Object(object) if object.runtime_class.is_none())
        );
    }

    #[test]
    fn study_constructor_requires_profile_and_step_has_exact_arity() {
        let text = ValueFact::scalar(ValueKindFact::String);
        let geometry = object_result(FeaConstructorKind::Model, false);
        let mut geometry = geometry;
        if let ValueKindFact::Object(object) = &mut geometry.kind {
            object.runtime_class = Some(StaticClassIdentity::new("geometry.Asset").owned());
        }
        let study = infer_constructor(
            &request(vec![text.clone(), geometry], vec![LiteralValue::Unknown; 2]),
            &entry(),
            FeaConstructorKind::Study,
        );
        assert!(study
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-FEA-PROFILE"));
        let step = infer_constructor(
            &request(
                vec![text.clone(), text.clone(), text],
                vec![LiteralValue::Unknown; 3],
            ),
            &entry(),
            FeaConstructorKind::Step,
        );
        assert!(step
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-FEA-ARITY"));
    }

    #[test]
    fn sweep_rejects_known_non_study_cell_and_numeric_failfast() {
        let text = ValueFact::scalar(ValueKindFact::String);
        let number = ValueFact::scalar(ValueKindFact::Numeric(runmat_types::NumericFact {
            class: runmat_types::NumericClass::Double,
            domain: runmat_types::NumericDomain::Real,
        }));
        let studies = ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(number.clone()),
                elements: vec![number.clone()],
                elements_complete: true,
            }),
            ShapeFact::from(vec![Some(1), Some(1)]),
            StorageFact::Dense,
        );
        let result = infer_constructor(
            &request(
                vec![text.clone(), studies, text, number],
                vec![
                    LiteralValue::Unknown,
                    LiteralValue::Unknown,
                    LiteralValue::String("FailFast".into()),
                    LiteralValue::Unknown,
                ],
            ),
            &entry(),
            FeaConstructorKind::Sweep,
        );
        assert!(result
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-FEA-STUDIES"));
        assert!(result
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-FEA-FAILFAST"));
    }

    #[test]
    fn author_study_requires_summary_and_profile_before_result_class() {
        let text = ValueFact::scalar(ValueKindFact::String);
        let geometry = ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(StaticClassIdentity::new("geometry.Asset").owned()),
                properties: BTreeMap::new(),
                properties_complete: false,
                handle_semantics: Some(false),
            }),
            ShapeFact::Scalar,
            StorageFact::Opaque,
        );
        let numeric_summary =
            ValueFact::scalar(ValueKindFact::Numeric(runmat_types::NumericFact {
                class: runmat_types::NumericClass::Double,
                domain: runmat_types::NumericDomain::Real,
            }));
        let inferred = infer_constructor(
            &request(
                vec![text, geometry, numeric_summary],
                vec![LiteralValue::Unknown; 3],
            ),
            &entry(),
            FeaConstructorKind::AuthorStudy,
        );
        assert!(inferred
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-FEA-AUTHORING-SUMMARY"));
        assert!(inferred
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-FEA-PROFILE"));
        assert!(
            matches!(&inferred.outputs[0].kind, ValueKindFact::Object(object) if object.runtime_class.as_ref().is_some_and(|class| class.is(FEA_STUDY_CLASS)))
        );
    }
}
