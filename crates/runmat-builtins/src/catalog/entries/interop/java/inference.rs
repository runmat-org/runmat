//! Public Java form checks and facts, independent of JVM execution policy.

use std::collections::BTreeMap;

use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

#[derive(Clone, Copy)]
pub(super) enum JavaForm {
    Construct,
    ConstructEdt,
    Method,
    MethodEdt,
    Array,
    AddPath,
    RemovePath,
    Classpath,
    Environment,
    Feature,
    IsJava,
}

fn text_possible(fact: &ValueFact) -> bool {
    match fact.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::String => !fact.shape.is_definitely_nonscalar(),
        ValueKindFact::Character => match fact.shape.known_dims() {
            None => true,
            Some(dims) => {
                dims.first()
                    .is_none_or(|row| row.is_none_or(|row| row == 1))
                    && dims
                        .iter()
                        .skip(2)
                        .all(|page| page.is_none_or(|page| page == 1))
            }
        },
        _ => false,
    }
}

fn path_possible(fact: &ValueFact) -> bool {
    if text_possible(fact) {
        return true;
    }
    match &fact.kind {
        ValueKindFact::Cell(cell) => {
            text_possible(&cell.element) && cell.elements.iter().all(text_possible)
        }
        _ => false,
    }
}

fn checked_text(request: &CallRequest, index: usize, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if request
        .arguments
        .get(index)
        .is_some_and(|fact| !text_possible(fact))
    {
        diagnostics.push(argument_error(
            "RM-JAVA-TEXT",
            "Java class, method, option, and feature names must be text",
            index,
        ));
    }
}

fn arity(
    request: &CallRequest,
    minimum: usize,
    maximum: Option<usize>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if request.arguments.len() < minimum
        || maximum.is_some_and(|maximum| request.arguments.len() > maximum)
    {
        diagnostics.push(argument_error(
            "RM-JAVA-ARITY",
            "Java operation received an unsupported number of arguments",
            0,
        ));
    }
}

pub(super) fn environment() -> ValueFact {
    let text = ValueFact::scalar(ValueKindFact::String);
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(super::JAVA_ENVIRONMENT_CLASS.owned()),
            properties: BTreeMap::from([
                ("Version".into(), text.clone()),
                ("Home".into(), text.clone()),
                ("Library".into(), text.clone()),
                ("Status".into(), text.clone()),
                ("Configuration".into(), text),
            ]),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn java_reference() -> ValueFact {
    let mut fact = ValueFact::proven(
        ValueKindFact::Foreign(ForeignFact {
            family: "java".into(),
            type_name: None,
            type_version: None,
            ownership: ForeignOwnershipFact::Shared,
            affinity: ForeignAffinityFact::OriginProcess,
            lifetime: ForeignLifetimeFact::Session,
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    fact.alias = AliasFact::Identity;
    fact.mutation = MutationFact::HandleSemantics;
    fact
}

fn classpath() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::Unknown,
                StorageFact::Opaque,
            )),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![None, Some(1)]),
        StorageFact::Opaque,
    )
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    form: JavaForm,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let dynamic_classpath_mode = matches!(form, JavaForm::Classpath)
        && request.arguments.len() == 1
        && request.literals.literal_string_at(0).is_none();
    let output = match form {
        JavaForm::Construct | JavaForm::ConstructEdt => {
            arity(request, 1, None, &mut diagnostics);
            checked_text(request, 0, &mut diagnostics);
            vec![java_reference()]
        }
        JavaForm::Method | JavaForm::MethodEdt => {
            arity(request, 2, None, &mut diagnostics);
            checked_text(request, 0, &mut diagnostics);
            if request.arguments.get(1).is_some_and(|fact| {
                !text_possible(fact)
                    && !matches!(
                        fact.kind,
                        ValueKindFact::Foreign(_) | ValueKindFact::Unknown
                    )
            }) {
                diagnostics.push(argument_error(
                    "RM-JAVA-RECEIVER",
                    "Java method receiver must be a class name or Java object",
                    1,
                ));
            }
            vec![ValueFact::unknown(DynamicReason::ForeignBoundary)]
        }
        JavaForm::Array => {
            arity(request, 2, None, &mut diagnostics);
            checked_text(request, 0, &mut diagnostics);
            for index in 1..request.arguments.len() {
                if request.arguments.get(index).is_some_and(|fact| {
                    !matches!(fact.kind, ValueKindFact::Unknown)
                        && (!matches!(fact.kind, ValueKindFact::Numeric(_)) || !fact.is_scalar())
                }) {
                    diagnostics.push(argument_error(
                        "RM-JAVA-DIMENSION",
                        "Java array dimensions must be checked numeric integers",
                        index,
                    ));
                }
            }
            vec![java_reference()]
        }
        JavaForm::AddPath | JavaForm::RemovePath => {
            arity(request, 1, None, &mut diagnostics);
            if matches!(form, JavaForm::AddPath)
                && request.arguments.len() == 1
                && request
                    .literals
                    .literal_string_at(0)
                    .as_deref()
                    .and_then(super::JavaInsertionPosition::from_public_terminal)
                    .is_some()
            {
                diagnostics.push(argument_error(
                    "RM-JAVA-PATH",
                    "javaaddpath requires a classpath entry before '-end'",
                    0,
                ));
            }
            for (index, fact) in request.arguments.iter().enumerate() {
                if !path_possible(fact) {
                    diagnostics.push(argument_error(
                        "RM-JAVA-PATH",
                        "Java classpath entries must be text or cells of text",
                        index,
                    ));
                }
            }
            Vec::new()
        }
        JavaForm::Classpath => {
            if request.arguments.len() == 1 {
                if request
                    .arguments
                    .first()
                    .is_some_and(|fact| !path_possible(fact))
                {
                    diagnostics.push(argument_error(
                        "RM-JAVA-PATH",
                        "Java classpath entries must be text or cells of text",
                        0,
                    ));
                }
                match request
                    .literals
                    .literal_string_at(0)
                    .as_deref()
                    .map(super::JavaClasspathControl::parse_public)
                {
                    Some(super::JavaClasspathControl::Query(_)) => vec![classpath()],
                    Some(
                        super::JavaClasspathControl::Verbosity | super::JavaClasspathControl::Path,
                    ) => Vec::new(),
                    None => vec![ValueFact::unknown(DynamicReason::ForeignBoundary)],
                }
            } else if request.arguments.is_empty() {
                vec![classpath()]
            } else {
                for (index, fact) in request.arguments.iter().enumerate() {
                    if !path_possible(fact) {
                        diagnostics.push(argument_error(
                            "RM-JAVA-PATH",
                            "Java classpath entries must be text or cells of text",
                            index,
                        ));
                    }
                }
                Vec::new()
            }
        }
        JavaForm::Environment => {
            if !request.arguments.is_empty() {
                arity(request, 2, Some(2), &mut diagnostics);
                checked_text(request, 0, &mut diagnostics);
                checked_text(request, 1, &mut diagnostics);
                if request
                    .literals
                    .literal_string_at(0)
                    .is_some_and(|name| super::JavaEnvironmentControl::parse(&name).is_none())
                {
                    diagnostics.push(argument_error(
                        "RM-JAVA-OPTION",
                        "jenv supports Version and ExecutionMode",
                        0,
                    ));
                }
                if request.literals.literal_string_at(0).is_some_and(|name| {
                    super::JavaEnvironmentControl::parse(&name)
                        == Some(super::JavaEnvironmentControl::ExecutionMode)
                }) && request
                    .literals
                    .literal_string_at(1)
                    .is_some_and(|mode| super::JavaExecutionMode::parse(&mode).is_none())
                {
                    diagnostics.push(argument_error(
                        "RM-JAVA-EXECUTION-MODE",
                        "jenv currently supports only InProcess execution mode",
                        1,
                    ));
                }
            }
            vec![environment()]
        }
        JavaForm::Feature => {
            arity(request, 1, Some(1), &mut diagnostics);
            checked_text(request, 0, &mut diagnostics);
            if request
                .literals
                .literal_string_at(0)
                .is_some_and(|feature| super::JavaFeature::parse(&feature).is_none())
            {
                diagnostics.push(argument_error(
                    "RM-JAVA-FEATURE",
                    "usejava feature must be jvm, desktop, awt, or swing",
                    0,
                ));
            }
            vec![ValueFact::scalar(ValueKindFact::Logical)]
        }
        JavaForm::IsJava => {
            arity(request, 1, Some(1), &mut diagnostics);
            vec![ValueFact::scalar(ValueKindFact::Logical)]
        }
    };
    let mut contract = if dynamic_classpath_mode {
        CallContract::dynamic(DynamicReason::ForeignBoundary)
    } else {
        CallContract::fixed(output)
    };
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    if matches!(form, JavaForm::Environment) && request.arguments.is_empty() {
        result.effects.0.remove(&EffectKind::EnvironmentWrite);
    }
    if matches!(form, JavaForm::Feature) {
        match request
            .literals
            .literal_string_at(0)
            .and_then(|name| super::JavaFeature::parse(&name))
        {
            Some(super::JavaFeature::Jvm) => {
                result.effects.0.remove(&EffectKind::EnvironmentWrite);
            }
            Some(super::JavaFeature::Desktop) => {
                result.effects.0.remove(&EffectKind::EnvironmentWrite);
                result.effects.0.remove(&EffectKind::FilesystemRead);
            }
            _ => {}
        }
    }
    result
}
