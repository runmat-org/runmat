use super::{CheckedCallArgument, CheckedExpression, DeclarationSignature};
use crate::{
    FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase, FeaSymbolKind, FeaType,
    LogicalSourcePath,
};
use runmat_types::{
    AnalysisValueType, OperationOverloadFacts, OperationParameter, OperationParameterArity,
    OperationQuantityRule, OperationTypePattern, Span,
};
use std::collections::BTreeSet;

pub(super) fn arguments_match(
    parameters: &[OperationParameter],
    arguments: &[CheckedCallArgument],
) -> bool {
    let mut assigned = BTreeSet::new();
    let mut positional = 0usize;
    for argument in arguments {
        let (parameter_index, value) = match argument {
            CheckedCallArgument::Positional(value) => {
                while parameters.get(positional).is_some_and(|parameter| {
                    assigned.contains(&positional)
                        || parameter.arity == OperationParameterArity::Optional
                            && arguments.iter().any(|candidate| {
                                matches!(candidate, CheckedCallArgument::Named { name, .. } if name == &parameter.name)
                            })
                }) {
                    positional += 1;
                }
                (positional, value)
            }
            CheckedCallArgument::Named { name, value, .. } => {
                let Some(index) = parameters
                    .iter()
                    .position(|parameter| &parameter.name == name)
                else {
                    return false;
                };
                (index, value)
            }
        };
        let Some(parameter) = parameters.get(parameter_index) else {
            return parameters
                .last()
                .is_some_and(|parameter| parameter.arity == OperationParameterArity::Variadic);
        };
        if !assigned.insert(parameter_index) && parameter.arity != OperationParameterArity::Variadic
        {
            return false;
        }
        if !pattern_matches(&parameter.value, &value.value_type) {
            return false;
        }
        if matches!(argument, CheckedCallArgument::Positional(_))
            && parameter.arity != OperationParameterArity::Variadic
        {
            positional += 1;
        }
    }
    parameters.iter().enumerate().all(|(index, parameter)| {
        parameter.arity != OperationParameterArity::Required || assigned.contains(&index)
    })
}

pub(super) fn declaration_arguments_match(
    signature: &DeclarationSignature,
    arguments: &[CheckedCallArgument],
) -> bool {
    let ordered = signature.inputs.iter().collect::<Vec<_>>();
    let mut assigned = std::collections::BTreeMap::<String, &CheckedExpression>::new();
    let mut positional = 0usize;
    for argument in arguments {
        let (name, value) = match argument {
            CheckedCallArgument::Positional(value) => {
                let Some((name, _)) = ordered.get(positional) else {
                    return false;
                };
                positional += 1;
                ((*name).clone(), value)
            }
            CheckedCallArgument::Named { name, value, .. } => (name.clone(), value),
        };
        let Some(input) = signature.inputs.get(&name) else {
            return false;
        };
        if assigned.insert(name, value).is_some() || !input.value_type.accepts(&value.value_type) {
            return false;
        }
    }
    signature
        .inputs
        .iter()
        .all(|(name, input)| !input.required || assigned.contains_key(name))
}

pub(crate) fn pattern_matches(pattern: &OperationTypePattern, value: &FeaType) -> bool {
    match pattern {
        OperationTypePattern::Any => true,
        OperationTypePattern::NumericScalar => value.is_numeric(),
        OperationTypePattern::Quantity(dimension) => (*value).is_quantity(*dimension),
        OperationTypePattern::VectorQuantity { dimension, length } => matches!(
            value,
            FeaType::Vector {
                element,
                length: value_length,
            } if element.is_quantity(*dimension)
                && length.is_none_or(|expected| *value_length == Some(expected))
        ),
        OperationTypePattern::IntegerScalar | OperationTypePattern::Size => {
            matches!(value, FeaType::Integer)
        }
        OperationTypePattern::StringScalar
        | OperationTypePattern::Style
        | OperationTypePattern::PropertyName => matches!(value, FeaType::String | FeaType::Enum),
        OperationTypePattern::NumericArray | OperationTypePattern::Axes => {
            matches!(value, FeaType::Vector { element, .. } if element.is_numeric())
        }
        OperationTypePattern::LogicalScalar => matches!(value, FeaType::Boolean),
        OperationTypePattern::Record(fields) => match value {
            FeaType::Record(values) => {
                fields.len() == values.len()
                    && fields.iter().all(|(key, pattern)| {
                        values
                            .get(key)
                            .is_some_and(|value| pattern_matches(pattern, value))
                    })
            }
            _ => false,
        },
        OperationTypePattern::LogicalArray => match value {
            FeaType::Boolean => true,
            FeaType::Vector { element, .. } => matches!(element.as_ref(), FeaType::Boolean),
            _ => false,
        },
        OperationTypePattern::Prototype | OperationTypePattern::PropertyValue => true,
        // FEA source has no function values; callables come only from host scripts.
        OperationTypePattern::Callable => false,
        OperationTypePattern::Analysis(expected) => analysis_pattern_matches(*expected, value),
        OperationTypePattern::List(element) => match value {
            FeaType::List(value) => pattern_matches(element, value),
            FeaType::Vector { element: value, .. } => pattern_matches(element, value),
            _ => false,
        },
    }
}

pub(super) fn outputs_type(outputs: &[OperationTypePattern]) -> FeaType {
    match outputs {
        [] => FeaType::Unknown,
        [output] => pattern_type(output),
        outputs => FeaType::Vector {
            element: Box::new(
                outputs
                    .iter()
                    .map(pattern_type)
                    .reduce(|left, right| super::type_rules::numeric_join(&left, &right))
                    .unwrap_or(FeaType::Unknown),
            ),
            length: Some(outputs.len()),
        },
    }
}

pub(super) fn operation_output_type(
    overload: &OperationOverloadFacts,
    arguments: &[CheckedCallArgument],
    diagnostics: &mut Vec<FeaDiagnostic>,
    path: &LogicalSourcePath,
    span: Span,
) -> FeaType {
    let mut output = outputs_type(
        &overload
            .signature
            .outputs
            .iter()
            .map(|parameter| parameter.value.clone())
            .collect::<Vec<_>>(),
    );
    match overload.quantity {
        OperationQuantityRule::PreserveInput { input } => {
            if let Some(value) = argument_value(arguments, input) {
                output = value.value_type.clone();
            }
        }
        OperationQuantityRule::RequiredInput { input, dimension } => {
            if !argument_value(arguments, input)
                .is_some_and(|value| (value.value_type).is_quantity(dimension))
            {
                diagnostics.push(FeaDiagnostic::error(
                    "fea.semantic.operation_quantity",
                    "operation input has the wrong physical dimension",
                    FeaDiagnosticPhase::Semantic,
                    Some(FeaDiagnosticLocation {
                        path: path.clone(),
                        span,
                    }),
                ));
                output = FeaType::Error;
            }
        }
        OperationQuantityRule::FixedOutput {
            output: _,
            dimension,
        } => {
            output = FeaType::Quantity(dimension);
        }
        OperationQuantityRule::NotApplicable | OperationQuantityRule::CheckedByLowering => {}
    }
    if let Some(operation) = crate::operations::OutputOperation::from_lowering(&overload.lowering) {
        match operation.result_type(arguments) {
            Ok(value_type) => output = value_type,
            Err(message) => {
                diagnostics.push(FeaDiagnostic::error(
                    "fea.semantic.output_operation",
                    message,
                    FeaDiagnosticPhase::Semantic,
                    Some(FeaDiagnosticLocation {
                        path: path.clone(),
                        span,
                    }),
                ));
                output = FeaType::Error;
            }
        }
    }
    output
}

fn argument_value(arguments: &[CheckedCallArgument], index: usize) -> Option<&CheckedExpression> {
    arguments.get(index).map(|argument| match argument {
        CheckedCallArgument::Positional(value) | CheckedCallArgument::Named { value, .. } => value,
    })
}

fn pattern_type(pattern: &OperationTypePattern) -> FeaType {
    match pattern {
        OperationTypePattern::Any
        | OperationTypePattern::Prototype
        | OperationTypePattern::PropertyValue
        | OperationTypePattern::Callable => FeaType::Unknown,
        OperationTypePattern::NumericScalar => FeaType::Real,
        OperationTypePattern::Quantity(dimension) => FeaType::Quantity(*dimension),
        OperationTypePattern::VectorQuantity { dimension, length } => FeaType::Vector {
            element: Box::new(FeaType::Quantity(*dimension)),
            length: *length,
        },
        OperationTypePattern::IntegerScalar | OperationTypePattern::Size => FeaType::Integer,
        OperationTypePattern::StringScalar
        | OperationTypePattern::Style
        | OperationTypePattern::PropertyName => FeaType::String,
        OperationTypePattern::NumericArray | OperationTypePattern::Axes => FeaType::Vector {
            element: Box::new(FeaType::Real),
            length: None,
        },
        OperationTypePattern::LogicalScalar => FeaType::Boolean,
        OperationTypePattern::Record(fields) => FeaType::Record(
            fields
                .iter()
                .map(|(key, pattern)| (key.clone(), pattern_type(pattern)))
                .collect(),
        ),
        OperationTypePattern::LogicalArray => FeaType::Vector {
            element: Box::new(FeaType::Boolean),
            length: None,
        },
        OperationTypePattern::Analysis(value) => analysis_type(*value),
        OperationTypePattern::List(value) => FeaType::List(Box::new(pattern_type(value))),
    }
}

fn analysis_type(value: AnalysisValueType) -> FeaType {
    match value {
        AnalysisValueType::Geometry => FeaType::Declaration(FeaSymbolKind::Geometry),
        AnalysisValueType::GeometryFeature => FeaType::GeometryFeature,
        AnalysisValueType::Region => FeaType::Unknown,
        AnalysisValueType::Material => FeaType::Declaration(FeaSymbolKind::Material),
        AnalysisValueType::Model => FeaType::Declaration(FeaSymbolKind::Model),
        AnalysisValueType::Study => FeaType::Declaration(FeaSymbolKind::Study),
        AnalysisValueType::Sweep => FeaType::Declaration(FeaSymbolKind::Sweep),
        AnalysisValueType::Field => FeaType::Field(crate::FeaFieldType {
            value: Box::new(FeaType::Unknown),
            location: None,
            quantity: None,
        }),
        AnalysisValueType::History => FeaType::History(crate::FeaHistoryType {
            sample: Box::new(FeaType::Unknown),
            domain: None,
        }),
        AnalysisValueType::AttributedExtremum => {
            FeaType::AttributedExtremum(Box::new(FeaType::Unknown))
        }
        AnalysisValueType::CaseTable => FeaType::CaseTable(Box::new(FeaType::Unknown)),
        AnalysisValueType::ExecutionProfile => FeaType::Declaration(FeaSymbolKind::Execution),
    }
}

fn analysis_pattern_matches(expected: AnalysisValueType, value: &FeaType) -> bool {
    match expected {
        AnalysisValueType::Region => matches!(value, FeaType::Region(_)),
        AnalysisValueType::GeometryFeature => matches!(value, FeaType::GeometryFeature),
        AnalysisValueType::Field => matches!(value, FeaType::Field(_)),
        AnalysisValueType::History => matches!(value, FeaType::History(_)),
        AnalysisValueType::AttributedExtremum => {
            matches!(value, FeaType::AttributedExtremum(_))
        }
        AnalysisValueType::CaseTable => matches!(value, FeaType::CaseTable(_)),
        _ => analysis_type(expected) == *value,
    }
}
