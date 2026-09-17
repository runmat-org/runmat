use super::{CheckedCallArgument, CheckedExpression};
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
            CheckedCallArgument::Named { name, value } => {
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

fn pattern_matches(pattern: &OperationTypePattern, value: &FeaType) -> bool {
    match pattern {
        OperationTypePattern::Any => true,
        OperationTypePattern::NumericScalar => value.is_numeric(),
        OperationTypePattern::IntegerScalar | OperationTypePattern::Size => {
            matches!(value, FeaType::Integer)
        }
        OperationTypePattern::StringScalar
        | OperationTypePattern::Style
        | OperationTypePattern::PropertyName => matches!(value, FeaType::String | FeaType::Enum),
        OperationTypePattern::NumericArray | OperationTypePattern::Axes => {
            matches!(value, FeaType::Vector { element, .. } if element.is_numeric())
        }
        OperationTypePattern::LogicalArray => match value {
            FeaType::Boolean => true,
            FeaType::Vector { element, .. } => matches!(element.as_ref(), FeaType::Boolean),
            _ => false,
        },
        OperationTypePattern::Prototype | OperationTypePattern::PropertyValue => true,
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
                .is_some_and(|value| value.value_type == FeaType::Quantity(dimension.into()))
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
            output = FeaType::Quantity(dimension.into());
        }
        OperationQuantityRule::NotApplicable | OperationQuantityRule::CheckedByLowering => {}
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
        | OperationTypePattern::PropertyValue => FeaType::Unknown,
        OperationTypePattern::NumericScalar => FeaType::Real,
        OperationTypePattern::IntegerScalar | OperationTypePattern::Size => FeaType::Integer,
        OperationTypePattern::StringScalar
        | OperationTypePattern::Style
        | OperationTypePattern::PropertyName => FeaType::String,
        OperationTypePattern::NumericArray | OperationTypePattern::Axes => FeaType::Vector {
            element: Box::new(FeaType::Real),
            length: None,
        },
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
        AnalysisValueType::Region => FeaType::Unknown,
        AnalysisValueType::Material => FeaType::Declaration(FeaSymbolKind::Material),
        AnalysisValueType::Model => FeaType::Declaration(FeaSymbolKind::Model),
        AnalysisValueType::Study => FeaType::Declaration(FeaSymbolKind::Study),
        AnalysisValueType::Sweep => FeaType::Declaration(FeaSymbolKind::Sweep),
        AnalysisValueType::Field => FeaType::Field(Box::new(FeaType::Unknown)),
        AnalysisValueType::History => FeaType::History(Box::new(FeaType::Unknown)),
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
        AnalysisValueType::Field => matches!(value, FeaType::Field(_)),
        AnalysisValueType::History => matches!(value, FeaType::History(_)),
        AnalysisValueType::AttributedExtremum => {
            matches!(value, FeaType::AttributedExtremum(_))
        }
        AnalysisValueType::CaseTable => matches!(value, FeaType::CaseTable(_)),
        _ => analysis_type(expected) == *value,
    }
}
