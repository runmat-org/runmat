mod affine;
use super::{parse_unit_expression, FeaType, PhysicalDimension};
use crate::{
    AnalysisOperationCatalog, FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase,
    LogicalSourcePath, OperationResolutionRequest,
};
use runmat_design_syntax::{
    FeaBinaryOperator, FeaCallArgument, FeaExpression, FeaExpressionKind, FeaReference,
    FeaUnaryOperator,
};
use runmat_types::{
    OperationFrontend, OperationIdentity, OperationLoweringId, OperationOverloadIdentity,
    OperationSignature, Span,
};
use std::collections::BTreeMap;

mod checked;
mod matching;
mod reference;
mod scope;
mod type_rules;

pub use checked::{CheckedCallArgument, ResolvedCadParameterValue};
pub(crate) use matching::pattern_matches as operation_pattern_matches;
use matching::{arguments_match, declaration_arguments_match, operation_output_type, outputs_type};
use reference::{contextual_reference_type, is_untyped_zero, quantity_type, reference_path};
pub use scope::{DeclarationInput, DeclarationSignature, ExpressionScope, FunctionSignature};
use type_rules::{divide_type, multiply_type, numeric_join, types_compatible};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckedExpression {
    pub kind: CheckedExpressionKind,
    pub value_type: FeaType,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckedExpressionKind {
    Number(String),
    String(String),
    Boolean(bool),
    Enum(String),
    Reference(Vec<String>),
    CadParameterValue(ResolvedCadParameterValue),
    List(Vec<CheckedExpression>),
    Record(BTreeMap<String, CheckedExpression>),
    Unary {
        operator: FeaUnaryOperator,
        operand: Box<CheckedExpression>,
    },
    Binary {
        left: Box<CheckedExpression>,
        operator: FeaBinaryOperator,
        right: Box<CheckedExpression>,
    },
    OperationCall {
        operation: OperationIdentity,
        overload: OperationOverloadIdentity,
        lowering: OperationLoweringId,
        signature: OperationSignature,
        arguments: Vec<CheckedCallArgument>,
    },
    FunctionCall {
        function: Vec<String>,
        signature: CheckedCallableSignature,
        arguments: Vec<CheckedCallArgument>,
    },
    Quantity {
        value: Box<CheckedExpression>,
        unit: runmat_types::physical::UnitDefinition,
        /// The unit exactly as authored, for diagnostics that name what was found.
        source_unit_symbol: String,
    },
    Missing,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckedCallableSignature {
    Function(FunctionSignature),
    Declaration(DeclarationSignature),
}

pub struct ExpressionCheck<'a, C: AnalysisOperationCatalog + ?Sized> {
    pub path: &'a LogicalSourcePath,
    pub source: &'a str,
    pub scope: &'a ExpressionScope,
    pub catalog: &'a C,
}

pub fn check_expression<C: AnalysisOperationCatalog + ?Sized>(
    check: ExpressionCheck<'_, C>,
    expression: &FeaExpression,
) -> (CheckedExpression, Vec<FeaDiagnostic>) {
    let mut checker = ExpressionChecker {
        check,
        diagnostics: Vec::new(),
    };
    let expression = checker.check(expression);
    (expression, checker.diagnostics)
}

struct ExpressionChecker<'a, C: AnalysisOperationCatalog + ?Sized> {
    check: ExpressionCheck<'a, C>,
    diagnostics: Vec<FeaDiagnostic>,
}

impl<C: AnalysisOperationCatalog + ?Sized> ExpressionChecker<'_, C> {
    fn check(&mut self, expression: &FeaExpression) -> CheckedExpression {
        let (kind, value_type) = match &expression.kind {
            FeaExpressionKind::Number(value) => (
                CheckedExpressionKind::Number(value.clone()),
                if value.contains(['.', 'e', 'E']) {
                    FeaType::Real
                } else {
                    FeaType::Integer
                },
            ),
            FeaExpressionKind::String(value) => (
                CheckedExpressionKind::String(value.clone()),
                FeaType::String,
            ),
            FeaExpressionKind::Reference(reference) => self.check_reference(reference),
            FeaExpressionKind::List(values) => self.check_list(values, expression.span),
            FeaExpressionKind::Record(entries) => {
                let mut checked = BTreeMap::new();
                let mut types = BTreeMap::new();
                for entry in entries {
                    let value = self.check(&entry.value);
                    types.insert(entry.name.text.clone(), value.value_type.clone());
                    if checked.insert(entry.name.text.clone(), value).is_some() {
                        self.error(
                            "fea.semantic.duplicate_record_member",
                            format!("duplicate record member `{}`", entry.name.text),
                            entry.span,
                        );
                    }
                }
                (
                    CheckedExpressionKind::Record(checked),
                    FeaType::Record(types),
                )
            }
            FeaExpressionKind::Group(value) => {
                let checked = self.check(value);
                return CheckedExpression {
                    kind: checked.kind,
                    value_type: checked.value_type,
                    span: expression.span,
                };
            }
            FeaExpressionKind::Unary { operator, operand } => {
                let operand = self.check(operand);
                let value_type = if operand.value_type.is_numeric() {
                    operand.value_type.clone()
                } else {
                    self.error(
                        "fea.semantic.invalid_unary_operand",
                        "unary sign requires a numeric operand",
                        operand.span,
                    );
                    FeaType::Error
                };
                (
                    CheckedExpressionKind::Unary {
                        operator: *operator,
                        operand: Box::new(operand),
                    },
                    value_type,
                )
            }
            FeaExpressionKind::Binary {
                left,
                operator,
                right,
            } => {
                let left = self.check(left);
                let right = self.check(right);
                let value_type = self.binary_type(&left, *operator, &right, expression.span);
                (
                    CheckedExpressionKind::Binary {
                        left: Box::new(left),
                        operator: *operator,
                        right: Box::new(right),
                    },
                    value_type,
                )
            }
            FeaExpressionKind::Call { callee, arguments } => {
                return self.check_call(callee, arguments, expression.span);
            }
            FeaExpressionKind::PrefixApplication { callee, argument } => {
                let arguments = [FeaCallArgument::Positional((**argument).clone())];
                return self.check_call(callee, &arguments, expression.span);
            }
            FeaExpressionKind::Quantity { value, unit } => {
                let value = self.check(value);
                let unit_source = &self.check.source[unit.span.start..unit.span.end];
                match parse_unit_expression(unit_source) {
                    Ok(unit) => {
                        let value_type = match quantity_type(
                            &value.value_type,
                            runmat_types::physical::DynamicDimension::from_unit(unit),
                        ) {
                            Some(value_type) => value_type,
                            None => {
                                self.error(
                                    "fea.semantic.invalid_quantity_value",
                                    "a quantity requires a numeric scalar or regular numeric array",
                                    value.span,
                                );
                                FeaType::Error
                            }
                        };
                        (
                            CheckedExpressionKind::Quantity {
                                value: Box::new(value),
                                unit,
                                source_unit_symbol: unit_source.trim().to_owned(),
                            },
                            value_type,
                        )
                    }
                    Err(error) => {
                        self.error(
                            "fea.semantic.invalid_unit",
                            super::pending::with_pending(
                                error.to_string(),
                                super::pending::unit(unit_source),
                            ),
                            unit.span,
                        );
                        (CheckedExpressionKind::Missing, FeaType::Error)
                    }
                }
            }
            FeaExpressionKind::Missing => (CheckedExpressionKind::Missing, FeaType::Error),
        };
        let mut checked = CheckedExpression {
            kind,
            value_type,
            span: expression.span,
        };
        match checked.quantity_kind_hint() {
            Ok(Some(kind)) => {
                if let Some(value_type) = checked.value_type.with_quantity_kind(kind) {
                    checked.value_type = value_type;
                }
            }
            Err(error) => {
                self.error(
                    "fea.semantic.invalid_affine_quantity",
                    error.to_string(),
                    expression.span,
                );
                checked.value_type = FeaType::Error;
            }
            _ => {}
        }
        checked
    }

    fn check_reference(&mut self, reference: &FeaReference) -> (CheckedExpressionKind, FeaType) {
        let path = reference_path(reference);
        if path.len() == 1 {
            match path[0].as_str() {
                "true" => return (CheckedExpressionKind::Boolean(true), FeaType::Boolean),
                "false" => return (CheckedExpressionKind::Boolean(false), FeaType::Boolean),
                _ => {}
            }
        }
        if let Some(value_type) = self.check.scope.reference(reference) {
            return (CheckedExpressionKind::Reference(path), value_type.clone());
        }
        if let Some(value_type) = contextual_reference_type(&path) {
            return (CheckedExpressionKind::Reference(path), value_type);
        }
        if path.len() == 1 {
            return (CheckedExpressionKind::Enum(path[0].clone()), FeaType::Enum);
        }
        self.error(
            "fea.semantic.unresolved_reference",
            super::pending::with_pending(
                format!("unresolved reference `{}`", path.join(".")),
                super::pending::reference(&path),
            ),
            reference.span,
        );
        (CheckedExpressionKind::Reference(path), FeaType::Error)
    }

    fn check_list(
        &mut self,
        values: &[FeaExpression],
        span: Span,
    ) -> (CheckedExpressionKind, FeaType) {
        let checked = values
            .iter()
            .map(|value| self.check(value))
            .collect::<Vec<_>>();
        let element = checked
            .iter()
            .find(|value| !is_untyped_zero(value))
            .or_else(|| checked.first())
            .map_or(FeaType::Unknown, |value| value.value_type.clone());
        if checked
            .iter()
            .any(|value| !is_untyped_zero(value) && !types_compatible(&element, &value.value_type))
        {
            self.error(
                "fea.semantic.heterogeneous_list",
                "list elements must have compatible types and physical dimensions",
                span,
            );
            return (CheckedExpressionKind::List(checked), FeaType::Error);
        }
        let length = checked.len();
        (
            CheckedExpressionKind::List(checked),
            FeaType::Vector {
                element: Box::new(element),
                length: Some(length),
            },
        )
    }

    fn check_call(
        &mut self,
        callee: &FeaExpression,
        arguments: &[FeaCallArgument],
        span: Span,
    ) -> CheckedExpression {
        let FeaExpressionKind::Reference(reference) = &callee.kind else {
            self.error(
                "fea.semantic.invalid_callee",
                "FEA calls require a named operation, imported function, or declaration",
                callee.span,
            );
            return CheckedExpression {
                kind: CheckedExpressionKind::Missing,
                value_type: FeaType::Error,
                span,
            };
        };
        let checked_arguments = arguments
            .iter()
            .map(|argument| match argument {
                FeaCallArgument::Positional(value) => {
                    CheckedCallArgument::Positional(self.check(value))
                }
                FeaCallArgument::Named { name, value, .. } => CheckedCallArgument::Named {
                    name: name.text.clone(),
                    name_span: name.span,
                    value: self.check(value),
                },
            })
            .collect::<Vec<_>>();
        if let Some(signature) = self.check.scope.function(reference) {
            let signature = signature.clone();
            if !arguments_match(&signature.inputs, &checked_arguments) {
                self.error(
                    "fea.semantic.function_arguments",
                    "arguments do not match the imported RunMat function signature",
                    span,
                );
            }
            return CheckedExpression {
                kind: CheckedExpressionKind::FunctionCall {
                    function: reference_path(reference),
                    signature: CheckedCallableSignature::Function(signature.clone()),
                    arguments: checked_arguments,
                },
                value_type: outputs_type(&signature.outputs),
                span,
            };
        }
        if let Some(signature) = self.check.scope.declaration(reference) {
            let signature = signature.clone();
            if !declaration_arguments_match(&signature, &checked_arguments) {
                self.error(
                    "fea.semantic.declaration_arguments",
                    "declaration arguments are missing, unknown, duplicated, or have incompatible types",
                    span,
                );
            }
            return CheckedExpression {
                kind: CheckedExpressionKind::FunctionCall {
                    function: reference_path(reference),
                    signature: CheckedCallableSignature::Declaration(signature.clone()),
                    arguments: checked_arguments,
                },
                value_type: signature.result,
                span,
            };
        }
        if reference.segments.len() != 1 {
            self.error(
                "fea.semantic.unknown_callable",
                format!("`{}` is not callable", reference_path(reference).join(".")),
                span,
            );
            return CheckedExpression {
                kind: CheckedExpressionKind::Missing,
                value_type: FeaType::Error,
                span,
            };
        }
        let identity =
            match runmat_types::OperationIdentity::new(reference.segments[0].text.clone()) {
                Ok(identity) => identity,
                Err(_) => unreachable!("parser identifiers are non-empty"),
            };
        let operation = match self.check.catalog.resolve(&OperationResolutionRequest {
            operation: identity,
        }) {
            Ok(operation) => operation.operation,
            Err(error) => {
                let pending = super::pending::operation(&reference.segments[0].text);
                self.error(
                    error.code,
                    super::pending::with_pending(error.message, pending),
                    span,
                );
                return CheckedExpression {
                    kind: CheckedExpressionKind::Missing,
                    value_type: FeaType::Error,
                    span,
                };
            }
        };
        let matches = operation
            .overloads
            .iter()
            .filter(|overload| overload.frontends.contains(&OperationFrontend::Fea))
            .filter(|overload| arguments_match(&overload.signature.inputs, &checked_arguments))
            .collect::<Vec<_>>();
        let overload = match matches.as_slice() {
            [overload] => *overload,
            [] => {
                // `mean(x, over = time)`: a time mean over a transient step is planned.
                let time_mean = operation.identity.as_str() == "mean"
                    && checked_arguments.iter().any(|argument| {
                        matches!(argument, CheckedCallArgument::Named { name, .. } if name == "over")
                    });
                self.error(
                    "fea.semantic.no_matching_overload",
                    super::pending::with_pending(
                        format!(
                            "no FEA overload of `{}` accepts these arguments",
                            operation.identity
                        ),
                        time_mean.then_some(super::pending::TIME_MEAN),
                    ),
                    span,
                );
                return CheckedExpression {
                    kind: CheckedExpressionKind::Missing,
                    value_type: FeaType::Error,
                    span,
                };
            }
            _ => {
                self.error(
                    "fea.semantic.ambiguous_overload",
                    format!(
                        "multiple FEA overloads of `{}` accept these arguments",
                        operation.identity
                    ),
                    span,
                );
                return CheckedExpression {
                    kind: CheckedExpressionKind::Missing,
                    value_type: FeaType::Error,
                    span,
                };
            }
        };
        let value_type = operation_output_type(
            overload,
            &checked_arguments,
            &mut self.diagnostics,
            self.check.path,
            span,
        );
        CheckedExpression {
            kind: CheckedExpressionKind::OperationCall {
                operation: operation.identity.clone(),
                overload: overload.identity.clone(),
                lowering: overload.lowering.clone(),
                signature: overload.signature.clone(),
                arguments: checked_arguments,
            },
            value_type,
            span,
        }
    }

    fn binary_type(
        &mut self,
        left: &CheckedExpression,
        operator: FeaBinaryOperator,
        right: &CheckedExpression,
        span: Span,
    ) -> FeaType {
        if matches!(
            operator,
            FeaBinaryOperator::Add | FeaBinaryOperator::Subtract
        ) && [left, right].iter().any(|value| {
            matches!(
                value.quantity_kind_hint(),
                Ok(Some(
                    runmat_types::physical::QuantityKind::Temperature
                        | runmat_types::physical::QuantityKind::TemperatureDifference
                ))
            )
        }) {
            if let (Some(a), Some(b)) = (
                left.value_type.quantity_descriptor(),
                right.value_type.quantity_descriptor(),
            ) {
                if a.dimension() == b.dimension() && a.angle_power() == b.angle_power() {
                    let expression = CheckedExpression {
                        kind: CheckedExpressionKind::Binary {
                            left: Box::new(left.clone()),
                            operator,
                            right: Box::new(right.clone()),
                        },
                        value_type: FeaType::Quantity(a.without_semantic_kind()),
                        span,
                    };
                    return match expression.quantity_kind_hint() {
                        Ok(kind) => FeaType::Quantity(
                            a.with_semantic_kind(kind).expect("checked affine result"),
                        ),
                        Err(error) => {
                            self.error(
                                "fea.semantic.invalid_affine_quantity",
                                error.to_string(),
                                span,
                            );
                            FeaType::Error
                        }
                    };
                }
            }
        }
        match operator {
            FeaBinaryOperator::Equal
            | FeaBinaryOperator::NotEqual
            | FeaBinaryOperator::Less
            | FeaBinaryOperator::LessEqual
            | FeaBinaryOperator::Greater
            | FeaBinaryOperator::GreaterEqual => {
                if !types_compatible(&left.value_type, &right.value_type) {
                    self.error(
                        "fea.semantic.incompatible_comparison",
                        "comparison operands have incompatible types or physical dimensions",
                        span,
                    );
                }
                FeaType::Boolean
            }
            FeaBinaryOperator::Add | FeaBinaryOperator::Subtract => {
                if types_compatible(&left.value_type, &right.value_type) {
                    numeric_join(&left.value_type, &right.value_type)
                } else {
                    self.error(
                        "fea.semantic.incompatible_addition",
                        "addition and subtraction require compatible numeric types and dimensions",
                        span,
                    );
                    FeaType::Error
                }
            }
            FeaBinaryOperator::Multiply => multiply_type(&left.value_type, &right.value_type)
                .unwrap_or_else(|| {
                    self.error(
                        "fea.semantic.invalid_multiplication",
                        "multiplication requires numeric operands",
                        span,
                    );
                    FeaType::Error
                }),
            FeaBinaryOperator::Divide => divide_type(&left.value_type, &right.value_type)
                .unwrap_or_else(|| {
                    self.error(
                        "fea.semantic.invalid_division",
                        "division requires numeric operands",
                        span,
                    );
                    FeaType::Error
                }),
            FeaBinaryOperator::Power => {
                if !right.value_type.is_numeric() {
                    self.error(
                        "fea.semantic.invalid_power",
                        "power exponent must be numeric",
                        span,
                    );
                    FeaType::Error
                } else if matches!(
                    left.value_type,
                    FeaType::Integer | FeaType::Real | FeaType::Complex
                ) || left
                    .value_type
                    .is_quantity(PhysicalDimension::DIMENSIONLESS.into())
                {
                    left.value_type.clone()
                } else {
                    self.error(
                        "fea.semantic.dimensioned_power",
                        "dimensioned powers require a compile-time integer exponent",
                        span,
                    );
                    FeaType::Error
                }
            }
            FeaBinaryOperator::Range => {
                if !types_compatible(&left.value_type, &right.value_type) {
                    self.error(
                        "fea.semantic.incompatible_range",
                        "range bounds require compatible numeric types and dimensions",
                        span,
                    );
                    FeaType::Error
                } else {
                    FeaType::List(Box::new(numeric_join(&left.value_type, &right.value_type)))
                }
            }
        }
    }

    fn error(&mut self, code: impl Into<String>, message: impl Into<String>, span: Span) {
        self.diagnostics.push(FeaDiagnostic::error(
            code,
            message,
            FeaDiagnosticPhase::Semantic,
            Some(FeaDiagnosticLocation {
                path: self.check.path.clone(),
                span,
            }),
        ));
    }
}
