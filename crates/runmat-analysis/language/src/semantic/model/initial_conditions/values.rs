use super::super::{diagnostic, properties::*};
use crate::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement,
    CheckedFeaStatementKind, FeaDiagnostic, FeaType, PhysicalDimension,
};
use runmat_analysis_core::{
    CoordinateFrame, InitialFieldArtifactRef, InitialValue, PhysicalQuantityKind, PhysicalSignal,
    SignalDomainKind, SignalId, SignalSampleApproach, SignalShape,
};

pub(super) fn scalar_value<Q>(
    declaration: &CheckedFeaDeclaration,
    body: &[CheckedFeaStatement],
    signals: &[PhysicalSignal],
    contract: ValueContract,
    construct: impl FnOnce(f64) -> Q,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<InitialValue<Q>> {
    match value_source(declaration, body, diagnostics)? {
        ValueSource::Uniform(expression) => {
            if expression
                .validate_quantity_kind(contract.quantity)
                .is_err()
            {
                diagnostic(
                    declaration,
                    expression.span,
                    "initial values require the declared absolute or difference quantity",
                    diagnostics,
                );
                return None;
            }
            constant_scalar(
                declaration,
                expression,
                "uniform",
                runmat_types::physical::DynamicDimension::quantity_type(contract.quantity),
                diagnostics,
            )
            .map(construct)
            .map(InitialValue::Uniform)
        }
        ValueSource::Sample(body) => sample(
            declaration,
            body,
            signals,
            contract.quantity,
            SignalShape::Scalar,
            diagnostics,
        ),
        ValueSource::Artifact(body) => artifact(declaration, body, diagnostics),
    }
}

pub(super) fn vector_value<Q>(
    declaration: &CheckedFeaDeclaration,
    body: &[CheckedFeaStatement],
    signals: &[PhysicalSignal],
    contract: ValueContract,
    construct: impl Fn(f64) -> Q,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<InitialValue<Q>> {
    match value_source(declaration, body, diagnostics)? {
        ValueSource::Uniform(_) => required_vector(
            declaration,
            body,
            "uniform",
            runmat_types::physical::DynamicDimension::quantity_type(contract.quantity),
            diagnostics,
        )
        .map(|values| InitialValue::UniformVector3(values.map(construct))),
        ValueSource::Sample(body) => sample(
            declaration,
            body,
            signals,
            contract.quantity,
            SignalShape::Vector3,
            diagnostics,
        ),
        ValueSource::Artifact(body) => artifact(declaration, body, diagnostics),
    }
}

#[derive(Clone, Copy)]
pub(super) struct ValueContract {
    quantity: PhysicalQuantityKind,
}

impl ValueContract {
    pub(super) const fn new(quantity: PhysicalQuantityKind) -> Self {
        Self { quantity }
    }
}

enum ValueSource<'a> {
    Uniform(&'a CheckedExpression),
    Sample(&'a [CheckedFeaStatement]),
    Artifact(&'a [CheckedFeaStatement]),
}

fn value_source<'a>(
    declaration: &CheckedFeaDeclaration,
    body: &'a [CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<ValueSource<'a>> {
    let uniform = body.iter().find_map(|statement| match &statement.kind {
        CheckedFeaStatementKind::Assignment { target, value, .. }
            if target.as_slice() == ["uniform"] =>
        {
            Some(ValueSource::Uniform(value))
        }
        _ => None,
    });
    let blocks = body.iter().filter_map(|statement| match &statement.kind {
        CheckedFeaStatementKind::Block {
            header, statements, ..
        } if header.trim() == "sample" => Some(ValueSource::Sample(statements)),
        CheckedFeaStatementKind::Block {
            header, statements, ..
        } if header.trim() == "field_artifact" => Some(ValueSource::Artifact(statements)),
        _ => None,
    });
    let mut sources = uniform.into_iter().chain(blocks);
    let source = sources.next();
    if source.is_none() || sources.next().is_some() {
        diagnostic(
            declaration,
            declaration.span,
            "initial condition requires exactly one of `uniform`, `sample`, or `field_artifact`",
            diagnostics,
        );
        return None;
    }
    source
}

fn sample<Q>(
    declaration: &CheckedFeaDeclaration,
    body: &[CheckedFeaStatement],
    signals: &[PhysicalSignal],
    quantity: PhysicalQuantityKind,
    shape: SignalShape,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<InitialValue<Q>> {
    validate_assignments(
        declaration,
        body,
        &["signal", "at", "approach"],
        diagnostics,
    );
    let signal_expression = required_assignment(declaration, body, "signal", diagnostics)?;
    let id = signal_reference(declaration, signal_expression, diagnostics)?;
    let signal = signals.iter().find(|signal| signal.signal_id() == &id);
    let Some(signal) =
        signal.filter(|signal| signal.quantity() == quantity && signal.shape() == &shape)
    else {
        diagnostic(
            declaration,
            signal_expression.span,
            "initial-condition sample requires a declared signal with the exact quantity and shape",
            diagnostics,
        );
        return None;
    };
    let at_expression = required_assignment(declaration, body, "at", diagnostics)?;
    let abscissa = sample_abscissa(declaration, at_expression, signal.domain(), diagnostics)?;
    let approach_expression = required_assignment(declaration, body, "approach", diagnostics)?;
    let approach = match &approach_expression.kind {
        CheckedExpressionKind::Enum(name) if name == "at" => SignalSampleApproach::At,
        CheckedExpressionKind::Enum(name) if name == "left_limit" => {
            SignalSampleApproach::LeftLimit
        }
        CheckedExpressionKind::Enum(name) if name == "right_limit" => {
            SignalSampleApproach::RightLimit
        }
        _ => {
            diagnostic(
                declaration,
                approach_expression.span,
                "sample approach must be `at`, `left_limit`, or `right_limit`",
                diagnostics,
            );
            return None;
        }
    };
    Some(InitialValue::SignalSample {
        signal_id: id,
        abscissa,
        approach,
    })
}

fn sample_abscissa(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    domain: SignalDomainKind,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    let dimension = match domain {
        SignalDomainKind::Time => PhysicalDimension::TIME,
        SignalDomainKind::StudyParameter | SignalDomainKind::Constant => {
            PhysicalDimension::DIMENSIONLESS
        }
        SignalDomainKind::Frequency => {
            diagnostic(
                declaration,
                expression.span,
                "frequency-domain signals cannot define an initial condition",
                diagnostics,
            );
            return None;
        }
    };
    constant_scalar(declaration, expression, "at", dimension, diagnostics)
}

fn artifact<Q>(
    declaration: &CheckedFeaDeclaration,
    body: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<InitialValue<Q>> {
    validate_assignments(
        declaration,
        body,
        &[
            "artifact_id",
            "content_digest",
            "topology_digest",
            "field_id",
        ],
        diagnostics,
    );
    Some(InitialValue::FieldArtifact(InitialFieldArtifactRef {
        artifact_id: string_member(declaration, body, "artifact_id", diagnostics)?,
        content_digest: string_member(declaration, body, "content_digest", diagnostics)?,
        topology_digest: string_member(declaration, body, "topology_digest", diagnostics)?,
        field_id: string_member(declaration, body, "field_id", diagnostics)?,
    }))
}

fn constant_scalar(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    name: &str,
    dimension: impl Into<runmat_types::physical::DynamicDimension>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    let dimension = dimension.into();
    let compatible = if dimension == PhysicalDimension::DIMENSIONLESS.into() {
        matches!(expression.value_type, FeaType::Integer | FeaType::Real)
            || (expression.value_type).is_quantity(dimension)
    } else {
        (expression.value_type).is_quantity(dimension)
    };
    if compatible {
        scalar(expression)
    } else {
        diagnostic(
            declaration,
            expression.span,
            format!("initial-condition member `{name}` has the wrong physical dimension"),
            diagnostics,
        );
        None
    }
}

fn signal_reference(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<SignalId> {
    match &expression.kind {
        CheckedExpressionKind::Reference(path) if path.len() == 2 && path[0] == "signal" => {
            Some(SignalId(path[1].clone()))
        }
        _ => {
            diagnostic(
                declaration,
                expression.span,
                "sample signal must be `signal.<name>`",
                diagnostics,
            );
            None
        }
    }
}

fn string_member(
    declaration: &CheckedFeaDeclaration,
    body: &[CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<String> {
    let expression = required_assignment(declaration, body, name, diagnostics)?;
    match &expression.kind {
        CheckedExpressionKind::String(value) if !value.trim().is_empty() => Some(value.clone()),
        _ => {
            diagnostic(
                declaration,
                expression.span,
                format!("`{name}` requires a non-empty string"),
                diagnostics,
            );
            None
        }
    }
}

pub(super) fn parse_frame(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<CoordinateFrame> {
    match &expression.kind {
        CheckedExpressionKind::Enum(name) if name == "global" => Some(CoordinateFrame::Global),
        CheckedExpressionKind::Enum(name) if name == "material" => Some(CoordinateFrame::Material),
        CheckedExpressionKind::Enum(name) if name == "element" => Some(CoordinateFrame::Element),
        CheckedExpressionKind::Enum(name) if name == "boundary_normal_tangent" => {
            Some(CoordinateFrame::BoundaryNormalTangent)
        }
        _ => {
            diagnostic(
                declaration,
                expression.span,
                "frame requires a supported coordinate-frame enum",
                diagnostics,
            );
            None
        }
    }
}
