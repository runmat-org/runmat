use lsp_types::{Hover, HoverContents, MarkupContent, MarkupKind, Position};
use runmat_analysis_language::{CheckedFeaStatement, CheckedFeaStatementKind, FeaType};
use runmat_lexer::fea::{FeaKeyword, FeaToken, FeaTokenKind};
use runmat_parser::fea::{
    FeaBlock, FeaExpression, FeaExpressionKind, FeaModuleItem, FeaStatement, FeaStatementKind,
    FeaSyntaxModule,
};
use runmat_types::Span;

use super::{span_range, symbol_kind_name, token_at, FeaDocumentAnalysis};

impl FeaDocumentAnalysis {
    pub fn hover(&self, source: &str, position: &Position) -> Option<Hover> {
        let syntax = self.syntax()?;
        let token = token_at(syntax, source, position)?;
        let contents = match token.kind {
            FeaTokenKind::Keyword(keyword) => keyword_markdown(keyword),
            FeaTokenKind::Identifier => self.identifier_hover(syntax, source, position, token),
            FeaTokenKind::Number => self.literal_hover(syntax, token, "Numeric literal"),
            FeaTokenKind::String => self.literal_hover(syntax, token, "String or path literal"),
            _ => return None,
        };
        Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: contents,
            }),
            range: Some(span_range(source, token.span)),
        })
    }

    fn identifier_hover(
        &self,
        syntax: &FeaSyntaxModule,
        source: &str,
        position: &Position,
        token: FeaToken,
    ) -> String {
        if let Some((module, symbol)) = self.target_at(source, position) {
            let origin = if module == self.root {
                "Declared in this module.".to_string()
            } else {
                format!("Defined in `{}`.", module.as_str())
            };
            return format!(
                "**{}.{}**\n\nTyped {} declaration. {origin}",
                symbol_kind_name(symbol.kind),
                symbol.name,
                symbol_kind_name(symbol.kind),
            );
        }

        let lexeme = syntax.tokens.lexeme(token);
        if let Some(role) = declared_name_role(syntax, token.span) {
            return format!("**`{lexeme}`**\n\n{role}");
        }
        if let Some(quantity) = quantity_at(syntax, token.span.start) {
            return super::hover_docs::quantity_markdown(
                "Unit expression",
                &syntax.source()[quantity.start..quantity.end],
                self.assignment_value_type_at(token.span.start),
            );
        }
        if let Some(documentation) = super::hover_docs::authoring_doc(lexeme) {
            let value_type = self.assignment_value_type_at(token.span.start);
            return super::hover_docs::property_markdown(lexeme, value_type, documentation);
        }
        if assignment_name_at(syntax, token.span) {
            let value_type = self.assignment_value_type_at(token.span.start);
            return super::hover_docs::property_markdown(
                lexeme,
                value_type,
                super::hover_docs::fallback_authoring_doc(),
            );
        }
        if let Some(context) = containing_statement(syntax, token.span.start) {
            return format!("**`{lexeme}`**\n\n{context}");
        }
        format!(
            "**`{lexeme}`**\n\nFEA identifier. Names are resolved and type-checked within their declaration scope."
        )
    }

    fn assignment_value_type_at(&self, offset: usize) -> Option<&FeaType> {
        self.program
            .declarations
            .iter()
            .filter(|((module, _), declaration)| {
                module == &self.root
                    && declaration.span.start <= offset
                    && offset < declaration.span.end
            })
            .filter_map(|(_, declaration)| checked_statement_at(&declaration.statements, offset))
            .min_by_key(|statement| statement.span.end.saturating_sub(statement.span.start))
            .and_then(statement_value_type)
    }

    fn literal_hover(&self, syntax: &FeaSyntaxModule, token: FeaToken, label: &str) -> String {
        if let Some(quantity) = quantity_at(syntax, token.span.start) {
            return super::hover_docs::quantity_markdown(
                "Physical quantity",
                &syntax.source()[quantity.start..quantity.end],
                self.assignment_value_type_at(token.span.start),
            );
        }
        format!("**{label} `{}`**", syntax.tokens.lexeme(token))
    }
}

fn checked_statement_at(
    statements: &[CheckedFeaStatement],
    offset: usize,
) -> Option<&CheckedFeaStatement> {
    statements
        .iter()
        .filter(|statement| statement.span.start <= offset && offset < statement.span.end)
        .flat_map(|statement| {
            let nested = match &statement.kind {
                CheckedFeaStatementKind::GeneratedByFunction { statements, .. }
                | CheckedFeaStatementKind::SignalDefinition { statements, .. }
                | CheckedFeaStatementKind::ModelDefinition { statements, .. }
                | CheckedFeaStatementKind::Block { statements, .. }
                | CheckedFeaStatementKind::OutputDefinition { statements, .. } => {
                    checked_statement_at(statements, offset)
                }
                _ => None,
            };
            std::iter::once(statement).chain(nested)
        })
        .min_by_key(|statement| statement.span.end.saturating_sub(statement.span.start))
}

fn statement_value_type(statement: &CheckedFeaStatement) -> Option<&FeaType> {
    match &statement.kind {
        CheckedFeaStatementKind::Assignment { value, .. }
        | CheckedFeaStatementKind::Let { value, .. }
        | CheckedFeaStatementKind::Feature { value, .. }
        | CheckedFeaStatementKind::SourceParameterUpdate { value, .. }
        | CheckedFeaStatementKind::Requirement { value, .. }
        | CheckedFeaStatementKind::Output { value, .. } => Some(&value.value_type),
        _ => None,
    }
}

fn assignment_name_at(syntax: &FeaSyntaxModule, span: Span) -> bool {
    syntax.items.iter().any(|item| {
        let FeaModuleItem::Declaration(declaration) = item else {
            return false;
        };
        block_has_assignment_name(&declaration.body, span)
    })
}

fn block_has_assignment_name(block: &FeaBlock, span: Span) -> bool {
    block.statements.iter().any(|statement| {
        if matches!(
            &statement.kind,
            FeaStatementKind::Assignment { target, .. }
                if target.segments.iter().any(|segment| segment.span == span)
        ) {
            return true;
        }
        match &statement.kind {
            FeaStatementKind::GeneratedByFunction { body, .. }
            | FeaStatementKind::SignalDefinition { body, .. }
            | FeaStatementKind::ModelDefinition { body, .. }
            | FeaStatementKind::Block { body, .. }
            | FeaStatementKind::OutputDefinition { body, .. } => {
                block_has_assignment_name(body, span)
            }
            _ => false,
        }
    })
}

fn declared_name_role(syntax: &FeaSyntaxModule, span: Span) -> Option<String> {
    for item in &syntax.items {
        match item {
            FeaModuleItem::Import(import) if import.name.span == span => {
                return Some(format!(
                    "Imported {} symbol from `{}`.",
                    super::import_kind_name(import.kind),
                    import.source
                ));
            }
            FeaModuleItem::Declaration(declaration) if declaration.name.span == span => {
                return Some(format!(
                    "Top-level {} declaration.",
                    super::declaration_kind_name(declaration.kind)
                ));
            }
            FeaModuleItem::Declaration(declaration) => {
                if let Some(role) = block_declared_name_role(&declaration.body, span) {
                    return Some(role);
                }
            }
            FeaModuleItem::Import(_) | FeaModuleItem::Error(_) => {}
        }
    }
    None
}

fn block_declared_name_role(block: &FeaBlock, span: Span) -> Option<String> {
    for statement in &block.statements {
        let (name, role, nested) = match &statement.kind {
            FeaStatementKind::SignalDefinition { name, body, .. } => {
                (Some(name), Some("Typed signal declaration."), Some(body))
            }
            FeaStatementKind::ModelDefinition { kind, name, body } => (
                Some(name),
                Some(match kind {
                    runmat_parser::fea::FeaModelDefinitionKind::Section => "Section declaration.",
                    runmat_parser::fea::FeaModelDefinitionKind::Layup => {
                        "Composite layup declaration."
                    }
                    runmat_parser::fea::FeaModelDefinitionKind::Orientation => {
                        "Material orientation declaration."
                    }
                    runmat_parser::fea::FeaModelDefinitionKind::Constraint => {
                        "Constraint declaration."
                    }
                    runmat_parser::fea::FeaModelDefinitionKind::Load => "Load declaration.",
                    runmat_parser::fea::FeaModelDefinitionKind::LoadCase => {
                        "Load-case declaration."
                    }
                    runmat_parser::fea::FeaModelDefinitionKind::Interface => {
                        "Interface declaration."
                    }
                    runmat_parser::fea::FeaModelDefinitionKind::Connector => {
                        "Connector declaration."
                    }
                    runmat_parser::fea::FeaModelDefinitionKind::Excitation => {
                        "Excitation declaration."
                    }
                    runmat_parser::fea::FeaModelDefinitionKind::ResponseSpectrum => {
                        "Response-spectrum declaration."
                    }
                    runmat_parser::fea::FeaModelDefinitionKind::RandomExcitation => {
                        "Random-excitation declaration."
                    }
                    runmat_parser::fea::FeaModelDefinitionKind::Initial => {
                        "Initial-condition declaration."
                    }
                }),
                Some(body),
            ),
            FeaStatementKind::Input { name, .. } => {
                (Some(name), Some("Typed declaration input."), None)
            }
            FeaStatementKind::Let { name, .. } => {
                (Some(name), Some("Local immutable value."), None)
            }
            FeaStatementKind::Feature { name, .. } => {
                (Some(name), Some("Parametric geometry feature."), None)
            }
            FeaStatementKind::Output { name, .. }
            | FeaStatementKind::OutputDefinition { name, .. } => {
                (Some(name), Some("Named study output."), None)
            }
            FeaStatementKind::GeneratedByFunction { body, .. }
            | FeaStatementKind::Block { body, .. } => (None, None, Some(body)),
            _ => (None, None, None),
        };
        if name.is_some_and(|name| name.span == span) {
            return role.map(str::to_string);
        }
        if let Some(nested) = nested {
            if let Some(role) = block_declared_name_role(nested, span) {
                return Some(role);
            }
        }
    }
    None
}

fn containing_statement(syntax: &FeaSyntaxModule, offset: usize) -> Option<&'static str> {
    syntax.items.iter().find_map(|item| {
        let FeaModuleItem::Declaration(declaration) = item else {
            return None;
        };
        statement_context(&declaration.body, offset)
    })
}

fn statement_context(block: &FeaBlock, offset: usize) -> Option<&'static str> {
    block
        .statements
        .iter()
        .filter(|statement| statement.span.start <= offset && offset < statement.span.end)
        .min_by_key(|statement| statement.span.end.saturating_sub(statement.span.start))
        .map(statement_description)
}

fn statement_description(statement: &FeaStatement) -> &'static str {
    match &statement.kind {
        FeaStatementKind::GeneratedByFunction { .. } => "Generated-signal configuration. The referenced `.m` function is checked against the signal contract.",
        FeaStatementKind::UniformSample { .. } => "Signal sampling configuration with typed start, end, and sample-rate quantities.",
        FeaStatementKind::AppliedLoad { .. } => "Applied load. The target region and load value are checked against the selected physics.",
        FeaStatementKind::SignalDefinition { .. } => "Signal definition with an explicit physical quantity and independent domain.",
        FeaStatementKind::ModelDefinition { .. } => "Typed model resource definition.",
        FeaStatementKind::Input { .. } => "Typed declaration input.",
        FeaStatementKind::Let { .. } => "Local immutable value.",
        FeaStatementKind::Feature { .. } => "Parametric geometry feature.",
        FeaStatementKind::SourceParameterUpdate { .. } => "Update to a parameter exposed by source geometry.",
        FeaStatementKind::Use(_) => "Reference to a reusable declaration.",
        FeaStatementKind::Requirement { .. } => "Required bound or capability for this study.",
        FeaStatementKind::Assignment { .. } => "Typed property assignment.",
        FeaStatementKind::PropertyAssignment { .. } => "Assigns a material or property to a geometry region.",
        FeaStatementKind::Output { .. } | FeaStatementKind::OutputDefinition { .. } => "Named, typed study output.",
        FeaStatementKind::Include(_) => "Includes a declared resource in this enclosing definition.",
        FeaStatementKind::Retain(_) => "Retains an intermediate field or artifact for later use.",
        FeaStatementKind::Block { .. } => "Typed configuration block.",
        FeaStatementKind::Command { .. } => "FEA language command.",
    }
}

fn quantity_at(syntax: &FeaSyntaxModule, offset: usize) -> Option<Span> {
    syntax.items.iter().find_map(|item| {
        let FeaModuleItem::Declaration(declaration) = item else {
            return None;
        };
        block_quantity_at(&declaration.body, offset)
    })
}

fn block_quantity_at(block: &FeaBlock, offset: usize) -> Option<Span> {
    block.statements.iter().find_map(|statement| {
        statement_expressions(statement)
            .into_iter()
            .find_map(|expression| expression_quantity_at(expression, offset))
            .or_else(|| match &statement.kind {
                FeaStatementKind::GeneratedByFunction { body, .. }
                | FeaStatementKind::SignalDefinition { body, .. }
                | FeaStatementKind::ModelDefinition { body, .. }
                | FeaStatementKind::Block { body, .. }
                | FeaStatementKind::OutputDefinition { body, .. } => {
                    block_quantity_at(body, offset)
                }
                _ => None,
            })
    })
}

fn statement_expressions(statement: &FeaStatement) -> Vec<&FeaExpression> {
    match &statement.kind {
        FeaStatementKind::UniformSample { start, end, rate } => vec![start, end, rate],
        FeaStatementKind::AppliedLoad { target, value, .. } => vec![target, value],
        FeaStatementKind::Input { default, .. } => default.iter().collect(),
        FeaStatementKind::Let { value, .. }
        | FeaStatementKind::Feature { value, .. }
        | FeaStatementKind::SourceParameterUpdate { value, .. }
        | FeaStatementKind::Requirement { value, .. }
        | FeaStatementKind::Output { value, .. }
        | FeaStatementKind::Include(value)
        | FeaStatementKind::Retain(value) => vec![value],
        FeaStatementKind::Assignment { value, .. } => vec![value],
        FeaStatementKind::PropertyAssignment { property, target } => vec![property, target],
        _ => Vec::new(),
    }
}

fn expression_quantity_at(expression: &FeaExpression, offset: usize) -> Option<Span> {
    match &expression.kind {
        FeaExpressionKind::Quantity { .. }
            if expression.span.start <= offset && offset < expression.span.end =>
        {
            Some(expression.span)
        }
        FeaExpressionKind::Quantity { value, .. }
        | FeaExpressionKind::Group(value)
        | FeaExpressionKind::Unary { operand: value, .. }
        | FeaExpressionKind::PrefixApplication {
            argument: value, ..
        } => expression_quantity_at(value, offset),
        FeaExpressionKind::Binary { left, right, .. } => {
            expression_quantity_at(left, offset).or_else(|| expression_quantity_at(right, offset))
        }
        FeaExpressionKind::Call { callee, arguments } => expression_quantity_at(callee, offset)
            .or_else(|| {
                arguments.iter().find_map(|argument| match argument {
                    runmat_parser::fea::FeaCallArgument::Positional(value)
                    | runmat_parser::fea::FeaCallArgument::Named { value, .. } => {
                        expression_quantity_at(value, offset)
                    }
                })
            }),
        FeaExpressionKind::List(values) => values
            .iter()
            .find_map(|value| expression_quantity_at(value, offset)),
        FeaExpressionKind::Record(entries) => entries
            .iter()
            .find_map(|entry| expression_quantity_at(&entry.value, offset)),
        _ => None,
    }
}

fn keyword_markdown(keyword: FeaKeyword) -> String {
    let (syntax, description) = match keyword {
        FeaKeyword::Import => ("import", "Imports a typed declaration from another `.fea` or `.m` module."),
        FeaKeyword::From => ("from", "Names the source file for an import."),
        FeaKeyword::Over => ("over", "Declares the independent domain of a signal, such as time or frequency."),
        FeaKeyword::Geometry => ("geometry", "Declares CAD loaded from a file or constructed parametrically, together with named regions."),
        FeaKeyword::Mesh => ("mesh", "Declares a mesh derived from a geometry declaration."),
        FeaKeyword::Material => ("material", "Declares typed physical properties. Materials affect a model only through an explicit region assignment."),
        FeaKeyword::Model => ("model", "Collects assignments, constraints, loads, signals, interfaces, and other physical model resources."),
        FeaKeyword::Study => ("study", "Declares an executable analysis by binding geometry, mesh, model, procedures, outputs, and optional execution policy."),
        FeaKeyword::Sweep => ("sweep", "Declares a typed parameter study over one or more axes."),
        FeaKeyword::Execution => ("execution", "Declares optional local, GPU, parallel, or distributed placement policy without changing the physical model."),
        FeaKeyword::Function => ("function", "Identifies a RunMat `.m` function imported into the study."),
        FeaKeyword::Region => ("region", "Names a typed set of geometry entities selected from CAD topology."),
        FeaKeyword::Input => ("input", "Declares a typed configurable input, optionally with a default or bounds."),
        FeaKeyword::Let => ("let", "Binds an immutable local value."),
        FeaKeyword::Construct => ("construct", "Starts a parametric geometry construction pipeline."),
        FeaKeyword::Feature => ("feature", "Creates a named parametric geometry operation whose result can feed later features."),
        FeaKeyword::Derive => ("derive", "Creates geometry derived from another declared geometry source."),
        FeaKeyword::Generated => ("generated", "Defines a signal by evaluating an imported RunMat function under an explicit sampling contract."),
        FeaKeyword::Assign => ("assign", "Assigns a material or physical property to a typed geometry region."),
        FeaKeyword::To => ("to", "Connects an assigned property or resource to its target."),
        FeaKeyword::Constraint => ("constraint", "Declares an essential, relational, contact, or other typed model constraint."),
        FeaKeyword::Load => ("load", "Declares a typed physical load on a region."),
        FeaKeyword::LoadCase => ("load_case", "Groups loads and excitations into a reusable analysis case."),
        FeaKeyword::Analysis => ("analysis", "Adds a typed solve procedure to the study sequence."),
        FeaKeyword::Output => ("output", "Declares a named result expression and optional output policy."),
        FeaKeyword::Retain => ("retain", "Keeps an intermediate field or artifact needed by later procedures or inspection."),
        FeaKeyword::Include => ("include", "Includes a previously declared load, excitation, or other reusable resource."),
        FeaKeyword::On => ("on", "Binds a declaration to its geometry, mesh, or region target."),
        FeaKeyword::Select => ("select", "Selects CAD topology by stable identifier, metadata, query, or a constructed result."),
        FeaKeyword::Expect => ("expect", "Asserts selector cardinality so topology drift fails explicitly."),
        FeaKeyword::Result => ("result", "Selects or publishes the result of a geometry construction pipeline."),
        FeaKeyword::Case => ("case", "Names one load case for an analysis procedure."),
        FeaKeyword::Cases => ("cases", "Names multiple load cases for an analysis procedure."),
        FeaKeyword::Axis => ("axis", "Declares one typed sweep dimension."),
        FeaKeyword::Transition => ("transition", "Transfers typed state between procedures in a coupled or sequential study."),
        FeaKeyword::Initial => ("initial", "Declares a typed initial condition or imported initial state."),
        FeaKeyword::Interface => ("interface", "Declares behavior across a pair of model regions."),
        FeaKeyword::Connector => ("connector", "Declares a discrete typed connection between topology targets."),
        FeaKeyword::Section => ("section", "Declares section properties for reduced-dimensional structural elements."),
        FeaKeyword::Layup => ("layup", "Declares an ordered composite ply layup."),
        FeaKeyword::Orientation => ("orientation", "Declares a local material or section coordinate system."),
        FeaKeyword::Signal => ("signal", "Declares a typed quantity varying over time, frequency, or another domain."),
        FeaKeyword::Table => ("table", "Declares a named quantity over a domain, such as an impedance over frequency from samples or a model."),
        FeaKeyword::Excitation => ("excitation", "Applies a signal-driven input to a typed model target."),
        FeaKeyword::ResponseSpectrum => ("response_spectrum", "Declares a response-spectrum input for modal combination."),
        FeaKeyword::RandomExcitation => ("random_excitation", "Declares spectral-density or cross-spectral input for random response."),
    };
    format!("**`{syntax}`**\n\n{description}")
}
