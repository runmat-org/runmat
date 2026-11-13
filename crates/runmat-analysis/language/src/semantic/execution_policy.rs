use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{
    AcceleratorPreference, AuthoredExecutionPolicy, BackendPreference, CheckpointPreference,
    CooperativePolicy, ExecutionResourceLimits, FanoutPolicy, InclusiveCountRange,
    OperatorPartitioner, ParallelismLimit, PartitionBalance, PlacementPreference, ResiliencePolicy,
};
use runmat_design_syntax::{FeaAssignmentRelation, FeaBinaryOperator};

use crate::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement,
    CheckedFeaStatementKind, ExecutionProfileSource, FeaDiagnostic, FeaDiagnosticLocation,
    FeaDiagnosticPhase, FeaSymbolKey, FeaSymbolKind, LogicalSourcePath, PhysicalDimension,
};

mod profile;

pub(super) fn lower_execution_policies<S: ExecutionProfileSource + ?Sized>(
    source: &S,
    declarations: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> BTreeMap<(LogicalSourcePath, FeaSymbolKey), AuthoredExecutionPolicy> {
    declarations
        .iter()
        .filter_map(|(key, declaration)| {
            (key.1.kind == FeaSymbolKind::Execution).then(|| {
                lower_policy(source, declaration, diagnostics).map(|policy| (key.clone(), policy))
            })?
        })
        .collect()
}

fn lower_policy<S: ExecutionProfileSource + ?Sized>(
    source: &S,
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredExecutionPolicy> {
    let initial_errors = diagnostics.len();
    let placement = top_assignment(declaration, "placement")
        .and_then(|value| placement(value, declaration, diagnostics))
        .unwrap_or(PlacementPreference::Auto);
    let backend = top_assignment(declaration, "backend")
        .and_then(|value| closed_enum(value, declaration, "backend", diagnostics))
        .and_then(|value| match value.as_str() {
            "auto" => Some(BackendPreference::Auto),
            "cpu" => Some(BackendPreference::Cpu),
            "accelerator" => Some(BackendPreference::Accelerator),
            _ => invalid(
                declaration,
                "backend must be auto, cpu, or accelerator",
                diagnostics,
            ),
        })
        .unwrap_or(BackendPreference::Auto);
    let resources = resource_limits(declaration, diagnostics);
    let fanout = fanout_policy(declaration, diagnostics);
    let cooperative = cooperative_policy(declaration, diagnostics);
    let resilience = resilience_policy(declaration, diagnostics);
    profile::validate_named_profile(
        source,
        declaration,
        &placement,
        backend,
        resources.accelerators,
        diagnostics,
    );
    validate_members(declaration, diagnostics);
    if diagnostics.len() != initial_errors {
        return None;
    }
    let policy = AuthoredExecutionPolicy {
        module_path: declaration.module.as_str().to_string(),
        policy_id: declaration.key.name.clone(),
        placement,
        backend,
        resources,
        fanout,
        cooperative,
        resilience,
    };
    if let Err(error) = policy.validate() {
        error_at(
            declaration,
            declaration.span,
            error.to_string(),
            diagnostics,
        );
        return None;
    }
    Some(policy)
}

fn placement(
    value: &CheckedExpression,
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<PlacementPreference> {
    match &value.kind {
        CheckedExpressionKind::Enum(value) if value == "auto" => Some(PlacementPreference::Auto),
        CheckedExpressionKind::Enum(value) if value == "local" => Some(PlacementPreference::Local),
        CheckedExpressionKind::Enum(value) if value == "cluster" => {
            Some(PlacementPreference::Cluster { profile: None })
        }
        CheckedExpressionKind::Reference(path) if matches!(path.as_slice(), [namespace, _] if namespace == "profile") => {
            Some(PlacementPreference::Cluster {
                profile: Some(path[1].clone()),
            })
        }
        _ => invalid(
            declaration,
            "placement must be auto, local, cluster, or profile.<name>",
            diagnostics,
        ),
    }
}

fn resource_limits(
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> ExecutionResourceLimits {
    let Some(statements) = block(declaration, "resources") else {
        return ExecutionResourceLimits::default();
    };
    let workers = assignment(statements, "workers").and_then(|(_, value)| {
        count_range(value)
            .or_else(|| invalid(declaration, "workers must be N or N..M", diagnostics))
    });
    let cpu_threads_per_worker =
        assignment(statements, "cpu_threads_per_worker").and_then(|(_, value)| {
            let range = count_range(value).or_else(|| {
                invalid(
                    declaration,
                    "cpu_threads_per_worker must be one positive whole number",
                    diagnostics,
                )
            })?;
            if range.minimum != range.maximum {
                return invalid(
                    declaration,
                    "cpu_threads_per_worker must be one positive whole number",
                    diagnostics,
                );
            }
            Some(range.minimum)
        });
    let maximum_memory_bytes_per_worker =
        assignment(statements, "memory_per_worker").and_then(|(relation, value)| {
            if relation != FeaAssignmentRelation::LessEqual
                || !(value.value_type).is_quantity((PhysicalDimension::INFORMATION).into())
            {
                return invalid(
                    declaration,
                    "memory_per_worker must use `<=` with an information quantity",
                    diagnostics,
                );
            }
            let bytes = super::procedure::canonical_scalar(value)?;
            finite_u64(bytes).or_else(|| {
                invalid(
                    declaration,
                    "memory_per_worker must be a positive whole-byte bound of at most 2^53 - 1 bytes",
                    diagnostics,
                )
            })
        });
    let accelerators = assignment(statements, "accelerators")
        .and_then(|(_, value)| closed_enum(value, declaration, "accelerators", diagnostics))
        .and_then(|value| match value.as_str() {
            "disabled" => Some(AcceleratorPreference::Disabled),
            "preferred" => Some(AcceleratorPreference::Preferred),
            "required" => Some(AcceleratorPreference::Required),
            _ => invalid(
                declaration,
                "accelerators must be disabled, preferred, or required",
                diagnostics,
            ),
        })
        .unwrap_or(AcceleratorPreference::Disabled);
    ExecutionResourceLimits {
        workers,
        cpu_threads_per_worker,
        maximum_memory_bytes_per_worker,
        accelerators,
    }
}

fn fanout_policy(
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> FanoutPolicy {
    let Some(statements) = block(declaration, "fanout") else {
        return FanoutPolicy::default();
    };
    FanoutPolicy {
        sweep_cases: parallelism(statements, "sweep_cases", declaration, diagnostics),
        frequency_samples: parallelism(statements, "frequency_samples", declaration, diagnostics),
    }
}

fn cooperative_policy(
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> CooperativePolicy {
    let Some(statements) = block(declaration, "cooperative") else {
        return CooperativePolicy::default();
    };
    let partitioner = assignment(statements, "partitioner")
        .and_then(|(_, value)| closed_enum(value, declaration, "partitioner", diagnostics))
        .and_then(|value| match value.as_str() {
            "auto" => Some(OperatorPartitioner::Auto),
            "graph" => Some(OperatorPartitioner::Graph),
            "geometric" => Some(OperatorPartitioner::Geometric),
            _ => invalid(
                declaration,
                "partitioner must be auto, graph, or geometric",
                diagnostics,
            ),
        })
        .unwrap_or(OperatorPartitioner::Auto);
    let balance = assignment(statements, "balance")
        .and_then(|(_, value)| closed_enum(value, declaration, "balance", diagnostics))
        .and_then(|value| match value.as_str() {
            "nonzeros" => Some(PartitionBalance::Nonzeros),
            "dofs" => Some(PartitionBalance::Dofs),
            "elements" => Some(PartitionBalance::Elements),
            _ => invalid(
                declaration,
                "balance must be nonzeros, dofs, or elements",
                diagnostics,
            ),
        })
        .unwrap_or(PartitionBalance::Nonzeros);
    CooperativePolicy {
        operator_partitions: parallelism(
            statements,
            "operator_partitions",
            declaration,
            diagnostics,
        ),
        partitioner,
        balance,
    }
}

fn resilience_policy(
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> ResiliencePolicy {
    let Some(statements) = block(declaration, "resilience") else {
        return ResiliencePolicy::default();
    };
    let retries = assignment(statements, "retries").map_or(0, |(_, value)| {
        positive_or_zero_u16(value).unwrap_or_else(|| {
            invalid::<()>(
                declaration,
                "retries must be a nonnegative integer",
                diagnostics,
            );
            0
        })
    });
    let checkpoint = assignment(statements, "checkpoint")
        .and_then(|(_, value)| closed_enum(value, declaration, "checkpoint", diagnostics))
        .and_then(|value| match value.as_str() {
            "disabled" => Some(CheckpointPreference::Disabled),
            "automatic" => Some(CheckpointPreference::Automatic),
            "required" => Some(CheckpointPreference::Required),
            _ => invalid(
                declaration,
                "checkpoint must be disabled, automatic, or required",
                diagnostics,
            ),
        })
        .unwrap_or(CheckpointPreference::Automatic);
    ResiliencePolicy {
        retries,
        checkpoint,
    }
}

fn parallelism(
    statements: &[CheckedFeaStatement],
    name: &str,
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> ParallelismLimit {
    let Some((_, value)) = assignment(statements, name) else {
        return ParallelismLimit::Auto;
    };
    match &value.kind {
        CheckedExpressionKind::Enum(value) if value == "auto" => ParallelismLimit::Auto,
        _ => positive_u32(value).map_or_else(
            || {
                invalid::<()>(
                    declaration,
                    format!("{name} must be auto or a positive integer"),
                    diagnostics,
                );
                ParallelismLimit::Auto
            },
            ParallelismLimit::Fixed,
        ),
    }
}

fn block<'a>(
    declaration: &'a CheckedFeaDeclaration,
    name: &str,
) -> Option<&'a [CheckedFeaStatement]> {
    declaration.statements.iter().find_map(|statement| {
        let CheckedFeaStatementKind::Block {
            header, statements, ..
        } = &statement.kind
        else {
            return None;
        };
        (header.trim() == name).then_some(statements.as_slice())
    })
}

pub(super) fn top_assignment<'a>(
    declaration: &'a CheckedFeaDeclaration,
    name: &str,
) -> Option<&'a CheckedExpression> {
    assignment(&declaration.statements, name).map(|(_, value)| value)
}

fn assignment<'a>(
    statements: &'a [CheckedFeaStatement],
    name: &str,
) -> Option<(FeaAssignmentRelation, &'a CheckedExpression)> {
    statements.iter().find_map(|statement| {
        let CheckedFeaStatementKind::Assignment {
            target,
            relation,
            value,
        } = &statement.kind
        else {
            return None;
        };
        matches!(target.as_slice(), [target] if target == name).then_some((*relation, value))
    })
}

fn closed_enum(
    value: &CheckedExpression,
    declaration: &CheckedFeaDeclaration,
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<String> {
    let CheckedExpressionKind::Enum(value) = &value.kind else {
        return invalid(
            declaration,
            format!("{name} requires a closed enum value"),
            diagnostics,
        );
    };
    Some(value.clone())
}

fn count_range(value: &CheckedExpression) -> Option<InclusiveCountRange> {
    match &value.kind {
        CheckedExpressionKind::Binary {
            left,
            operator: FeaBinaryOperator::Range,
            right,
        } => Some(InclusiveCountRange {
            minimum: positive_u32(left)?,
            maximum: positive_u32(right)?,
        }),
        _ => positive_u32(value).map(|count| InclusiveCountRange {
            minimum: count,
            maximum: count,
        }),
    }
}

fn positive_u32(value: &CheckedExpression) -> Option<u32> {
    let CheckedExpressionKind::Number(value) = &value.kind else {
        return None;
    };
    let value = value.parse().ok()?;
    (value > 0).then_some(value)
}

fn positive_or_zero_u16(value: &CheckedExpression) -> Option<u16> {
    let CheckedExpressionKind::Number(value) = &value.kind else {
        return None;
    };
    value.parse().ok()
}

/// A positive whole count within the shared FEA wire bound (2^53 - 1).
fn finite_u64(value: f64) -> Option<u64> {
    runmat_types::wire_integer::bounded_whole_f64(value).filter(|value| *value > 0)
}

fn validate_members(declaration: &CheckedFeaDeclaration, diagnostics: &mut Vec<FeaDiagnostic>) {
    let mut top_seen = BTreeSet::new();
    for statement in &declaration.statements {
        match &statement.kind {
            CheckedFeaStatementKind::Assignment {
                target, relation, ..
            } => {
                let [name] = target.as_slice() else {
                    error_at(
                        declaration,
                        statement.span,
                        "unknown execution-policy member",
                        diagnostics,
                    );
                    continue;
                };
                if !matches!(name.as_str(), "placement" | "backend") {
                    error_at(
                        declaration,
                        statement.span,
                        "unknown execution-policy member",
                        diagnostics,
                    );
                } else if !top_seen.insert(name.as_str()) {
                    error_at(
                        declaration,
                        statement.span,
                        format!("duplicate execution-policy member `{name}`"),
                        diagnostics,
                    );
                } else if *relation != FeaAssignmentRelation::Equal {
                    error_at(
                        declaration,
                        statement.span,
                        format!("execution-policy member `{name}` requires `=`"),
                        diagnostics,
                    );
                }
            }
            CheckedFeaStatementKind::Block {
                header, statements, ..
            } => {
                let block_name = header.trim();
                let allowed: &[&str] = match block_name {
                    "resources" => &[
                        "workers",
                        "cpu_threads_per_worker",
                        "memory_per_worker",
                        "accelerators",
                    ],
                    "fanout" => &["sweep_cases", "frequency_samples"],
                    "cooperative" => &["operator_partitions", "partitioner", "balance"],
                    "resilience" => &["retries", "checkpoint"],
                    _ => {
                        error_at(
                            declaration,
                            statement.span,
                            "unknown execution-policy block",
                            diagnostics,
                        );
                        continue;
                    }
                };
                if !top_seen.insert(block_name) {
                    error_at(
                        declaration,
                        statement.span,
                        format!("duplicate execution-policy block `{block_name}`"),
                        diagnostics,
                    );
                }
                validate_block_members(declaration, block_name, statements, allowed, diagnostics);
            }
            _ => error_at(
                declaration,
                statement.span,
                "unknown execution-policy member",
                diagnostics,
            ),
        }
    }
}

fn validate_block_members(
    declaration: &CheckedFeaDeclaration,
    block_name: &str,
    statements: &[CheckedFeaStatement],
    allowed: &[&str],
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    let mut seen = BTreeSet::new();
    for statement in statements {
        let CheckedFeaStatementKind::Assignment {
            target, relation, ..
        } = &statement.kind
        else {
            error_at(
                declaration,
                statement.span,
                format!("{block_name} accepts assignments only"),
                diagnostics,
            );
            continue;
        };
        let [name] = target.as_slice() else {
            error_at(
                declaration,
                statement.span,
                format!("unknown {block_name} member"),
                diagnostics,
            );
            continue;
        };
        if !allowed.contains(&name.as_str()) {
            error_at(
                declaration,
                statement.span,
                format!("unknown {block_name} member `{name}`"),
                diagnostics,
            );
        } else if !seen.insert(name.as_str()) {
            error_at(
                declaration,
                statement.span,
                format!("duplicate {block_name} member `{name}`"),
                diagnostics,
            );
        } else if (name == "memory_per_worker") != (*relation == FeaAssignmentRelation::LessEqual) {
            let expected = if name == "memory_per_worker" {
                "<="
            } else {
                "="
            };
            error_at(
                declaration,
                statement.span,
                format!("{block_name}.{name} requires `{expected}`"),
                diagnostics,
            );
        }
    }
}

fn invalid<T>(
    declaration: &CheckedFeaDeclaration,
    message: impl Into<String>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<T> {
    error_at(declaration, declaration.span, message, diagnostics);
    None
}

pub(super) fn error_at(
    declaration: &CheckedFeaDeclaration,
    span: runmat_types::Span,
    message: impl Into<String>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostics.push(FeaDiagnostic::error(
        "fea.semantic.execution_policy",
        message,
        FeaDiagnosticPhase::Lowering,
        Some(FeaDiagnosticLocation {
            path: declaration.module.clone(),
            span,
        }),
    ));
}
