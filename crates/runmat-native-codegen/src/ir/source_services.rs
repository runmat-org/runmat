//! Runtime service demands reconstructed from retained canonical MIR payloads.
//! Display names and general Unknown effects cannot grant source evaluation.

use super::{NativeAssembly, NativeOperation};
use runmat_builtins::BuiltinWorkspaceEffect;
use runmat_execution::host::NativeDeploymentServiceKind;
use runmat_mir::{MirExpressionStep, MirRvalue, MirStmtKind};
use std::collections::BTreeSet;

impl NativeAssembly {
    /// Complete selected deployment-service demand for this exact Native IR.
    ///
    /// `required_runtime_services` owns the non-effect supplement (currently
    /// source evaluation). Effect-backed services remain derived by the
    /// canonical deployment policy; callers must not reconstruct only one
    /// half of that closure.
    pub fn required_deployment_services(&self) -> BTreeSet<NativeDeploymentServiceKind> {
        deployment_services(&self.required_effects(), self.required_runtime_services())
    }

    pub fn required_runtime_services(&self) -> BTreeSet<NativeDeploymentServiceKind> {
        let mut services = BTreeSet::new();
        if self
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .any(|instruction| match &instruction.operation {
                NativeOperation::Rvalue { value, .. } => rvalue_requires_source_evaluation(value),
                NativeOperation::Statement(statement) => {
                    statement_requires_source_evaluation(statement)
                }
            })
        {
            services.insert(NativeDeploymentServiceKind::SourceEvaluation);
        }
        services
    }
}

fn deployment_services(
    effects: &runmat_types::EffectSet,
    supplemental: BTreeSet<NativeDeploymentServiceKind>,
) -> BTreeSet<NativeDeploymentServiceKind> {
    effects
        .0
        .iter()
        .copied()
        .filter_map(NativeDeploymentServiceKind::for_effect)
        .chain(supplemental)
        .collect()
}

fn dynamic_eval(effect: Option<BuiltinWorkspaceEffect>) -> bool {
    matches!(
        effect,
        Some(
            BuiltinWorkspaceEffect::DynamicEval
                | BuiltinWorkspaceEffect::DynamicEvalCurrentWorkspace
        )
    )
}

fn steps_require_source_evaluation(steps: &[MirExpressionStep]) -> bool {
    steps.iter().any(|step| match step {
        MirExpressionStep::Let { value, .. } => rvalue_requires_source_evaluation(value),
        MirExpressionStep::CaptureSequence { source, .. } => {
            let mut required = false;
            source.visit_direct_expression_steps(|nested| {
                required |= steps_require_source_evaluation(nested)
            });
            required
        }
    })
}

fn rvalue_requires_source_evaluation(value: &MirRvalue) -> bool {
    if matches!(value, MirRvalue::Call(call) if dynamic_eval(call.workspace_effect)) {
        return true;
    }
    if let MirRvalue::ShortCircuit { right_temps, .. } = value {
        if right_temps
            .iter()
            .any(|statement| statement_requires_source_evaluation(&statement.kind))
        {
            return true;
        }
    }
    let mut required = false;
    value.visit_direct_expression_steps(|steps| required |= steps_require_source_evaluation(steps));
    required
}

fn statement_requires_source_evaluation(statement: &MirStmtKind) -> bool {
    if matches!(
        statement,
        MirStmtKind::WorkspaceEffect {
            effect: runmat_types::WorkspaceEffect::DynamicEval,
            ..
        }
    ) {
        return true;
    }
    let direct = match statement {
        MirStmtKind::PrepareFixedAssignment(preparation) => {
            steps_require_source_evaluation(preparation.steps())
        }
        MirStmtKind::PreparedAssign { value, .. }
        | MirStmtKind::Assign { value, .. }
        | MirStmtKind::MultiAssign { value, .. }
        | MirStmtKind::SequenceAssign { value, .. }
        | MirStmtKind::Expr(value) => rvalue_requires_source_evaluation(value),
        MirStmtKind::CaptureSequence { source, .. } => {
            let mut required = false;
            source.visit_direct_expression_steps(|steps| {
                required |= steps_require_source_evaluation(steps)
            });
            required
        }
        MirStmtKind::PlaceMutation(_)
        | MirStmtKind::StatementOutput(_)
        | MirStmtKind::WorkspaceEffect { .. }
        | MirStmtKind::EnvironmentEffect(_) => false,
    };
    if direct {
        return true;
    }
    let mut required = false;
    statement.visit_expression_regions(|region| {
        required |= steps_require_source_evaluation(region.steps())
    });
    required
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deployment_service_closure_uses_canonical_effect_policy() {
        let effects = runmat_types::EffectSet(BTreeSet::from([
            runmat_types::EffectKind::HostCallback,
            runmat_types::EffectKind::Unknown,
            runmat_types::EffectKind::MayThrow,
        ]));
        let services = deployment_services(
            &effects,
            BTreeSet::from([NativeDeploymentServiceKind::SourceEvaluation]),
        );
        assert_eq!(
            services,
            BTreeSet::from([
                NativeDeploymentServiceKind::Effect(runmat_types::EffectKind::HostCallback,),
                NativeDeploymentServiceKind::HostedDynamicDispatch,
                NativeDeploymentServiceKind::SourceEvaluation,
            ])
        );
    }

    #[test]
    fn only_typed_dynamic_workspace_effect_requires_source_service() {
        let dynamic = MirStmtKind::WorkspaceEffect {
            effect: runmat_types::WorkspaceEffect::DynamicEval,
            bindings: Vec::new(),
        };
        assert!(statement_requires_source_evaluation(&dynamic));
        let ordinary = MirStmtKind::WorkspaceEffect {
            effect: runmat_types::WorkspaceEffect::ClearsBinding,
            bindings: Vec::new(),
        };
        assert!(!statement_requires_source_evaluation(&ordinary));
    }
}
