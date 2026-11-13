//! Portable serial numerical host for admitted structural solve/state tasks.
//! Desktop and browser share these numerical receipt and resource checks.
use crate::*;
use runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan;
use runmat_analysis_fea::execution::*;
use runmat_analysis_fea::planning::resource::*;
use runmat_analysis_fea::{
    run_linear_static_with_prepared_constraint_plan, LinearStaticSolveOptions,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
mod structural_linear;
mod state_publication;

#[derive(Clone, Copy, Debug, Default)]
pub struct SerialStructuralAnalysisKernel;
impl AnalysisKernel for SerialStructuralAnalysisKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        match invocation.kind {
            AnalysisWorkKind::Solve => structural_linear::execute(invocation),
            AnalysisWorkKind::PublishState => state_publication::execute(invocation),
            _ => Err(AnalysisBridgeError::Invalid(
                "This task requires an execution kernel not provided by this host.".into(),
            )),
        }
    }

    fn execute_with_capacity<'a>(
        &'a self,
        invocation: AnalysisKernelInvocation<'a>,
        capacity: &'a dyn AsyncCapacityOwner,
    ) -> AnalysisKernelFuture<'a> {
        Box::pin(async move {
            match invocation.kind {
                AnalysisWorkKind::Solve => {
                    structural_linear::execute_with_capacity(invocation, Some(capacity)).await
                }
                _ => self.execute(invocation),
            }
        })
    }

    fn execute_reserved_with_capacity<'a>(
        &'a self,
        invocation: ReservedAnalysisKernelInvocation<'a>,
        capacity: &'a dyn AsyncCapacityOwner,
    ) -> AnalysisKernelFuture<'a> {
        self.execute_with_capacity(invocation.into_invocation(), capacity)
    }
}
#[doc(hidden)]
pub fn execute_structural_linear_kernel(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    structural_linear::execute(invocation)
}
#[doc(hidden)]
pub fn execute_state_publication_kernel(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    state_publication::execute(invocation)
}
fn unique_input<'a>(
    invocation: &'a AnalysisKernelInvocation<'_>,
    kind: AnalysisNumericalObjectKind,
) -> AnalysisBridgeResult<&'a crate::PreparedAnalysisObjects> {
    let mut matches = invocation
        .inputs
        .iter()
        .filter(|input| input.manifest.kind == kind);
    let input = matches.next().ok_or_else(|| {
        AnalysisBridgeError::Invalid(format!("analysis workload is missing its {kind:?} input"))
    })?;
    if matches.next().is_some() {
        return Err(AnalysisBridgeError::Invalid(format!(
            "analysis workload has multiple {kind:?} inputs"
        )));
    }
    Ok(input)
}

fn optional_unique_input<'a>(
    invocation: &'a AnalysisKernelInvocation<'_>,
    kind: AnalysisNumericalObjectKind,
) -> AnalysisBridgeResult<Option<&'a crate::PreparedAnalysisObjects>> {
    let mut matches = invocation
        .inputs
        .iter()
        .filter(|input| input.manifest.kind == kind);
    let input = matches.next();
    if matches.next().is_some() {
        return Err(AnalysisBridgeError::Invalid(format!(
            "analysis workload has multiple {kind:?} inputs"
        )));
    }
    Ok(input)
}

fn constraint_plan_manifest_identity(
    invocation: &AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<runmat_execution::Digest> {
    Ok(
        unique_input(invocation, AnalysisNumericalObjectKind::ConstraintPlan)?
            .manifest
            .logical_identity,
    )
}

mod host;
pub use host::{
    execute_serial_structural_sequence, execute_serial_structural_sequence_with_capacity,
    SerialAnalysisHost, SerialAnalysisServices,
};
