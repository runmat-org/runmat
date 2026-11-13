use runmat_analysis_core::{AcceleratorPreference, BackendPreference, PlacementPreference};
use runmat_types::CapabilityRequirement;

use crate::{
    CheckedFeaDeclaration, ExecutionPlacement, ExecutionProfileIdentity, ExecutionProfileRequest,
    ExecutionProfileSource, FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase,
};

pub(super) fn validate_named_profile<S: ExecutionProfileSource + ?Sized>(
    source: &S,
    declaration: &CheckedFeaDeclaration,
    placement: &PlacementPreference,
    backend: BackendPreference,
    accelerators: AcceleratorPreference,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    let PlacementPreference::Cluster {
        profile: Some(profile),
    } = placement
    else {
        return;
    };
    let Ok(identity) = ExecutionProfileIdentity::new(profile.clone()) else {
        push_error(
            declaration,
            "execution profile name must not be empty".to_string(),
            diagnostics,
        );
        return;
    };
    let facts = match source.resolve(&ExecutionProfileRequest {
        profile: identity.clone(),
    }) {
        Ok(facts) => facts,
        Err(error) => {
            let mut diagnostic = FeaDiagnostic::error(
                error.code,
                error.message,
                FeaDiagnosticPhase::Semantic,
                Some(location(declaration)),
            );
            diagnostic.availability = error.availability;
            diagnostics.push(diagnostic);
            return;
        }
    };
    if facts.profile != identity {
        push_error(
            declaration,
            format!(
                "execution-profile resolver returned `{}` for requested `{identity}`",
                facts.profile
            ),
            diagnostics,
        );
        return;
    }
    if facts.placement != ExecutionPlacement::Remote
        || !facts
            .capabilities
            .0
            .contains(&CapabilityRequirement::DistributedRuntime)
    {
        push_error(
            declaration,
            format!("execution profile `{identity}` must admit remote distributed execution"),
            diagnostics,
        );
    }
    if (backend == BackendPreference::Accelerator
        || accelerators == AcceleratorPreference::Required)
        && !facts
            .capabilities
            .0
            .contains(&CapabilityRequirement::Accelerator)
    {
        push_error(
            declaration,
            format!("execution profile `{identity}` does not admit accelerator execution"),
            diagnostics,
        );
    }
}

fn push_error(
    declaration: &CheckedFeaDeclaration,
    message: String,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostics.push(FeaDiagnostic::error(
        "fea.semantic.invalid_execution_profile",
        message,
        FeaDiagnosticPhase::Semantic,
        Some(location(declaration)),
    ));
}

fn location(declaration: &CheckedFeaDeclaration) -> FeaDiagnosticLocation {
    FeaDiagnosticLocation {
        path: declaration.module.clone(),
        span: declaration.span,
    }
}
