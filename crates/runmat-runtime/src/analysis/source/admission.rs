//! Static run admission reported at check time.
//!
//! Every rule a run would apply before meshing (family x procedure registry, procedure graph,
//! constraint, initial-condition and observation kinds, strict accelerator placement) runs here
//! through the same `runmat_analysis_fea::admission` functions native preparation calls, so
//! `runmat check` and `runmat run` report the same diagnostic id.

use runmat_analysis_core::AuthoredStudyInputBinding;
use runmat_analysis_fea::admission::{admit_accelerator_policy, admit_study, AdmissionIssue};
use runmat_analysis_language::{
    CheckedFeaProgram, FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase,
    FeaDiagnosticSeverity, FeaSymbolKey, FeaSymbolKind, LogicalSourcePath, ResolvedAnalysisStudy,
};

/// Sweep cases admitted per sweep. Admission depends on the authored shape, which inputs rarely
/// change; the bound keeps checking large sweeps cheap.
const MAXIMUM_ADMITTED_SWEEP_CASES: usize = 64;

pub(super) fn append_admission_diagnostics(
    program: &mut CheckedFeaProgram,
    root: &LogicalSourcePath,
) {
    if program
        .bound
        .graph
        .diagnostics
        .iter()
        .chain(&program.bound.diagnostics)
        .chain(&program.diagnostics)
        .any(|diagnostic| diagnostic.severity == FeaDiagnosticSeverity::Error)
    {
        return;
    }
    let module = root.as_str();
    let mut diagnostics = Vec::new();
    let studies = program
        .studies
        .values()
        .filter(|study| study.module_path == module)
        .map(|study| study.study_id.clone())
        .collect::<Vec<_>>();
    for id in studies {
        let Ok(resolved) = program.resolve_study(module, &id) else {
            continue;
        };
        let issues = admit_resolved(&resolved, &[Vec::new()]);
        push(program, root, FeaSymbolKind::Study, &id, issues, &mut diagnostics);
    }
    let sweeps = program
        .sweeps
        .values()
        .filter(|sweep| sweep.module_path == module)
        .map(|sweep| sweep.sweep_id.clone())
        .collect::<Vec<_>>();
    for id in sweeps {
        let Ok(resolved) = program.resolve_sweep(module, &id) else {
            continue;
        };
        let Ok(cases) = resolved.sweep.expand_cases(&resolved.study.study) else {
            continue;
        };
        let inputs = cases
            .into_iter()
            .take(MAXIMUM_ADMITTED_SWEEP_CASES)
            .map(|case| {
                case.bindings
                    .into_iter()
                    .map(|binding| AuthoredStudyInputBinding {
                        input_id: binding.input_id,
                        value: binding.value,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let issues = admit_resolved(&resolved.study, &inputs);
        push(program, root, FeaSymbolKind::Sweep, &id, issues, &mut diagnostics);
    }
    program.diagnostics.extend(diagnostics);
}

/// Admission issues over each input set the root realizes, deduplicated. A study whose inputs
/// cannot be realized without a sweep case is left to run time.
fn admit_resolved(
    resolved: &ResolvedAnalysisStudy,
    input_sets: &[Vec<AuthoredStudyInputBinding>],
) -> Vec<AdmissionIssue> {
    let mut issues = Vec::new();
    for inputs in input_sets {
        let Ok(realized) =
            resolved
                .study
                .realize_with_models(&resolved.models, &resolved.geometry, inputs)
        else {
            continue;
        };
        let found = match admit_study(&realized, &resolved.material_catalog) {
            Ok(_) => admit_accelerator_policy(
                &realized.sequence.steps,
                resolved.execution_policy.as_ref(),
            )
            .err()
            .into_iter()
            .collect(),
            Err(found) => found,
        };
        for issue in found {
            if !issues.contains(&issue) {
                issues.push(issue);
            }
        }
    }
    issues
}

fn push(
    program: &CheckedFeaProgram,
    root: &LogicalSourcePath,
    kind: FeaSymbolKind,
    id: &str,
    issues: Vec<AdmissionIssue>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    let location = program
        .declarations
        .get(&(
            root.clone(),
            FeaSymbolKey {
                kind,
                name: id.to_string(),
            },
        ))
        .map(|declaration| FeaDiagnosticLocation {
            path: root.clone(),
            span: declaration.span,
        });
    let noun = match kind {
        FeaSymbolKind::Sweep => "sweep",
        _ => "study",
    };
    for issue in issues {
        let mut diagnostic = FeaDiagnostic::error(
            issue.code,
            format!("{noun} `{id}`: {}", issue.display_message()),
            FeaDiagnosticPhase::Lowering,
            location.clone(),
        );
        if let Some(pending) = issue.pending {
            diagnostic = diagnostic.with_label(format!("pending:{pending}"));
        }
        diagnostics.push(diagnostic);
    }
}
