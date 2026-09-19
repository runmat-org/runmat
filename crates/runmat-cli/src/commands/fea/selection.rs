use anyhow::{bail, Context, Result};
use runmat_analysis_language::{ResolvedAnalysisStudy, ResolvedAnalysisSweep};

use crate::cli::FeaRootArgs;

use super::source::CheckedFeaSource;

#[derive(Debug, Clone, PartialEq)]
pub(super) enum SelectedFeaRoot {
    Study(ResolvedAnalysisStudy),
    Sweep(ResolvedAnalysisSweep),
}

impl SelectedFeaRoot {
    pub(super) fn id(&self) -> &str {
        match self {
            Self::Study(root) => &root.study.study_id,
            Self::Sweep(root) => &root.sweep.sweep_id,
        }
    }

    pub(super) const fn kind(&self) -> &'static str {
        match self {
            Self::Study(_) => "study",
            Self::Sweep(_) => "sweep",
        }
    }
}

pub(super) fn select_root(
    source: &CheckedFeaSource,
    selection: &FeaRootArgs,
) -> Result<SelectedFeaRoot> {
    let module = source.root_module.as_str();
    if let Some(study) = selection.study.as_deref() {
        return source
            .program
            .resolve_study(module, study)
            .map(SelectedFeaRoot::Study)
            .with_context(|| format!("resolve study `{study}` from root module `{module}`"));
    }
    if let Some(sweep) = selection.sweep.as_deref() {
        return source
            .program
            .resolve_sweep(module, sweep)
            .map(SelectedFeaRoot::Sweep)
            .with_context(|| format!("resolve sweep `{sweep}` from root module `{module}`"));
    }

    let studies = source
        .program
        .studies
        .values()
        .filter(|study| study.module_path == module)
        .map(|study| ("study", study.study_id.as_str()))
        .collect::<Vec<_>>();
    let sweeps = source
        .program
        .sweeps
        .values()
        .filter(|sweep| sweep.module_path == module)
        .map(|sweep| ("sweep", sweep.sweep_id.as_str()))
        .collect::<Vec<_>>();
    let roots = studies.into_iter().chain(sweeps).collect::<Vec<_>>();
    match roots.as_slice() {
        [("study", id)] => source
            .program
            .resolve_study(module, id)
            .map(SelectedFeaRoot::Study)
            .map_err(anyhow::Error::new),
        [("sweep", id)] => source
            .program
            .resolve_sweep(module, id)
            .map(SelectedFeaRoot::Sweep)
            .map_err(anyhow::Error::new),
        [] => bail!("root module `{module}` declares no study or sweep"),
        _ => {
            let choices = roots
                .iter()
                .map(|(kind, id)| format!("{kind} `{id}`"))
                .collect::<Vec<_>>()
                .join(", ");
            bail!(
                "root module `{module}` declares multiple runnable roots ({choices}); select one with --study or --sweep"
            )
        }
    }
}
