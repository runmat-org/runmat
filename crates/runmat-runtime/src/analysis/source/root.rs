use runmat_analysis_language::{ResolvedAnalysisStudy, ResolvedAnalysisSweep};

use super::{CheckedFeaSource, FeaCheckStatus, FeaRootSelector};

#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedFeaRoot {
    Study(ResolvedAnalysisStudy),
    Sweep(ResolvedAnalysisSweep),
}

impl ResolvedFeaRoot {
    pub fn selector(&self) -> FeaRootSelector {
        match self {
            Self::Study(root) => FeaRootSelector::Study {
                id: root.study.study_id.clone(),
            },
            Self::Sweep(root) => FeaRootSelector::Sweep {
                id: root.sweep.sweep_id.clone(),
            },
        }
    }

    pub fn id(&self) -> &str {
        match self {
            Self::Study(root) => &root.study.study_id,
            Self::Sweep(root) => &root.sweep.sweep_id,
        }
    }

    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Study(_) => "study",
            Self::Sweep(_) => "sweep",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FeaRootSelectionError {
    #[error("FEA source is {status} and has no runnable root")]
    SourceNotValid { status: FeaCheckStatus },
    #[error("root module `{module}` declares no study or sweep")]
    NoRoot { module: String },
    #[error(
        "root module `{module}` declares {} runnable roots; select one with {}",
        roots.len(),
        roots
            .iter()
            .map(|root| format!("--{} {}", root.kind(), root.id()))
            .collect::<Vec<_>>()
            .join(" or ")
    )]
    MultipleRoots {
        module: String,
        roots: Vec<FeaRootSelector>,
    },
    #[error("cannot resolve {selector:?} from root module `{module}`: {message}")]
    Resolution {
        module: String,
        selector: FeaRootSelector,
        message: String,
    },
}

impl CheckedFeaSource {
    pub fn resolve_root(
        &self,
        selector: Option<&FeaRootSelector>,
    ) -> Result<ResolvedFeaRoot, FeaRootSelectionError> {
        let status = self.check_result().status;
        if !status.is_valid() {
            return Err(FeaRootSelectionError::SourceNotValid { status });
        }
        let module = self.root_module.as_str();
        let selected = match selector {
            Some(selector) => selector.clone(),
            None => match self.root_selectors().as_slice() {
                [selector] => selector.clone(),
                [] => {
                    return Err(FeaRootSelectionError::NoRoot {
                        module: module.to_string(),
                    });
                }
                roots => {
                    return Err(FeaRootSelectionError::MultipleRoots {
                        module: module.to_string(),
                        roots: roots.to_vec(),
                    });
                }
            },
        };
        match &selected {
            FeaRootSelector::Study { id } => self
                .program
                .resolve_study(module, id)
                .map(ResolvedFeaRoot::Study),
            FeaRootSelector::Sweep { id } => self
                .program
                .resolve_sweep(module, id)
                .map(ResolvedFeaRoot::Sweep),
        }
        .map_err(|error| FeaRootSelectionError::Resolution {
            module: module.to_string(),
            selector: selected,
            message: error.to_string(),
        })
    }
}
