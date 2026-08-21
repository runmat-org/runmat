mod candidate;
mod explain;
mod incompatibility;
mod request;
mod solver;
mod update;

pub use candidate::{
    acquire_candidates, acquire_candidates_with_policy, CandidateIndex, CandidateMetadata,
    CandidateProvider, CandidateQuery, SourceSelectionPolicy,
};
pub use explain::{dependency_tree, why};
pub use incompatibility::{Incompatibility, RequirementPath};
pub use request::{ResolutionRequest, ResolutionRequirement};
pub use solver::{resolve, Resolution, ResolutionEdge, ResolutionPackage};
pub use update::{plan_update, UpdatePlan, UpdatePolicy};
