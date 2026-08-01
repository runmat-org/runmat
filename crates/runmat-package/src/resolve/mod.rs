mod candidate;
mod incompatibility;
mod request;
mod solver;

pub use candidate::{
    acquire_candidates, acquire_candidates_with_policy, CandidateIndex, CandidateMetadata,
    CandidateProvider, CandidateQuery, SourceSelectionPolicy,
};
pub use incompatibility::{Incompatibility, RequirementPath};
pub use request::{ResolutionRequest, ResolutionRequirement};
pub use solver::{resolve, Resolution, ResolutionEdge, ResolutionPackage};
