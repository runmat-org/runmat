//! Shared semantic identities for optimization option builders.
//!
//! Public spellings are decoded here. Option structs retain their historical
//! field names for solver interoperability, but neither builder dispatches on
//! those presentation strings after admission.

mod controls;
mod fields;
mod multiobjective;
mod observation;
mod quasi_newton;
mod solver;
mod pattern;
pub use controls::{
    OptimAlgorithm, OptimDifferenceType, OptimDisplay, OptimHessianSource,
    OptimSubproblemAlgorithm, OptimSwitch,
};
pub use fields::OptimOptionField;
pub use multiobjective::OptimMeritFunction;
pub use observation::OptimObserverState;
pub use pattern::{
    OptimPatternAlgorithm, OptimPatternDefault, OptimPatternOrder, OptimPatternPoll,
    OptimPatternScalarRule, OptimPatternSearch,
};
pub use quasi_newton::OptimQuasiNewtonUpdate;
pub use solver::OptimSolver;

mod global;
pub use global::{OptimHybridSolver, OptimSwarmCreation};

mod annealing;
pub use annealing::{OptimAcceptance, OptimAnnealData, OptimAnnealing, OptimTemperature};

mod genetic;
mod genetic_plot;
mod genetic_profile;
pub use genetic::{
    OptimGeneticAction, OptimGeneticConstraint, OptimGeneticCreation, OptimGeneticCrossover,
    OptimGeneticMigration, OptimGeneticMutation, OptimGeneticObserverState, OptimGeneticPopulation,
    OptimGeneticScaling, OptimGeneticSelection, OptimGeneticStall, OptimGeneticStateField,
};
pub use genetic_plot::{OptimGeneticPlotKind, OptimGeneticPlotSeries};
