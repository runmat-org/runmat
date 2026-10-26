//! Scalar optimization contracts and shared option identities.
mod coneprog;
mod eqnproblem;
mod fgoalattain;
mod fminbnd;
mod fmincon;
mod fminimax;
mod fminsearch;
mod fminunc;
mod fseminf;
mod fsolve;
mod fzero;
mod ga;
mod integer_constraint;
mod integral;
mod intlinprog;
mod legacy_options;
mod linprog;
mod lsqcurvefit;
mod lsqlin;
mod lsqnonlin;
mod lsqnonneg;
mod milp_observers;
mod multiobjective_documentation;
mod multiobjective_facts;
mod nonlinear_facts;
#[cfg(test)]
mod nonlinear_tests;
mod optimget;
mod optimmodel;
mod optimoptions;
mod optimplot;
mod optimproblem;
mod optimset;
mod optimvar;
mod optimwarmstart;
mod options;
#[cfg(test)]
mod options_tests;
mod particleswarm;
mod patternsearch;
mod program_facts;
mod quad;
mod quadprog;
mod scalar_facts;
mod secondordercone;
mod simulannealbnd;
mod surrogateopt;

pub use coneprog::*;
pub use eqnproblem::*;
pub use fgoalattain::*;
pub use fminbnd::*;
pub use fmincon::*;
pub use fminimax::*;
pub use fminsearch::*;
pub use fminunc::*;
pub use fseminf::*;
pub use fsolve::*;
pub use fzero::*;
pub use ga::*;
pub use integer_constraint::*;
pub use integral::*;
pub use intlinprog::*;
pub use legacy_options::{OptimParameterLookup, OptimParameterName};
pub use linprog::*;
pub use lsqcurvefit::*;
pub use lsqlin::*;
pub use lsqnonlin::*;
pub use lsqnonneg::*;
pub use milp_observers::*;
pub use nonlinear_facts::nonlinear_real_numeric_metadata;
pub use optimget::*;
pub use optimmodel::*;
pub use optimoptions::*;
pub use optimplot::*;
pub use optimproblem::*;
pub use optimset::*;
pub use optimvar::*;
pub use optimwarmstart::*;
pub use options::{OptimAcceptance, OptimAnnealData, OptimAnnealing, OptimTemperature};
pub use options::{
    OptimAlgorithm, OptimDifferenceType, OptimDisplay, OptimHessianSource, OptimHybridSolver,
    OptimMeritFunction, OptimObserverState, OptimOptionField, OptimQuasiNewtonUpdate, OptimSolver,
    OptimSubproblemAlgorithm, OptimSwarmCreation, OptimSwitch,
};
pub use options::{
    OptimGeneticAction, OptimGeneticConstraint, OptimGeneticCreation, OptimGeneticCrossover,
    OptimGeneticMigration, OptimGeneticMutation, OptimGeneticObserverState, OptimGeneticPlotKind,
    OptimGeneticPlotSeries, OptimGeneticPopulation, OptimGeneticScaling, OptimGeneticSelection,
    OptimGeneticStall, OptimGeneticStateField,
};
pub use options::{
    OptimPatternAlgorithm, OptimPatternDefault, OptimPatternOrder, OptimPatternPoll,
    OptimPatternScalarRule, OptimPatternSearch,
};
pub use particleswarm::*;
pub use patternsearch::*;
pub use program_facts::OptimNumericInput;
pub use quad::*;
pub use quadprog::*;
pub use secondordercone::*;
pub use simulannealbnd::*;
pub use surrogateopt::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend_from_slice(ga::ENTRIES);
    values.extend_from_slice(intlinprog::ENTRIES);
    values.extend_from_slice(integer_constraint::ENTRIES);
    values.extend_from_slice(milp_observers::ENTRIES);
    values.extend(fminbnd::ENTRIES.iter().copied());
    values.extend(fmincon::ENTRIES.iter().copied());
    values.extend(fseminf::ENTRIES.iter().copied());
    values.extend(fminimax::ENTRIES.iter().copied());
    values.extend(fgoalattain::ENTRIES.iter().copied());
    values.extend(fminsearch::ENTRIES.iter().copied());
    values.extend_from_slice(particleswarm::ENTRIES);
    values.extend_from_slice(patternsearch::ENTRIES);
    values.extend_from_slice(simulannealbnd::ENTRIES);
    values.extend_from_slice(surrogateopt::ENTRIES);
    values.extend(fminunc::ENTRIES.iter().copied());
    values.extend(fsolve::ENTRIES.iter().copied());
    values.extend(lsqcurvefit::ENTRIES.iter().copied());
    values.extend(lsqnonlin::ENTRIES.iter().copied());
    values.extend(lsqnonneg::ENTRIES.iter().copied());
    values.extend(coneprog::ENTRIES.iter().copied());
    values.extend(linprog::ENTRIES.iter().copied());
    values.extend(optimoptions::ENTRIES.iter().copied());
    values.extend(optimplot::ENTRIES.iter().copied());
    values.extend(optimset::ENTRIES.iter().copied());
    values.extend(optimget::ENTRIES.iter().copied());
    values.extend_from_slice(optimvar::ENTRIES);
    values.extend_from_slice(optimwarmstart::ENTRIES);
    values.extend_from_slice(optimmodel::ENTRIES);
    values.extend_from_slice(optimproblem::ENTRIES);
    values.extend_from_slice(eqnproblem::ENTRIES);
    values.extend(secondordercone::ENTRIES.iter().copied());
    values.extend(fzero::ENTRIES.iter().copied());
    values.extend(integral::ENTRIES.iter().copied());
    values.extend(quad::ENTRIES.iter().copied());
    values.extend(quadprog::ENTRIES.iter().copied());
    values.extend(lsqlin::ENTRIES.iter().copied());
}
