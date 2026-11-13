use super::{plot_documentation as docs, *};
use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind};
const fn entry(
    kind: OptimGeneticPlotKind,
    summary: &'static str,
    examples: &'static [BuiltinExample],
) -> BuiltinCatalogEntry {
    let mut result = super::helper_entries::entry(
        kind.name(),
        &GA_PLOT_HELPER_DESCRIPTOR,
        docs::doc(kind.name(), summary, examples),
        super::plot_inference::infer,
        false,
    );
    result.contract.purity = BuiltinPurity::Impure;
    result.contract.capabilities = &[CapabilityRequirement::UserInterface];
    result.contract.effects = if matches!(kind, OptimGeneticPlotKind::Stopping) {
        &[
            EffectKind::UserInterface,
            EffectKind::Clock,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    } else {
        &[
            EffectKind::UserInterface,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    };
    result.integer_capabilities = super::plot_descriptor::INTEGERS;
    result.suppress_auto_output = true;
    result
}

pub const GAPLOTBESTF_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    OptimGeneticPlotKind::BestFitness,
    "Plot best and mean finite-real genetic scores.",
    &[docs::example(
        "state=struct('Generation',0,'Score',[1;4]); returned=gaplotbestf([],state,'init');",
    )],
);

pub const GAPLOTBESTINDIV_CATALOG_ENTRY:BuiltinCatalogEntry=entry(OptimGeneticPlotKind::BestIndividual,"Display the numeric individual with best finite-real score.",&[docs::example("state=struct('Generation',0,'Score',[4;1],'Population',[2 3;1 4]); returned=gaplotbestindiv([],state,'init');")]);

pub const GAPLOTDISTANCE_CATALOG_ENTRY:BuiltinCatalogEntry=entry(OptimGeneticPlotKind::Distance,"Plot the mean pairwise population distance.",&[docs::example("state=struct('Generation',0,'Population',[0 0;3 4]); returned=gaplotdistance([],state,'init');")]);

pub const GAPLOTEXPECTATION_CATALOG_ENTRY:BuiltinCatalogEntry=entry(OptimGeneticPlotKind::Expectation,"Display current offspring expectations.",&[docs::example("state=struct('Generation',0,'Expectation',[0.5;1.5]); returned=gaplotexpectation([],state,'init');")]);

pub const GAPLOTGENEALOGY_CATALOG_ENTRY:BuiltinCatalogEntry=entry(OptimGeneticPlotKind::Genealogy,"Draw the actual recorded parent-child genealogy.",&[docs::example("state=struct('Generation',0,'GeneticParents',NaN(2,2)); returned=gaplotgenealogy([],state,'init');")]);

pub const GAPLOTMAXCONSTR_CATALOG_ENTRY:BuiltinCatalogEntry=entry(OptimGeneticPlotKind::MaxConstraint,"Plot maximum nonlinear constraint violation.",&[docs::example("state=struct('Generation',0,'NonlinIneq',[-1;2],'NonlinEq',[0;0.5]); returned=gaplotmaxconstr([],state,'init');")]);

pub const GAPLOTRANGE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    OptimGeneticPlotKind::Range,
    "Plot minimum, maximum and mean genetic scores.",
    &[docs::example(
        "state=struct('Generation',0,'Score',[1;4;7]); returned=gaplotrange([],state,'init');",
    )],
);

pub const GAPLOTSCOREDIVERSITY_CATALOG_ENTRY:BuiltinCatalogEntry=entry(OptimGeneticPlotKind::ScoreDiversity,"Display a score-distribution histogram.",&[docs::example("state=struct('Generation',0,'Score',[1;1;2;3]); returned=gaplotscorediversity([],state,'init');")]);

pub const GAPLOTSCORES_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    OptimGeneticPlotKind::Scores,
    "Display individual genetic scores.",
    &[docs::example(
        "state=struct('Generation',0,'Score',[1;4;7]); returned=gaplotscores([],state,'init');",
    )],
);

pub const GAPLOTSELECTION_CATALOG_ENTRY:BuiltinCatalogEntry=entry(OptimGeneticPlotKind::Selection,"Count how often each parent was selected.",&[docs::example("state=struct('Generation',0,'Score',[1;4;7],'Selection',[1 1 3]); returned=gaplotselection([],state,'init');")]);

pub const GAPLOTSTOPPING_CATALOG_ENTRY:BuiltinCatalogEntry=entry(OptimGeneticPlotKind::Stopping,"Display supplied genetic stopping-criterion levels.",&[docs::example("options=optimoptions('ga','MaxGenerations',10,'MaxStallGenerations',5); state=struct('Generation',2,'LastImprovement',1,'Best',[4 3 3]); returned=gaplotstopping(options,state,'init');")]);
