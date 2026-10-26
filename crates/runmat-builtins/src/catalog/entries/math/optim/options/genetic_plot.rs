//! Public GA plot identities, separate from solver callbacks and options.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OptimGeneticPlotKind {
    BestFitness,
    BestIndividual,
    Distance,
    Expectation,
    Genealogy,
    MaxConstraint,
    Range,
    ScoreDiversity,
    Scores,
    Selection,
    Stopping,
}
impl OptimGeneticPlotKind {
    pub const ALL: [Self; 11] = [
        Self::BestFitness,
        Self::BestIndividual,
        Self::Distance,
        Self::Expectation,
        Self::Genealogy,
        Self::MaxConstraint,
        Self::Range,
        Self::ScoreDiversity,
        Self::Scores,
        Self::Selection,
        Self::Stopping,
    ];
    pub const fn name(self) -> &'static str {
        match self {
            Self::BestFitness => "gaplotbestf",
            Self::BestIndividual => "gaplotbestindiv",
            Self::Distance => "gaplotdistance",
            Self::Expectation => "gaplotexpectation",
            Self::Genealogy => "gaplotgenealogy",
            Self::MaxConstraint => "gaplotmaxconstr",
            Self::Range => "gaplotrange",
            Self::ScoreDiversity => "gaplotscorediversity",
            Self::Scores => "gaplotscores",
            Self::Selection => "gaplotselection",
            Self::Stopping => "gaplotstopping",
        }
    }
    pub const fn title(self) -> &'static str {
        match self {
            Self::BestFitness => "Best and Mean Fitness",
            Self::BestIndividual => "Best Individual",
            Self::Distance => "Mean Population Distance",
            Self::Expectation => "Selection Expectation",
            Self::Genealogy => "Population Genealogy",
            Self::MaxConstraint => "Maximum Nonlinear Constraint",
            Self::Range => "Score Range",
            Self::ScoreDiversity => "Score Diversity",
            Self::Scores => "Population Scores",
            Self::Selection => "Parent Selection",
            Self::Stopping => "Stopping Criteria",
        }
    }
    pub const fn is_bar(self) -> bool {
        matches!(
            self,
            Self::BestIndividual
                | Self::Expectation
                | Self::ScoreDiversity
                | Self::Scores
                | Self::Selection
                | Self::Stopping
        )
    }
    pub const fn x_label(self) -> &'static str {
        match self {
            Self::BestIndividual => "Variable",
            Self::Expectation | Self::Scores | Self::Selection => "Individual",
            Self::ScoreDiversity => "Score interval",
            Self::Stopping => "Stopping criterion",
            Self::Genealogy => "Variation step",
            _ => "Generation",
        }
    }
    pub const fn y_label(self) -> &'static str {
        match self {
            Self::BestFitness | Self::Range | Self::Scores => "Score",
            Self::BestIndividual => "Coordinate",
            Self::Distance => "Distance",
            Self::Expectation => "Expected offspring",
            Self::Genealogy => "Individual",
            Self::MaxConstraint => "Violation",
            Self::ScoreDiversity | Self::Selection => "Count",
            Self::Stopping => "Fraction of stopping limit",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OptimGeneticPlotSeries {
    Primary,
    Secondary,
    Tertiary,
}
impl OptimGeneticPlotSeries {
    pub const fn label(self, kind: OptimGeneticPlotKind) -> Option<&'static str> {
        match (kind, self) {
            (OptimGeneticPlotKind::BestFitness, Self::Primary) => Some("Best"),
            (OptimGeneticPlotKind::BestFitness, Self::Secondary) => Some("Mean"),
            (OptimGeneticPlotKind::Range, Self::Primary) => Some("Minimum"),
            (OptimGeneticPlotKind::Range, Self::Secondary) => Some("Maximum"),
            (OptimGeneticPlotKind::Range, Self::Tertiary) => Some("Mean"),
            _ => None,
        }
    }
}
