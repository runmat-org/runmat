use super::identity::{AnalysisProcedure, CoupledPhysicsFamily, PhysicsFamily, SolutionSpace};

/// Whether a family runs a procedure in this release, or is planned with a stable pending id.
///
/// The pending id is the key shared by `runmat check` diagnostics (`pending:<id>`), the docs'
/// pending-feature registry and the agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnalysisIdentityStatus {
    Runs,
    Pending(&'static str),
}

/// One row of the closed family x procedure x solution-space registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AnalysisIdentity {
    pub family: PhysicsFamily,
    pub procedure: AnalysisProcedure,
    pub solution_space: SolutionSpace,
    pub status: AnalysisIdentityStatus,
}

const fn row(
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    solution_space: SolutionSpace,
    status: AnalysisIdentityStatus,
) -> AnalysisIdentity {
    AnalysisIdentity {
        family,
        procedure,
        solution_space,
        status,
    }
}

use AnalysisIdentityStatus::{Pending, Runs};
use AnalysisProcedure as P;
use SolutionSpace::{Direct, ModalReduced};
const STRUCTURAL: PhysicsFamily = PhysicsFamily::Structural;
const THERMAL: PhysicsFamily = PhysicsFamily::Thermal;
const EM: PhysicsFamily = PhysicsFamily::Electromagnetic;
const ACOUSTIC: PhysicsFamily = PhysicsFamily::Acoustic;
const FLUID: PhysicsFamily = PhysicsFamily::Fluid;
const TM: PhysicsFamily = PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical);
const ET: PhysicsFamily = PhysicsFamily::Coupled(CoupledPhysicsFamily::Electrothermal);
const CHT: PhysicsFamily = PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer);
const FSI: PhysicsFamily = PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction);

/// The one closed registry of analysis identities the language can author. `runmat check`, the
/// native execution gate and the docs status matrix all read this table; a combination that is
/// absent is not a procedure of that family.
pub const ANALYSIS_IDENTITIES: &[AnalysisIdentity] = &[
    row(STRUCTURAL, P::Static, Direct, Runs),
    row(STRUCTURAL, P::ModalExtraction, Direct, Runs),
    row(STRUCTURAL, P::PrestressedModalExtraction, Direct, Runs),
    row(STRUCTURAL, P::EigenvalueBuckling, Direct, Runs),
    row(STRUCTURAL, P::TransientResponse, Direct, Runs),
    row(STRUCTURAL, P::HarmonicResponse, Direct, Runs),
    row(STRUCTURAL, P::PrestressedHarmonicResponse, Direct, Runs),
    row(STRUCTURAL, P::Nonlinear, Direct, Runs),
    row(STRUCTURAL, P::TransientResponse, ModalReduced, Runs),
    row(STRUCTURAL, P::HarmonicResponse, ModalReduced, Runs),
    row(
        STRUCTURAL,
        P::PrestressedHarmonicResponse,
        ModalReduced,
        Runs,
    ),
    row(STRUCTURAL, P::ModalResponseSpectrum, ModalReduced, Runs),
    row(STRUCTURAL, P::RandomResponse, ModalReduced, Runs),
    row(THERMAL, P::Static, Direct, Runs),
    row(THERMAL, P::TransientResponse, Direct, Runs),
    row(EM, P::Static, Direct, Runs),
    row(EM, P::HarmonicResponse, Direct, Runs),
    row(
        EM,
        P::TransientResponse,
        Direct,
        Pending("electromagnetic.transient"),
    ),
    row(
        EM,
        P::Nonlinear,
        Direct,
        Pending("electromagnetic.nonlinear"),
    ),
    row(ACOUSTIC, P::HarmonicResponse, Direct, Runs),
    row(ACOUSTIC, P::ModalExtraction, Direct, Runs),
    row(ACOUSTIC, P::TransientResponse, Direct, Runs),
    row(ACOUSTIC, P::Static, Direct, Pending("acoustic.static")),
    row(
        ACOUSTIC,
        P::Nonlinear,
        Direct,
        Pending("acoustic.nonlinear"),
    ),
    row(FLUID, P::Static, Direct, Runs),
    row(FLUID, P::TransientResponse, Direct, Runs),
    row(
        FLUID,
        P::HarmonicResponse,
        Direct,
        Pending("fluid.harmonic"),
    ),
    row(FLUID, P::Nonlinear, Direct, Pending("fluid.nonlinear")),
    row(TM, P::Static, Direct, Runs),
    row(TM, P::Nonlinear, Direct, Runs),
    row(
        TM,
        P::TransientResponse,
        Direct,
        Pending("thermomechanical.transient"),
    ),
    row(
        TM,
        P::HarmonicResponse,
        Direct,
        Pending("thermomechanical.harmonic"),
    ),
    row(ET, P::Static, Direct, Runs),
    row(ET, P::TransientResponse, Direct, Runs),
    row(
        ET,
        P::HarmonicResponse,
        Direct,
        Pending("electrothermal.harmonic"),
    ),
    row(
        ET,
        P::Nonlinear,
        Direct,
        Pending("electrothermal.nonlinear"),
    ),
    row(CHT, P::Static, Direct, Runs),
    row(CHT, P::TransientResponse, Direct, Runs),
    row(
        CHT,
        P::HarmonicResponse,
        Direct,
        Pending("conjugate-heat-transfer.harmonic"),
    ),
    row(
        CHT,
        P::Nonlinear,
        Direct,
        Pending("conjugate-heat-transfer.nonlinear"),
    ),
    row(FSI, P::Static, Direct, Runs),
    row(FSI, P::TransientResponse, Direct, Runs),
    row(
        FSI,
        P::HarmonicResponse,
        Direct,
        Pending("fluid-structure-interaction.harmonic"),
    ),
    row(
        FSI,
        P::Nonlinear,
        Direct,
        Pending("fluid-structure-interaction.nonlinear"),
    ),
];

/// The registry status of one identity, or `None` when it is not a procedure of the family.
pub fn analysis_identity_status(
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    solution_space: SolutionSpace,
) -> Option<AnalysisIdentityStatus> {
    ANALYSIS_IDENTITIES
        .iter()
        .find(|row| {
            row.family == family
                && row.procedure == procedure
                && row.solution_space == solution_space
        })
        .map(|row| row.status)
}

/// An identity that runs in this release.
pub fn is_exposed_analysis_identity(
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    solution_space: SolutionSpace,
) -> bool {
    analysis_identity_status(family, procedure, solution_space) == Some(Runs)
}

/// User-facing family name, as authored in `.fea` sources.
pub const fn family_label(family: PhysicsFamily) -> &'static str {
    match family {
        PhysicsFamily::Structural => "structural",
        PhysicsFamily::Thermal => "thermal",
        PhysicsFamily::Electromagnetic => "electromagnetic",
        PhysicsFamily::Acoustic => "acoustic",
        PhysicsFamily::Fluid => "fluid",
        PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical) => "thermomechanical",
        PhysicsFamily::Coupled(CoupledPhysicsFamily::Electrothermal) => "electrothermal",
        PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer) => {
            "conjugate_heat_transfer"
        }
        PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction) => {
            "fluid_structure_interaction"
        }
    }
}

/// User-facing procedure name, as authored in `.fea` sources.
pub const fn procedure_label(procedure: AnalysisProcedure) -> &'static str {
    match procedure {
        P::Static => "static",
        P::ModalExtraction => "modal",
        P::PrestressedModalExtraction => "prestressed_modal",
        P::EigenvalueBuckling => "buckling",
        P::TransientResponse => "transient",
        P::HarmonicResponse => "harmonic",
        P::PrestressedHarmonicResponse => "prestressed_harmonic",
        P::ModalResponseSpectrum => "response_spectrum",
        P::RandomResponse => "random_response",
        P::Nonlinear => "nonlinear",
        P::Adaptive => "adaptive",
    }
}

/// The docs status matrix, generated from [`ANALYSIS_IDENTITIES`]. A test keeps the committed
/// copy (docs/simulation/generated/family-procedure-matrix.md) identical to this rendering.
pub fn render_analysis_identity_matrix() -> String {
    let mut out = String::from(
        "---\n\
         description: Every physics family and analysis procedure RunMat FEA knows, with its solution space and whether the current release runs it or lists it as planned.\n\
         keywords: [family, procedure, analysis matrix, release status, supported analyses, FEA]\n\
         doc_type: release-status\n\
         generated: RUNMAT_BLESS_DOCS=1 cargo test -p runmat-analysis-core --lib committed_docs_matrix_matches_the_registry\n\
         ---\n\
         # Families and procedures\n\n\
         <!-- Generated from runmat-analysis-core ANALYSIS_IDENTITIES; do not edit by hand. -->\n\n\
         Each row is one family and procedure pair from the closed registry that `runmat check`, \
         `runmat run` and the qualification gate share. A planned pair links its pending id; \
         `runmat check` reports the same id when a study uses it.\n\n\
         | Family | Procedure | Solution space | Status |\n\
         |---|---|---|---|\n",
    );
    for row in ANALYSIS_IDENTITIES {
        let space = match row.solution_space {
            Direct => "direct",
            ModalReduced => "modal reduced",
        };
        let status = match row.status {
            Runs => "runs".to_string(),
            Pending(id) => format!("[not yet in the current release](pending:{id})"),
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            family_label(row.family),
            procedure_label(row.procedure),
            space,
            status
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_current_exposed_identity_is_in_the_closed_table() {
        let identities = [
            (PhysicsFamily::Structural, AnalysisProcedure::Static),
            (
                PhysicsFamily::Structural,
                AnalysisProcedure::ModalExtraction,
            ),
            (
                PhysicsFamily::Structural,
                AnalysisProcedure::PrestressedModalExtraction,
            ),
            (
                PhysicsFamily::Structural,
                AnalysisProcedure::EigenvalueBuckling,
            ),
            (
                PhysicsFamily::Structural,
                AnalysisProcedure::TransientResponse,
            ),
            (
                PhysicsFamily::Structural,
                AnalysisProcedure::HarmonicResponse,
            ),
            (PhysicsFamily::Structural, AnalysisProcedure::Nonlinear),
            (
                PhysicsFamily::Structural,
                AnalysisProcedure::PrestressedHarmonicResponse,
            ),
            (PhysicsFamily::Thermal, AnalysisProcedure::Static),
            (PhysicsFamily::Thermal, AnalysisProcedure::TransientResponse),
            (PhysicsFamily::Electromagnetic, AnalysisProcedure::Static),
            (
                PhysicsFamily::Electromagnetic,
                AnalysisProcedure::HarmonicResponse,
            ),
            (PhysicsFamily::Acoustic, AnalysisProcedure::HarmonicResponse),
            (PhysicsFamily::Acoustic, AnalysisProcedure::ModalExtraction),
            (
                PhysicsFamily::Acoustic,
                AnalysisProcedure::TransientResponse,
            ),
            (PhysicsFamily::Fluid, AnalysisProcedure::Static),
            (PhysicsFamily::Fluid, AnalysisProcedure::TransientResponse),
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer),
                AnalysisProcedure::Static,
            ),
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer),
                AnalysisProcedure::TransientResponse,
            ),
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction),
                AnalysisProcedure::Static,
            ),
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction),
                AnalysisProcedure::TransientResponse,
            ),
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
                AnalysisProcedure::Static,
            ),
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
                AnalysisProcedure::Nonlinear,
            ),
        ];

        for (family, procedure) in identities {
            assert!(is_exposed_analysis_identity(
                family,
                procedure,
                SolutionSpace::Direct
            ));
        }
        assert!(!is_exposed_analysis_identity(
            PhysicsFamily::Acoustic,
            AnalysisProcedure::Static,
            SolutionSpace::Direct
        ));
        assert!(!is_exposed_analysis_identity(
            PhysicsFamily::Structural,
            AnalysisProcedure::Static,
            SolutionSpace::ModalReduced
        ));
        assert!(is_exposed_analysis_identity(
            PhysicsFamily::Structural,
            AnalysisProcedure::TransientResponse,
            SolutionSpace::ModalReduced
        ));
        assert!(is_exposed_analysis_identity(
            PhysicsFamily::Structural,
            AnalysisProcedure::HarmonicResponse,
            SolutionSpace::ModalReduced
        ));
        assert!(is_exposed_analysis_identity(
            PhysicsFamily::Structural,
            AnalysisProcedure::ModalResponseSpectrum,
            SolutionSpace::ModalReduced
        ));
        assert!(is_exposed_analysis_identity(
            PhysicsFamily::Structural,
            AnalysisProcedure::RandomResponse,
            SolutionSpace::ModalReduced
        ));
        assert!(!is_exposed_analysis_identity(
            PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
            AnalysisProcedure::TransientResponse,
            SolutionSpace::Direct
        ));
    }

    #[test]
    fn registry_rows_are_unique_and_pending_ids_are_stable_keys() {
        let mut seen = std::collections::BTreeSet::new();
        for row in ANALYSIS_IDENTITIES {
            assert!(
                seen.insert(format!(
                    "{:?}{:?}{:?}",
                    row.family, row.procedure, row.solution_space
                )),
                "duplicate registry row {row:?}"
            );
            if let Pending(id) = row.status {
                assert!(
                    !id.is_empty()
                        && id
                            .chars()
                            .all(|c| c.is_ascii_lowercase() || c == '.' || c == '-'),
                    "{id}"
                );
            }
        }
    }

    #[test]
    fn committed_docs_matrix_matches_the_registry() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../docs/simulation/generated/family-procedure-matrix.md"
        );
        if std::env::var_os("RUNMAT_BLESS_DOCS").is_some() {
            std::fs::write(path, render_analysis_identity_matrix()).expect("write status matrix");
        }
        let committed = std::fs::read_to_string(path).expect("committed status matrix");
        assert_eq!(
            committed,
            render_analysis_identity_matrix(),
            "regenerate {path} from render_analysis_identity_matrix()"
        );
    }
}
