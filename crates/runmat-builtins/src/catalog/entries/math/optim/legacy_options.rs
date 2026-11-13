//! One public legacy-parameter grammar for constructor and query boundaries.
//! Canonical spellings shared by two identities count once, not as ambiguity.
use super::{FminsearchOptionName, OptimOptionField, OptimSolver};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimParameterName {
    Legacy(FminsearchOptionName),
    Modern(OptimOptionField),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimParameterLookup {
    Matched(OptimParameterName),
    Ambiguous,
    Unknown,
}
impl OptimParameterName {
    pub const fn is_callback(self) -> bool {
        match self {
            Self::Legacy(field) => field.is_callback(),
            Self::Modern(field) => field.is_callback(),
        }
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Legacy(name) => name.name(),
            Self::Modern(name) => name.name(),
        }
    }
    pub fn resolve(input: &str) -> OptimParameterLookup {
        if input.is_empty() {
            return OptimParameterLookup::Unknown;
        }
        if let Some(field) = Self::exact(input) {
            return OptimParameterLookup::Matched(field);
        }
        Self::global_prefix(input)
    }

    fn exact(input: &str) -> Option<Self> {
        for field in FminsearchOptionName::ALL {
            if field.name().eq_ignore_ascii_case(input) {
                return Some(Self::Legacy(field));
            }
        }
        OptimOptionField::parse(input).map(Self::Modern)
    }

    /// A supplied public solver profile adds its actual published spellings.
    /// Exact aliases remain global, and a profile cannot grant value admission
    /// or infer a solver from stored option contents.
    pub fn resolve_for_solver(input: &str, solver: Option<OptimSolver>) -> OptimParameterLookup {
        if input.is_empty() {
            return OptimParameterLookup::Unknown;
        }
        if let Some(solver) = solver.filter(|solver| *solver != OptimSolver::Generic) {
            if let Some(field) = OptimOptionField::parse_for_solver(input, solver) {
                return OptimParameterLookup::Matched(Self::Modern(field));
            }
        }
        if let Some(field) = Self::exact(input) {
            return OptimParameterLookup::Matched(field);
        }
        if let Some(solver) = solver.filter(|solver| *solver != OptimSolver::Generic) {
            let mut candidate: Option<Self> = None;
            for field in OptimOptionField::ALL {
                if !solver.accepts(field) {
                    continue;
                }
                if field
                    .name_for_solver(solver)
                    .get(..input.len())
                    .is_some_and(|prefix| prefix.eq_ignore_ascii_case(input))
                {
                    let field = Self::Modern(field);
                    if candidate.is_some_and(|previous| !previous.equivalent(field, Some(solver))) {
                        return OptimParameterLookup::Ambiguous;
                    }
                    candidate = Some(field);
                }
            }
            if let Some(candidate) = candidate {
                return OptimParameterLookup::Matched(candidate);
            }
        }
        // Plain option queries historically also accept global legacy names.
        // Preserve that fallback when a selected profile adds no spelling.
        Self::global_prefix(input)
    }

    fn global_prefix(input: &str) -> OptimParameterLookup {
        let mut candidate: Option<Self> = None;
        for field in FminsearchOptionName::ALL
            .into_iter()
            .map(Self::Legacy)
            .chain(
                OptimOptionField::ALL
                    .into_iter()
                    .filter(|field| OptimSolver::Generic.accepts(*field))
                    .map(Self::Modern),
            )
        {
            if field
                .name()
                .get(..input.len())
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case(input))
            {
                if candidate.is_some_and(|previous| !previous.equivalent(field, None)) {
                    return OptimParameterLookup::Ambiguous;
                }
                candidate = Some(field);
            }
        }
        candidate.map_or(OptimParameterLookup::Unknown, OptimParameterLookup::Matched)
    }

    /// Compare declared semantic identities, not presentation spelling. No
    /// option value or algorithm is inspected to guess a missing profile.
    pub fn equivalent(self, other: Self, solver: Option<OptimSolver>) -> bool {
        let identity = |parameter: Self| {
            solver
                .and_then(|solver| OptimOptionField::parse_for_solver(parameter.name(), solver))
                .or_else(|| OptimOptionField::parse(parameter.name()))
        };
        match (identity(self), identity(other)) {
            (Some(left), Some(right)) => left == right,
            _ => self == other,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn published_profile_prefixes_do_not_guess_unprofiled_solver() {
        for (solver, input, expected) in [
            ("ga", "maxgener", "MaxIter"),
            ("ga", "fitnessl", "ObjectiveLimit"),
            ("particleswarm", "maxitera", "MaxIter"),
            ("patternsearch", "maxfunct", "MaxFunEvals"),
            ("patternsearch", "stept", "TolX"),
            ("patternsearch", "constraintt", "TolCon"),
            ("simulannealbnd", "maxitera", "MaxIter"),
            ("simulannealbnd", "maxfunct", "MaxFunEvals"),
        ] {
            let solver = OptimSolver::parse(solver).expect("adopted public solver profile");
            let OptimParameterLookup::Matched(field) =
                OptimParameterName::resolve_for_solver(input, Some(solver))
            else {
                panic!("unique published profile prefix {input}");
            };
            assert_eq!(field.name(), expected);
        }
        assert_eq!(
            OptimParameterName::resolve_for_solver("maxitera", None),
            OptimParameterLookup::Unknown
        );
        let OptimParameterLookup::Matched(display) = OptimParameterName::resolve("disp") else {
            panic!("legacy Display prefix");
        };
        assert_eq!(display.name(), "Display");
        assert_eq!(
            OptimParameterName::resolve_for_solver("disp", OptimSolver::parse("patternsearch")),
            OptimParameterLookup::Matched(OptimParameterName::Modern(OptimOptionField::Display))
        );
        assert_eq!(
            OptimParameterName::resolve_for_solver("MaxIterations", None),
            OptimParameterLookup::Matched(OptimParameterName::Modern(OptimOptionField::MaxIter))
        );
    }
    #[test]
    fn public_aliases_prefixes_and_shared_spellings_resolve_once() {
        for (input, expected) in [
            ("outp", "OutputFcn"),
            ("tolx", "TolX"),
            ("DISP", "Display"),
            ("alg", "Algorithm"),
            ("StepTolerance", "TolX"),
            ("MaxFunctionEvaluations", "MaxFunEvals"),
        ] {
            let OptimParameterLookup::Matched(field) = OptimParameterName::resolve(input) else {
                panic!("unique {input}");
            };
            assert_eq!(field.name(), expected);
        }
        for input in ["tol", "max"] {
            assert_eq!(
                OptimParameterName::resolve(input),
                OptimParameterLookup::Ambiguous
            );
        }
        for input in ["", "not_an_option", "töl"] {
            assert_eq!(
                OptimParameterName::resolve(input),
                OptimParameterLookup::Unknown
            );
        }
    }
}
