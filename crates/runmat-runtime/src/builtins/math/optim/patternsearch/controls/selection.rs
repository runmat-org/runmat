//! Actual classic versus nonuniform selection; dormant controls stay unread.
use super::{invalid, text, unsupported, Field, ScalarOptionFields};
use crate::BuiltinResult;
use runmat_builtins::{OptimDisplay, OptimPatternAlgorithm, OptimPatternOrder, OptimPatternPoll};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::builtins::math::optim::patternsearch) struct Classic {
    pub method: OptimPatternPoll,
    pub order: OptimPatternOrder,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::builtins::math::optim::patternsearch) enum Polling {
    Classic(Classic),
    Nonuniform(OptimPatternAlgorithm),
}
impl Polling {
    pub fn classic(self) -> Option<Classic> {
        match self {
            Self::Classic(value) => Some(value),
            Self::Nonuniform(_) => None,
        }
    }
    pub fn method(self) -> Option<OptimPatternPoll> {
        self.classic().map(|classic| classic.method)
    }
    pub fn algorithm(self) -> OptimPatternAlgorithm {
        match self {
            Self::Classic(_) => OptimPatternAlgorithm::Classic,
            Self::Nonuniform(algorithm) => algorithm,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Classic(classic) => classic.method.name(),
            Self::Nonuniform(algorithm) => algorithm.name(),
        }
    }
    pub fn adaptive(self) -> bool {
        self.method().is_some_and(OptimPatternPoll::mesh_adaptive)
    }
}
pub(in crate::builtins::math::optim::patternsearch) struct Selection {
    pub polling: Polling,
    pub search: Option<super::super::search::Search>,
    pub display: OptimDisplay,
}
pub(in crate::builtins::math::optim::patternsearch) fn algorithm(
    fields: &ScalarOptionFields<'_>,
) -> BuiltinResult<OptimPatternAlgorithm> {
    OptimPatternAlgorithm::parse(&text(fields, Field::Algorithm)?)
        .ok_or_else(|| invalid("unrecognized Algorithm"))
}
pub(in crate::builtins::math::optim::patternsearch) fn select(
    fields: &ScalarOptionFields<'_>,
    n: usize,
) -> BuiltinResult<Selection> {
    let selected = algorithm(fields)?;
    let polling = if selected == OptimPatternAlgorithm::Classic {
        let method = OptimPatternPoll::parse(&text(fields, Field::PollMethod)?)
            .ok_or_else(|| invalid("unrecognized PollMethod"))?;
        let order = if method.mesh_adaptive() {
            OptimPatternOrder::Consecutive
        } else {
            OptimPatternOrder::parse(&text(fields, Field::PollOrderAlgorithm)?)
                .ok_or_else(|| invalid("unrecognized PollOrderAlgorithm"))?
        };
        Polling::Classic(Classic { method, order })
    } else {
        Polling::Nonuniform(selected)
    };
    let search = fields
        .exact(Field::SearchFcn)
        .map(|value| super::super::search::parse(value, n))
        .transpose()?;
    if selected != OptimPatternAlgorithm::Classic
        && search.is_some_and(|search| search.poll().is_some())
    {
        return Err(unsupported(
            "NUPS does not accept a poll method as SearchFcn",
        ));
    }
    let display = OptimDisplay::parse(&text(fields, Field::Display)?)
        .filter(|display| runmat_builtins::OptimSolver::Patternsearch.accepts_display(*display))
        .ok_or_else(|| invalid("unrecognized Display"))?;
    Ok(Selection {
        polling,
        search,
        display,
    })
}
