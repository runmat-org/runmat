//! The auxiliary genetic solve borrows the enclosing original merit epoch.
mod callbacks;
mod geometry;
mod oracle;
mod projection;
mod run;
pub(in crate::builtins::math::optim::patternsearch) use run::run;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Stop {
    Budget,
    ParentTime,
    LocalTime,
}
impl From<super::super::timing::AuxiliaryStop> for Stop {
    fn from(stop: super::super::timing::AuxiliaryStop) -> Self {
        match stop {
            super::super::timing::AuxiliaryStop::Parent => Self::ParentTime,
            super::super::timing::AuxiliaryStop::Local => Self::LocalTime,
        }
    }
}
