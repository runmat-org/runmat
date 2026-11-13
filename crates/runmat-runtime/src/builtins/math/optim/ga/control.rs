//! Shared admitted storage retains genetic control phases unchanged.
pub(super) use super::super::common::retained_control::Retained;
pub(super) const POLICY: super::super::common::retained_control::Policy =
    super::super::common::retained_control::Policy {
        identifier: "RunMat:ga:SearchAllocation",
        operation: "genetic search state allocation",
    };
