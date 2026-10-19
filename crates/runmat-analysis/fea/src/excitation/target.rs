use runmat_analysis_core::ExcitationTarget;

use super::model::RealizedExcitationTarget;

pub(super) fn realize_target(target: &ExcitationTarget) -> RealizedExcitationTarget {
    match target {
        ExcitationTarget::Region(target) => RealizedExcitationTarget::Region(target.clone()),
        ExcitationTarget::ReferencePoint(point) => RealizedExcitationTarget::ReferencePoint {
            reference_point_id: point.reference_point_id.clone(),
            position: point.position,
        },
        ExcitationTarget::Port { port_id } => RealizedExcitationTarget::Port(port_id.clone()),
        ExcitationTarget::Interface { interface_id } => {
            RealizedExcitationTarget::Interface(interface_id.clone())
        }
        ExcitationTarget::WholeModel => RealizedExcitationTarget::WholeModel,
    }
}
