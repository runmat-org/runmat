use super::*;
use crate::excitation::{
    realize_step_excitations, ExcitationSamplePoint, RealizedExcitationKind,
    RealizedExcitationTarget,
};
use crate::physics::maxwell::{ComplexPhasor, MaxwellCurrentSample};
use std::collections::BTreeMap;

pub(super) fn compile(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    step: &ProcedureStep,
    frequency_hz: f64,
    sampling: &SampledVolumeFieldTopology,
) -> Result<Vec<MaxwellCurrentSample>, MaxwellError> {
    // Reject external/generated/time-domain input before shared realization.
    // Accepted complex tables are evaluated once at the exact authored Hz value.
    for action in &model.excitations {
        let id = &action.excitation().signal_id;
        let signal = model.signals.iter().find(|signal| signal.signal_id() == id);
        if !matches!(signal, Some(PhysicalSignal::ElectricCurrentDensity(signal))
            if signal.frame == Some(CoordinateFrame::Global) && signal.shape == SignalShape::Vector3
                && (matches!(signal.representation, SignalRepresentation::Constant(_))
                    || matches!(&signal.representation, SignalRepresentation::Tabulated(table)
                        if table.domain == TabulatedSignalDomain::FrequencyHertz)))
        {
            return Err(MaxwellError::Unsupported(
                "explicit constant or frequency-tabulated global vector current required".into(),
            ));
        }
    }
    let raw: BTreeMap<_, _> = mesh
        .topology
        .volume_elements
        .iter()
        .map(|element| (element.stable_identity, element))
        .collect();
    let mut samples: Vec<_> = sampling
        .integration_samples()
        .iter()
        .map(|point| MaxwellCurrentSample {
            sample_identity: point.identity,
            current_density_a_per_m2: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::ElectricCurrentDensity,
                _,
            >::new([ComplexPhasor::ZERO; 3]),
        })
        .collect();
    let loads =
        realize_step_excitations(model, step, ExcitationSamplePoint::frequency(frequency_hz))
            .map_err(invalid)?;
    for load in loads.iter() {
        let RealizedExcitationKind::MagneticVolumeCurrentDensity(value) = load.kind else {
            return Err(MaxwellError::Unsupported(
                "non-volume current realization".into(),
            ));
        };
        if load.policy.frame != CoordinateFrame::Global
            || load.policy.kinematics != ExcitationKinematics::FixedFrame
            || load.policy.distribution != ExcitationDistribution::IntensityPerVolume
            || load.policy.projection.preserve_moment
            || load.policy.projection.preserve_resultant
        {
            return Err(MaxwellError::Unsupported(
                "fixed global volume-current density required".into(),
            ));
        }
        let mut matched = false;
        for (sample, point) in samples.iter_mut().zip(sampling.integration_samples()) {
            let selected = match &load.target {
                RealizedExcitationTarget::WholeModel => true,
                RealizedExcitationTarget::Region(target) => {
                    target.validate_resolution().map_err(invalid)?;
                    let element = raw
                        .get(&point.element_identity)
                        .ok_or_else(|| topology("foreign source element"))?;
                    target.entity_dimension == EntityDimension::Volume
                        && (target.matches_persistent_entity(&element.region_id)
                            || element
                                .provenance
                                .iter()
                                .any(|id| target.matches_persistent_entity(id)))
                }
                _ => false,
            };
            if selected {
                matched = true;
                sample.current_density_a_per_m2 = sample.current_density_a_per_m2
                    .try_map_canonical_storage(|mut components| {
                        for (axis, current) in components.iter_mut().enumerate() {
                            *current += ComplexPhasor::new(value.real[axis], value.imaginary[axis]);
                        }
                        components
                    })
                    .map_err(invalid)?;
            }
        }
        if !matched {
            return Err(invalid(
                "impressed current target matches no canonical volume",
            ));
        }
    }
    Ok(samples)
}
