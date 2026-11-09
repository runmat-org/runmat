//! Authored before-step physical data. No solve, pressure repair or divergence projection.
mod projection;
mod values;
use super::{CanonicalFluidError as Error, CanonicalFluidFields, CanonicalFluidProblem};
use runmat_analysis_core::{AnalysisModel, InitialCondition, ProcedureStep};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalFluidInitialState {
    #[serde(with = "runmat_types::physical::canonical_storage")]
    velocity_m_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Velocity, Vec<[f64; 3]>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pressure_reference_pa:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Pressure, Vec<f64>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pressure_fluctuation_pa:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Pressure, Vec<f64>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    face_volume_flux_m3_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::VolumeFlowRate, Vec<f64>>,
    applied_condition_ids: Vec<String>,
    default_velocity_nodes: usize,
}
impl CanonicalFluidInitialState {
    pub fn original_fields(&self) -> CanonicalFluidFields<'_> {
        CanonicalFluidFields {
            velocity_m_per_s: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Velocity,
                _,
            >::new(self.velocity_m_per_s.canonical_values()),
            pressure_reference_pa: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Pressure,
                _,
            >::new(self.pressure_reference_pa.canonical_values()),
            pressure_fluctuation_pa: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Pressure,
                _,
            >::new(
                self.pressure_fluctuation_pa.canonical_values()
            ),
            face_volume_flux_m3_per_s: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::VolumeFlowRate,
                _,
            >::new(
                self.face_volume_flux_m3_per_s.canonical_values()
            ),
        }
    }
    pub fn applied_condition_ids(&self) -> &[String] {
        &self.applied_condition_ids
    }
    pub fn default_velocity_nodes(&self) -> usize {
        self.default_velocity_nodes
    }
}
impl CanonicalFluidProblem<'_> {
    pub(crate) fn realize_initial_state(
        &self,
        model: &AnalysisModel,
        step: &ProcedureStep,
        fields: &[crate::execution::InitialFieldBinding],
    ) -> Result<CanonicalFluidInitialState, Error> {
        if crate::is_fea_cancelled() {
            return Err(Error::Cancelled);
        }
        let profile = runmat_canonical_codec::profile_serialized(
            &(model, fields),
            super::INPUT_IDENTITY_LIMITS,
        )
        .map_err(|e| match e.kind {
            runmat_canonical_codec::CanonicalCodecErrorKind::ResourceLimit => {
                Error::Resource("original initial source profile")
            }
            runmat_canonical_codec::CanonicalCodecErrorKind::Invalid => {
                Error::InputRealization(e.to_string())
            }
        })?;
        let extra = (self.source.nodes.len() as u64)
            .checked_mul(512)
            .and_then(|v| v.checked_add((self.topology.cells.len() as u64).checked_mul(128)?))
            .and_then(|v| v.checked_add((self.topology.faces.len() as u64).checked_mul(64)?))
            .and_then(|v| v.checked_add(profile.materialization_resident_bytes_estimate))
            .ok_or(Error::Resource("initial field byte overflow"))?;
        if self
            .reservation
            .resident_bytes_upper_bound
            .checked_add(extra)
            .is_none_or(|v| v > self.policy.resident_byte_budget() || v > isize::MAX as u64)
        {
            return Err(Error::Resource("original initial field residency"));
        }
        let runmat_analysis_core::IndependentDomain::TimeSeconds { start, .. } =
            step.independent_domain
        else {
            return Err(Error::Invalid("fluid initial state needs physical start"));
        };
        if !matches!(self.sample, crate::excitation::ProjectionSample::TimeSeconds {value, ..} if value.to_bits()==start.to_bits())
        {
            return Err(Error::Invalid(
                "initial source must be realized at authored start",
            ));
        }
        crate::execution::validate_initial_field_selection(
            model,
            step,
            runmat_execution::Digest::from_bytes(*self.source.mesh_digest.bytes()),
            self.source.nodes.len(),
            fields,
        )
        .map_err(|e| Error::InputRealization(e.to_string()))?;
        let mut velocities = vec![None; self.source.nodes.len()];
        let mut pressure = vec![None; self.source.nodes.len()];
        let mut applied_condition_ids = Vec::new();
        for condition in model
            .initial_conditions
            .iter()
            .filter(|c| c.applies_to_step(step))
        {
            let target = match condition {
                InitialCondition::FluidVelocity(c) => &c.target,
                InitialCondition::FluidPressure(c) => &c.target,
                _ => {
                    return Err(Error::Unsupported(
                        crate::admission::FLUID_INITIAL_CONDITION.message,
                    ))
                }
            };
            let nodes = crate::excitation::initial_target_nodes(self.source, target)
                .map_err(|e| Error::InputRealization(format!("original initial target {e:?}")))?;
            match condition {
                InitialCondition::FluidVelocity(c) => {
                    let value = values::velocity(model, c, condition, self.source, fields)?;
                    for node in nodes {
                        let v = value.at(node)?;
                        if velocities[node].is_some_and(|old| old != v) {
                            return Err(Error::Invalid("conflicting original initial velocity"));
                        }
                        velocities[node] = Some(v);
                    }
                }
                InitialCondition::FluidPressure(c) => {
                    let value = values::pressure(model, c, condition, self.source, fields)?;
                    for node in nodes {
                        let v = value.at(node)?;
                        if pressure[node].is_some_and(|old| old != v) {
                            return Err(Error::Invalid("conflicting original initial pressure"));
                        }
                        pressure[node] = Some(v);
                    }
                }
                _ => unreachable!(),
            }
            applied_condition_ids.push(condition.initial_condition_id().to_owned());
        }
        let default_velocity_nodes = velocities.iter().filter(|v| v.is_none()).count();
        let velocity = velocities
            .into_iter()
            .map(|v| v.unwrap_or([0.; 3]))
            .collect::<Vec<_>>();
        let projection::ProjectedInitialFields {
            velocity_m_per_s,
            pressure_fluctuation_pa,
            face_volume_flux_m3_per_s,
        } = projection::integrate(self, &velocity, &pressure)?;
        self.pressure_reference_plan
            .validate(self, pressure_fluctuation_pa.canonical_values())?;
        let continuity =
            super::continuity::incidence(self, face_volume_flux_m3_per_s.canonical_values())?;
        if continuity
            .relative
            .iter()
            .any(|r| *r > self.policy.physical_relative_tolerance || !r.is_finite())
        {
            return Err(Error::Invalid(
                "authored initial field is not incompressible",
            ));
        }
        Ok(CanonicalFluidInitialState {
            velocity_m_per_s,
            pressure_reference_pa: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Pressure,
                _,
            >::new(
                self.pressure_reference_pa.canonical_values().clone()
            ),
            pressure_fluctuation_pa,
            face_volume_flux_m3_per_s,
            applied_condition_ids,
            default_velocity_nodes,
        })
    }
}
