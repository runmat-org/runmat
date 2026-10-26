//! Immutable GPS/GSS cone or frozen LTMADS lattice for an ordered poll.
use super::super::super::{
    cone::Cone,
    evaluation::Point,
    invalid,
    linear::{Linear, Prefix, ProposalStop},
    mads,
    mesh::{self, Basis, Mesh},
    scaling::Scaling,
};
use super::super::Acceptance;
use super::{DirectionId, History};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use runmat_builtins::OptimPatternPoll;
enum Geometry {
    Cone(Option<Cone>),
    Mads {
        model: mads::FrameModel,
        binding: Option<Cone>,
    },
}
pub(in crate::builtins::math::optim::patternsearch::poll) struct ClassicFrame<'a> {
    geometry: Geometry,
    mesh: &'a Mesh,
    step: f64,
    scaling: &'a Scaling,
    dimension: usize,
    basis: Basis,
    acceptance: Acceptance,
}
impl<'a> ClassicFrame<'a> {
    pub async fn new(
        base: &Point,
        geometry: &Linear,
        mesh: &'a Mesh,
        scaling: &'a Scaling,
        method: OptimPatternPoll,
        binding: f64,
        acceptance: Acceptance,
        refinements: &mut Option<mads::Refinements>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        let basis = if method.minimal() {
            Basis::Minimal
        } else {
            Basis::Maximal
        };
        let step = mesh.step_for(method);
        let source = if method.mesh_adaptive() {
            if geometry.equalities > 0 {
                return Err(invalid("MADS does not support linear equality constraints"));
            }
            if refinements.is_none() {
                *refinements = Some(mads::Refinements::new(base.x.len())?);
            }
            let state = refinements.as_mut().expect("MADS refinement state");
            if state.dimension() != base.x.len() {
                return Err(invalid("MADS refinement dimension changed"));
            }
            let Some(model) = mads::selected_frame(
                state,
                step,
                basis,
                matches!(
                    method,
                    OptimPatternPoll::OrthoMadsMaximal | OptimPatternPoll::OrthoMadsMinimal
                ),
                cooperation,
            )
            .await?
            else {
                return Ok(None);
            };
            let binding = if geometry.matrix.m > 0 {
                let radius = step.sqrt()
                    * if basis == Basis::Minimal {
                        base.x.len() as f64
                    } else {
                        1.0
                    };
                let Some((cone, active)) = geometry
                    .binding_cone(&base.x, radius, binding, scaling, cooperation)
                    .await?
                else {
                    return Ok(None);
                };
                active.then_some(cone)
            } else {
                None
            };
            Geometry::Mads { model, binding }
        } else {
            let cone = if geometry.matrix.m > 0
                && (geometry.equalities > 0 || !geometry.axis_aligned() || basis == Basis::Minimal)
            {
                let Some(cone) = geometry
                    .cone(&base.x, step, binding, scaling, cooperation)
                    .await?
                else {
                    return Ok(None);
                };
                Some(cone)
            } else {
                None
            };
            Geometry::Cone(cone)
        };
        Ok(Some(Self {
            geometry: source,
            mesh,
            step,
            scaling,
            dimension: base.x.len(),
            basis,
            acceptance,
        }))
    }
    pub fn count(&self) -> BuiltinResult<usize> {
        match &self.geometry {
            Geometry::Mads { model, binding } => {
                let extra = binding
                    .as_ref()
                    .map(|cone| cone.len(Basis::Maximal))
                    .transpose()?
                    .unwrap_or(0);
                model
                    .count()?
                    .checked_add(extra)
                    .ok_or_else(|| invalid("MADS binding poll count overflow"))
            }
            Geometry::Cone(Some(cone)) => cone.len(self.basis),
            Geometry::Cone(None) => match self.basis {
                Basis::Maximal => self.dimension.checked_mul(2),
                Basis::Minimal => self.dimension.checked_add(1),
            }
            .ok_or_else(|| invalid("poll count overflow")),
        }
    }
    pub fn rotatable(&self) -> bool {
        match &self.geometry {
            Geometry::Mads { .. } => false,
            Geometry::Cone(cone) => cone.as_ref().is_none_or(|cone| cone.rotatable()),
        }
    }
    pub fn measure(&self) -> f64 {
        match &self.geometry {
            Geometry::Mads { .. } => {
                self.step.sqrt()
                    * if self.basis == Basis::Minimal {
                        self.dimension as f64
                    } else {
                        1.0
                    }
            }
            Geometry::Cone(_) => self.step,
        }
    }
    pub async fn candidate(
        &self,
        geometry: &Linear,
        base: &[f64],
        ray: &[f64],
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Result<Prefix, ProposalStop>> {
        match &self.geometry {
            Geometry::Mads { .. } => {
                geometry
                    .mesh_candidate(base, ray, self.step, tolerance, cooperation)
                    .await
            }
            Geometry::Cone(_) => {
                geometry
                    .candidate_with_step(base, ray, self.step, tolerance, cooperation)
                    .await
            }
        }
    }
    pub async fn direction(
        &self,
        index: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        let mut direction = match &self.geometry {
            Geometry::Mads { model, binding } => {
                let count = model.count()?;
                if index < count {
                    let Some(ray) = model.direction(index, cooperation).await? else {
                        return Ok(None);
                    };
                    ray
                } else {
                    let Some(cone) = binding else {
                        return Err(invalid("MADS binding direction absent"));
                    };
                    let Some(mut ray) = cone
                        .oriented_direction(index - count, Basis::Maximal, false, cooperation)
                        .await?
                    else {
                        return Ok(None);
                    };
                    let maximum = ray
                        .iter()
                        .fold(0.0f64, |maximum, value| maximum.max(value.abs()));
                    let Some(bound) = model.bound() else {
                        return Ok(None);
                    };
                    if maximum == 0.0 || !maximum.is_finite() {
                        return Ok(None);
                    }
                    for value in &mut ray {
                        *value = (*value / maximum * bound).round();
                        cooperation.completed(1).await?;
                    }
                    ray
                }
            }
            Geometry::Cone(Some(cone)) => {
                let Some(ray) = cone
                    .oriented_direction(index, self.basis, self.mesh.sign() < 0.0, cooperation)
                    .await?
                else {
                    return Ok(None);
                };
                ray
            }
            Geometry::Cone(None) => {
                mesh::direction(self.basis, self.dimension, index, self.mesh.sign())?
            }
        };
        if matches!(self.acceptance, Acceptance::SufficientDecrease) {
            let mut norm = 0.0f64;
            for &value in &direction {
                norm = norm.hypot(value);
                cooperation.completed(1).await?;
            }
            if norm == 0.0 || !norm.is_finite() {
                return Ok(None);
            }
            for value in &mut direction {
                let normalized = *value / norm;
                if *value != 0.0 && normalized == 0.0 {
                    return Ok(None);
                }
                *value = normalized;
                cooperation.completed(1).await?;
            }
        }
        if !self.scaling.apply(&mut direction, cooperation).await? {
            return Ok(None);
        }
        Ok(Some(direction))
    }
    pub async fn history(
        &self,
        id: DirectionId,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<History> {
        match id {
            DirectionId::Previous => Ok(History::Previous),
            DirectionId::Generated(index) => Ok(History::New(
                self.direction(index, cooperation)
                    .await?
                    .ok_or_else(|| invalid("accepted poll direction could not be retained"))?,
            )),
        }
    }
}
