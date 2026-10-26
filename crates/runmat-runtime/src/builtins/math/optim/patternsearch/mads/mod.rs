//! LTMADS lattice construction is separate from GPS cone geometry.
mod certificate;
mod generator;
mod level;
mod orthogonal;
mod model;
mod refinements;
mod sampling;
#[cfg(test)]
mod tests;
use super::mesh::Basis;
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
pub(super) use generator::Generator;
pub(super) use model::FrameModel;
pub(super) use refinements::Refinements;
pub(super) async fn frame(
    state: &mut Refinements,
    mesh: f64,
    basis: Basis,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Generator>> {
    let level = level::Level::for_mesh(mesh)?;
    Generator::new(state, level, basis, cooperation).await
}

pub(super) async fn selected_frame(
    state: &mut Refinements,
    mesh: f64,
    basis: Basis,
    orthogonal: bool,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<FrameModel>> {
    if !orthogonal {
        return Ok(frame(state, mesh, basis, cooperation)
            .await?
            .map(FrameModel::Triangular));
    }
    let level = level::Level::for_mesh(mesh)?;
    if state.orthogonal.is_none() {
        state.orthogonal = Some(orthogonal::Sequence::new(state.dimension(), cooperation).await?);
    }
    Ok(orthogonal::Householder::new(
        state.orthogonal.as_mut().expect("OrthoMADS sequence"),
        level,
        basis,
        cooperation,
    )
    .await?
    .map(FrameModel::Orthogonal))
}
