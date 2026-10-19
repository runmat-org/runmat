//! Prepared publication borrows the opaque accepted owner and reuses its tables.
use super::super::indexed_face::surface::SurfaceEncoding;
use super::*;
use runmat_canonical_codec::CanonicalEncodingProfile;
pub struct PreparedExactSurfacePassEncoding<'a> {
    kind: Encoding<'a>,
}
enum Encoding<'a> {
    Surface(Box<SurfaceEncoding<'a>>),
    Restart(&'a ExactSurfacePassResult),
}
pub(super) fn preparation_bytes(
    pass: &AcceptedExactSurfacePass,
    signal: &dyn MeshingCancellationSignal,
) -> Result<u64, super::super::ExactSurfaceMeshError> {
    match &pass.result().outcome {
        ExactSurfacePassOutcome::Converged { surface } => {
            SurfaceEncoding::preparation_storage_bytes(surface, signal)?
                .checked_add(std::mem::size_of::<SurfaceEncoding<'_>>() as u64)
                .ok_or_else(|| {
                    super::super::ExactSurfaceMeshError::new(
                        ExactSurfaceMeshErrorKind::ResourceLimit,
                        "prepared global encoder inventory overflow",
                    )
                })
        }
        ExactSurfacePassOutcome::RequiresCurveSplits { .. } => Ok(0),
    }
}
pub(super) fn prepare<'a>(
    pass: &'a AcceptedExactSurfacePass,
    signal: &dyn MeshingCancellationSignal,
) -> Result<PreparedExactSurfacePassEncoding<'a>, super::super::ExactSurfaceMeshError> {
    Ok(PreparedExactSurfacePassEncoding {
        kind: match &pass.result().outcome {
            ExactSurfacePassOutcome::Converged { surface } => {
                Encoding::Surface(Box::new(SurfaceEncoding::new(surface, signal)?))
            }
            ExactSurfacePassOutcome::RequiresCurveSplits { .. } => Encoding::Restart(pass.result()),
        },
    })
}
impl PreparedExactSurfacePassEncoding<'_> {
    pub fn encoding_profile(
        &self,
        signal: &dyn MeshingCancellationSignal,
    ) -> Result<CanonicalEncodingProfile, super::super::ExactSurfaceMeshError> {
        match &self.kind {
            Encoding::Surface(surface) => surface.profile(signal),
            Encoding::Restart(pass) => {
                super::super::pass_result::profile_decided_pass(pass, signal)
            }
        }
    }
    pub fn encode_profiled(
        &self,
        profile: &CanonicalEncodingProfile,
        signal: &dyn MeshingCancellationSignal,
    ) -> Result<Vec<u8>, super::super::ExactSurfaceMeshError> {
        match &self.kind {
            Encoding::Surface(surface)=>surface.encode_profiled(profile,signal),
            Encoding::Restart(pass)=>super::super::pass_result::encode_decided_exact_surface_pass_result_with_cancellation(pass,signal),
        }
    }
}
