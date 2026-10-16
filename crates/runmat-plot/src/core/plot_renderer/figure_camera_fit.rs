/// Figure-only viewport fit ownership. Geometry scenes retain their separate camera policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum FigureCameraFitState {
    #[default]
    Inactive,
    Auto {
        last_viewport: Option<(u32, u32)>,
    },
}

impl FigureCameraFitState {
    pub(super) fn pending_for(self, viewport: (u32, u32)) -> bool {
        matches!(self, Self::Auto { last_viewport } if last_viewport != Some(viewport))
    }

    pub(super) fn fitted(&mut self, viewport: (u32, u32)) {
        if let Self::Auto { last_viewport } = self {
            *last_viewport = Some(viewport);
        }
    }
}
