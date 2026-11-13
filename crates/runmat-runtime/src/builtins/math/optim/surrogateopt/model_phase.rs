//! Actual construction-to-search model transition at a measured batch boundary.
//! Only selected row IDs and coefficients survive: trial payloads stay in
//! Campaign, and a late append cannot change the frozen surrogate implicitly.
use super::{
    campaign::Campaign,
    domain::BoxView,
    history::{Snapshot, SnapshotView},
    invalid,
    merit::{self, Proposals, Selection, SelectionControls},
    rbf::{self, Fit, Model},
};
use crate::builtins::math::optim::common::retained_control::Retained;
use crate::{builtins::math::optim::common::ScalarCooperation, BuiltinResult};

pub(super) struct Frozen {
    model: Model,
    training: Snapshot,
    evaluated: Snapshot,
}
pub(super) enum Refresh {
    Ready(Retained<Frozen>),
    NeedMoreSites { sites: usize },
    Undefined,
    NoFunctions,
}
pub(super) struct Views<'a, 'domain> {
    pub training: SnapshotView<'a, 'domain>,
    pub evaluated: SnapshotView<'a, 'domain>,
}
pub(super) async fn refresh(
    campaign: &Campaign,
    domain: &BoxView<'_>,
    axes: &[usize],
    minimum: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Refresh> {
    if campaign.has_pending() {
        return Err(invalid(
            "surrogate refresh precedes its measured batch commit",
        ));
    }
    let rows = campaign.model_rows(cooperation).await?;
    if rows.len() < minimum {
        return Ok(Refresh::NeedMoreSites { sites: rows.len() });
    }
    let training = Snapshot::capture(campaign.history(), rows, cooperation).await?;
    let evaluated = Snapshot::capture(
        campaign.history(),
        campaign.history().evaluated_rows(cooperation).await?,
        cooperation,
    )
    .await?;
    let model = {
        let view = training
            .view(campaign.history(), axes, domain, cooperation)
            .await?;
        use rbf::Training;
        let sites = view.rows().len();
        match rbf::fit(&view, cooperation).await? {
            Fit::Poised(model) => model,
            Fit::Unpoised => return Ok(Refresh::NeedMoreSites { sites }),
            Fit::Undefined => return Ok(Refresh::Undefined),
            Fit::NoFunctions => return Ok(Refresh::NoFunctions),
        }
    };
    Ok(Refresh::Ready(Retained::new(
        Frozen {
            model,
            training,
            evaluated,
        },
        cooperation,
        super::CONTROL_STORAGE,
    )?))
}
impl Frozen {
    pub(super) fn model(&self) -> &Model {
        &self.model
    }
    pub(super) async fn views<'a, 'domain>(
        &'a self,
        campaign: &'a Campaign,
        domain: &'a BoxView<'domain>,
        axes: &'a [usize],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Views<'a, 'domain>> {
        Ok(Views {
            training: self
                .training
                .view(campaign.history(), axes, domain, cooperation)
                .await?,
            evaluated: self
                .evaluated_view(campaign, domain, axes, cooperation)
                .await?,
        })
    }
    pub(super) async fn evaluated_view<'a, 'domain>(
        &'a self,
        campaign: &'a Campaign,
        domain: &'a BoxView<'domain>,
        axes: &'a [usize],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<SnapshotView<'a, 'domain>> {
        self.evaluated
            .view(campaign.history(), axes, domain, cooperation)
            .await
    }
    pub(super) async fn select(
        &self,
        campaign: &Campaign,
        domain: &BoxView<'_>,
        axes: &[usize],
        proposals: &impl Proposals,
        controls: SelectionControls,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Selection> {
        let views = self.views(campaign, domain, axes, cooperation).await?;
        merit::select(
            &self.model,
            &views.training,
            &views.evaluated,
            proposals,
            controls,
            cooperation,
        )
        .await
    }
}
