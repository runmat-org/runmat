use super::host_allocation::{
    HostAllocationError, HostAllocationPolicy, HostAllocationReceipt, HostOutputPermit,
};
use super::overlay_storage::{
    AdmittedOverlayVec, CheckedOverlayGrowth, OverlayInstallError, OverlayPushSnapshot,
    OverlayRetirement,
};
use super::poll_projection::PollProjection;
use super::RuntimeContext;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RequestedOutputToken(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PresentationOutputToken(u64);

pub(crate) type RequestedOutputInstallResult = Result<
    (
        RequestedOutputToken,
        OverlayRetirement<(u64, Option<usize>)>,
    ),
    ChannelInstallError<Option<usize>>,
>;

pub(crate) type PresentationOutputInstallResult =
    Result<(PresentationOutputToken, OverlayRetirement<(u64, usize)>), ChannelInstallError<usize>>;

struct ChannelProjection<T> {
    inherited: Option<T>,
    overlays: AdmittedOverlayVec<(u64, T)>,
    next_token: u64,
}

impl<T: Copy> ChannelProjection<T> {
    const fn new() -> Self {
        Self {
            inherited: None,
            overlays: AdmittedOverlayVec::unbounded(),
            next_token: 0,
        }
    }

    fn snapshot(&self) -> Self {
        Self {
            inherited: self.current(),
            overlays: AdmittedOverlayVec::unbounded(),
            next_token: 0,
        }
    }

    fn admitted_snapshot(
        &self,
        context: &RuntimeContext,
        permit: HostOutputPermit,
    ) -> Result<Self, HostAllocationError> {
        Ok(Self {
            inherited: self.current(),
            overlays: AdmittedOverlayVec::managed_with_permit(context, permit)?,
            next_token: 0,
        })
    }

    fn snapshot_push(&self) -> Result<ChannelPushSnapshot, HostAllocationError> {
        let next_token = self
            .next_token
            .checked_add(1)
            .ok_or(HostAllocationError::Overflow)?;
        Ok(ChannelPushSnapshot {
            token: self.next_token,
            next_token,
            growth: self.overlays.snapshot_push()?,
        })
    }

    fn install_push(
        &mut self,
        prepared: PreparedChannelPush<T>,
    ) -> Result<OverlayRetirement<(u64, T)>, ChannelInstallError<T>> {
        if self.next_token != prepared.token {
            return Err(ChannelInstallError {
                cause: HostAllocationError::StaleAuthority,
                _value: prepared.value,
                _growth: prepared.growth,
            });
        }
        let token = prepared.token;
        match self
            .overlays
            .install_push(prepared.growth, (token, prepared.value))
        {
            Ok(retirement) => {
                self.next_token = prepared.next_token;
                Ok(retirement)
            }
            Err(OverlayInstallError {
                cause,
                value: (_, value),
                prepared,
            }) => Err(ChannelInstallError {
                cause,
                _value: value,
                _growth: prepared,
            }),
        }
    }

    fn push_unbounded(&mut self, value: T) -> Result<u64, HostAllocationError> {
        let next_token = self
            .next_token
            .checked_add(1)
            .ok_or(HostAllocationError::Overflow)?;
        let token = self.next_token;
        self.overlays
            .push_unbounded((token, value))
            .map_err(|_| HostAllocationError::StaleAuthority)?;
        self.next_token = next_token;
        Ok(token)
    }

    fn remove(&mut self, token: u64) -> Option<(u64, T)> {
        let index = self
            .overlays
            .as_slice()
            .iter()
            .position(|(candidate, _)| *candidate == token)?;
        self.overlays.retire(index)
    }

    fn current(&self) -> Option<T> {
        self.overlays
            .as_slice()
            .last()
            .map(|(_, value)| *value)
            .or(self.inherited)
    }
}

pub(crate) struct ChannelPushSnapshot {
    token: u64,
    next_token: u64,
    growth: OverlayPushSnapshot,
}

pub(crate) struct PreparedChannelPush<T> {
    token: u64,
    next_token: u64,
    value: T,
    growth: CheckedOverlayGrowth<(u64, T)>,
}

pub(crate) struct ChannelInstallError<T> {
    pub(crate) cause: HostAllocationError,
    _value: T,
    _growth: CheckedOverlayGrowth<(u64, T)>,
}

/// Output semantics projected into one admitted invocation. Requested output
/// cardinality controls builtin behavior; presentation cardinality controls
/// how an already-produced value is exposed. They intentionally remain
/// separate typed channels.
pub(crate) struct InvocationOutputProjection {
    requested: ChannelProjection<Option<usize>>,
    presentation: ChannelProjection<usize>,
}

impl InvocationOutputProjection {
    pub(crate) const fn new() -> Self {
        Self {
            requested: ChannelProjection::new(),
            presentation: ChannelProjection::new(),
        }
    }

    pub(crate) fn snapshot(&self) -> Self {
        Self {
            requested: self.requested.snapshot(),
            presentation: self.presentation.snapshot(),
        }
    }

    pub(crate) fn admitted_poll_snapshot(
        &self,
        context: &RuntimeContext,
        permit: HostOutputPermit,
    ) -> Result<Self, HostAllocationError> {
        Ok(Self {
            requested: self.requested.admitted_snapshot(context, permit.clone())?,
            presentation: self.presentation.admitted_snapshot(context, permit)?,
        })
    }

    pub(crate) fn requested_push_snapshot(
        &self,
    ) -> Result<ChannelPushSnapshot, HostAllocationError> {
        self.requested.snapshot_push()
    }

    pub(crate) fn ensure_requested_policy(
        owner: &PollProjection<Self>,
        context: &RuntimeContext,
    ) -> Result<(), HostAllocationError> {
        ensure_channel_policy(
            owner,
            context,
            |projection| &projection.requested,
            |projection| &mut projection.requested,
        )
    }

    pub(crate) fn push_requested_unbounded(
        &mut self,
        value: Option<usize>,
    ) -> Result<RequestedOutputToken, HostAllocationError> {
        self.requested
            .push_unbounded(value)
            .map(RequestedOutputToken)
    }

    pub(crate) fn prepare_requested(
        context: &RuntimeContext,
        snapshot: ChannelPushSnapshot,
        value: Option<usize>,
    ) -> Result<PreparedChannelPush<Option<usize>>, HostAllocationError> {
        prepare_channel_push(context, snapshot, value)
    }

    pub(crate) fn install_requested(
        &mut self,
        prepared: PreparedChannelPush<Option<usize>>,
    ) -> RequestedOutputInstallResult {
        let token = RequestedOutputToken(prepared.token);
        let retirement = self.requested.install_push(prepared)?;
        Ok((token, retirement))
    }

    pub(crate) fn remove_requested(
        &mut self,
        token: RequestedOutputToken,
    ) -> Option<(u64, Option<usize>)> {
        self.requested.remove(token.0)
    }

    pub(crate) fn current_requested(&self) -> Option<Option<usize>> {
        self.requested.current()
    }

    pub(crate) fn presentation_push_snapshot(
        &self,
    ) -> Result<ChannelPushSnapshot, HostAllocationError> {
        self.presentation.snapshot_push()
    }

    pub(crate) fn ensure_presentation_policy(
        owner: &PollProjection<Self>,
        context: &RuntimeContext,
    ) -> Result<(), HostAllocationError> {
        ensure_channel_policy(
            owner,
            context,
            |projection| &projection.presentation,
            |projection| &mut projection.presentation,
        )
    }

    pub(crate) fn push_presentation_unbounded(
        &mut self,
        value: usize,
    ) -> Result<PresentationOutputToken, HostAllocationError> {
        self.presentation
            .push_unbounded(value)
            .map(PresentationOutputToken)
    }

    pub(crate) fn prepare_presentation(
        context: &RuntimeContext,
        snapshot: ChannelPushSnapshot,
        value: usize,
    ) -> Result<PreparedChannelPush<usize>, HostAllocationError> {
        prepare_channel_push(context, snapshot, value)
    }

    pub(crate) fn install_presentation(
        &mut self,
        prepared: PreparedChannelPush<usize>,
    ) -> PresentationOutputInstallResult {
        let token = PresentationOutputToken(prepared.token);
        let retirement = self.presentation.install_push(prepared)?;
        Ok((token, retirement))
    }

    pub(crate) fn remove_presentation(
        &mut self,
        token: PresentationOutputToken,
    ) -> Option<(u64, usize)> {
        self.presentation.remove(token.0)
    }

    pub(crate) fn current_presentation(&self) -> Option<usize> {
        self.presentation.current()
    }
}

fn ensure_channel_policy<T: Copy>(
    owner: &PollProjection<InvocationOutputProjection>,
    context: &RuntimeContext,
    select: impl Fn(&InvocationOutputProjection) -> &ChannelProjection<T>,
    select_mut: impl Fn(&mut InvocationOutputProjection) -> &mut ChannelProjection<T>,
) -> Result<(), HostAllocationError> {
    if HostAllocationReceipt::policy(Some(context))? != HostAllocationPolicy::Managed {
        return Ok(());
    }
    let needs_owner = {
        let projection = owner.borrow();
        select(&projection).overlays.requires_managed_owner()?
    };
    if !needs_owner {
        return Ok(());
    }
    let prepared = AdmittedOverlayVec::prepare_managed_owner(context)?;
    let installed = {
        let mut projection = owner.borrow_mut();
        select_mut(&mut projection)
            .overlays
            .install_managed_owner(prepared)
    };
    match installed {
        Ok(retirement) => {
            drop(retirement);
            Ok(())
        }
        Err(failure) => {
            let cause = failure.cause;
            drop(failure);
            Err(cause)
        }
    }
}

impl Default for InvocationOutputProjection {
    fn default() -> Self {
        Self::new()
    }
}

fn prepare_channel_push<T>(
    context: &RuntimeContext,
    snapshot: ChannelPushSnapshot,
    value: T,
) -> Result<PreparedChannelPush<T>, HostAllocationError> {
    let ChannelPushSnapshot {
        token,
        next_token,
        growth,
    } = snapshot;
    let growth = AdmittedOverlayVec::<(u64, T)>::prepare_push(context, growth)?;
    let growth = growth.checkpoint(context).map_err(|(cause, _)| cause)?;
    Ok(PreparedChannelPush {
        token,
        next_token,
        value,
        growth,
    })
}
