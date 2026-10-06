#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RequestedOutputToken(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PresentationOutputToken(u64);

#[derive(Clone, Debug, Default)]
struct ChannelProjection<T> {
    inherited: Vec<T>,
    overlays: Vec<(u64, T)>,
    next_token: u64,
}

impl<T: Copy> ChannelProjection<T> {
    const fn new() -> Self {
        Self {
            inherited: Vec::new(),
            overlays: Vec::new(),
            next_token: 0,
        }
    }

    fn snapshot(&self) -> Self {
        let mut inherited = self.inherited.clone();
        inherited.extend(self.overlays.iter().map(|(_, value)| *value));
        Self {
            inherited,
            overlays: Vec::new(),
            next_token: 0,
        }
    }

    fn push(&mut self, value: T) -> u64 {
        let token = self.next_token;
        self.next_token = self
            .next_token
            .checked_add(1)
            .expect("output projection token space exhausted");
        self.overlays.push((token, value));
        token
    }

    fn remove(&mut self, token: u64) {
        if let Some(index) = self
            .overlays
            .iter()
            .position(|(candidate, _)| *candidate == token)
        {
            self.overlays.remove(index);
        }
    }

    fn current(&self) -> Option<T> {
        self.overlays
            .last()
            .map(|(_, value)| *value)
            .or_else(|| self.inherited.last().copied())
    }
}

/// Output semantics projected into one admitted invocation. Requested output
/// cardinality controls builtin behavior; presentation cardinality controls
/// how an already-produced value is exposed. They intentionally remain
/// separate typed channels.
#[derive(Clone, Debug, Default)]
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

    pub(crate) fn push_requested(&mut self, value: Option<usize>) -> RequestedOutputToken {
        RequestedOutputToken(self.requested.push(value))
    }

    pub(crate) fn remove_requested(&mut self, token: RequestedOutputToken) {
        self.requested.remove(token.0);
    }

    pub(crate) fn current_requested(&self) -> Option<Option<usize>> {
        self.requested.current()
    }

    pub(crate) fn push_presentation(&mut self, value: usize) -> PresentationOutputToken {
        PresentationOutputToken(self.presentation.push(value))
    }

    pub(crate) fn remove_presentation(&mut self, token: PresentationOutputToken) {
        self.presentation.remove(token.0);
    }

    pub(crate) fn current_presentation(&self) -> Option<usize> {
        self.presentation.current()
    }
}
