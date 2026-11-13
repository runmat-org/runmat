//! Only the state vectors that a selected solver actually owns.

pub(super) enum SolverSlot<T> {
    Sgdm { velocity: Vec<T> },
    Rmsprop { squared: Vec<T> },
    Adam { mean: Vec<T>, squared: Vec<T> },
}
