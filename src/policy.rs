/// Differentiation behavior for non-smooth branching operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DiffPolicy {
    /// Return an explicit mode-not-supported error.
    Strict,
    /// Allow primal evaluation and block derivative flow.
    #[default]
    StopGradient,
}
