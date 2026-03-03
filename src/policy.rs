/// Differentiation policy for non-smooth branching operations.
///
/// # Examples
///
/// ```rust
/// use ad_tensors_rs::DiffPolicy;
///
/// let policy = DiffPolicy::StopGradient;
/// assert_eq!(policy, DiffPolicy::StopGradient);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DiffPolicy {
    /// Return an explicit mode-not-supported error.
    Strict,
    /// Allow primal evaluation and block derivative flow.
    #[default]
    StopGradient,
}
