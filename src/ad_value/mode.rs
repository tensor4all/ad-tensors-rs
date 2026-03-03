/// Automatic differentiation mode.
///
/// # Examples
///
/// ```rust
/// use ad_tensors_rs::AdMode;
///
/// assert_eq!(AdMode::Primal, AdMode::Primal);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdMode {
    /// Plain evaluation without derivative propagation.
    Primal,
    /// Forward-mode value carrying tangent information.
    Forward,
    /// Reverse-mode value carrying graph metadata.
    Reverse,
}

/// Opaque identifier of a reverse-mode graph node.
///
/// # Examples
///
/// ```rust
/// use ad_tensors_rs::NodeId;
///
/// let node = NodeId(7);
/// assert_eq!(node.0, 7);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

/// Opaque identifier of a tape instance.
///
/// # Examples
///
/// ```rust
/// use ad_tensors_rs::TapeId;
///
/// let tape = TapeId(2);
/// assert_eq!(tape.0, 2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TapeId(pub u64);
