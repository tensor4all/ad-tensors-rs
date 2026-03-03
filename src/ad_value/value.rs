use super::{AdMode, NodeId, TapeId};

/// Generic AD value that can wrap any user-defined payload type `T`.
///
/// This is the primary extension point of the crate.
///
/// # Examples
///
/// ```rust
/// use ad_tensors_rs::{AdMode, AdValue, NodeId, TapeId};
///
/// let primal = AdValue::primal(3.0_f64);
/// assert_eq!(primal.mode(), AdMode::Primal);
///
/// let dual = AdValue::forward(3.0_f64, 1.0_f64);
/// assert_eq!(dual.mode(), AdMode::Forward);
///
/// let tracked = AdValue::reverse(3.0_f64, NodeId(1), TapeId(9), None);
/// assert_eq!(tracked.mode(), AdMode::Reverse);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum AdValue<T> {
    /// Primal-only value.
    Primal(T),
    /// Forward-mode value and tangent.
    Forward { primal: T, tangent: T },
    /// Reverse-mode value with graph metadata.
    Reverse {
        primal: T,
        node: NodeId,
        tape: TapeId,
        tangent: Option<T>,
    },
}

impl<T> AdValue<T> {
    /// Creates a primal-only value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::AdValue;
    ///
    /// let x = AdValue::primal(2_i32);
    /// assert!(matches!(x, AdValue::Primal(2)));
    /// ```
    pub fn primal(value: T) -> Self {
        Self::Primal(value)
    }

    /// Creates a forward-mode value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::AdValue;
    ///
    /// let x = AdValue::forward(2.0_f64, 1.0_f64);
    /// assert!(matches!(x, AdValue::Forward { .. }));
    /// ```
    pub fn forward(primal: T, tangent: T) -> Self {
        Self::Forward { primal, tangent }
    }

    /// Creates a reverse-mode value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdValue, NodeId, TapeId};
    ///
    /// let x = AdValue::reverse(2.0_f64, NodeId(3), TapeId(5), Some(0.1));
    /// assert!(matches!(x, AdValue::Reverse { .. }));
    /// ```
    pub fn reverse(primal: T, node: NodeId, tape: TapeId, tangent: Option<T>) -> Self {
        Self::Reverse {
            primal,
            node,
            tape,
            tangent,
        }
    }

    /// Returns the AD mode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdMode, AdValue};
    ///
    /// let x = AdValue::forward(1.0_f64, 1.0_f64);
    /// assert_eq!(x.mode(), AdMode::Forward);
    /// ```
    pub fn mode(&self) -> AdMode {
        match self {
            Self::Primal(_) => AdMode::Primal,
            Self::Forward { .. } => AdMode::Forward,
            Self::Reverse { .. } => AdMode::Reverse,
        }
    }

    /// Returns a reference to the primal payload.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::AdValue;
    ///
    /// let x = AdValue::forward(10_i32, 1_i32);
    /// assert_eq!(x.primal_ref(), &10);
    /// ```
    pub fn primal_ref(&self) -> &T {
        match self {
            Self::Primal(value) => value,
            Self::Forward { primal, .. } => primal,
            Self::Reverse { primal, .. } => primal,
        }
    }

    /// Returns a mutable reference to the primal payload.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::AdValue;
    ///
    /// let mut x = AdValue::primal(1_i32);
    /// *x.primal_mut() = 7;
    /// assert_eq!(x.primal_ref(), &7);
    /// ```
    pub fn primal_mut(&mut self) -> &mut T {
        match self {
            Self::Primal(value) => value,
            Self::Forward { primal, .. } => primal,
            Self::Reverse { primal, .. } => primal,
        }
    }

    /// Returns a reference to tangent payload when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::AdValue;
    ///
    /// let x = AdValue::forward(2.0_f64, 3.0_f64);
    /// assert_eq!(x.tangent_ref(), Some(&3.0));
    /// ```
    pub fn tangent_ref(&self) -> Option<&T> {
        match self {
            Self::Primal(_) => None,
            Self::Forward { tangent, .. } => Some(tangent),
            Self::Reverse { tangent, .. } => tangent.as_ref(),
        }
    }

    /// Returns reverse-mode node id when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdValue, NodeId, TapeId};
    ///
    /// let x = AdValue::reverse(1.0_f64, NodeId(4), TapeId(6), None);
    /// assert_eq!(x.node_id(), Some(NodeId(4)));
    /// ```
    pub fn node_id(&self) -> Option<NodeId> {
        match self {
            Self::Reverse { node, .. } => Some(*node),
            _ => None,
        }
    }

    /// Returns reverse-mode tape id when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::{AdValue, NodeId, TapeId};
    ///
    /// let x = AdValue::reverse(1.0_f64, NodeId(4), TapeId(6), None);
    /// assert_eq!(x.tape_id(), Some(TapeId(6)));
    /// ```
    pub fn tape_id(&self) -> Option<TapeId> {
        match self {
            Self::Reverse { tape, .. } => Some(*tape),
            _ => None,
        }
    }

    /// Maps the payload type while preserving AD mode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ad_tensors_rs::AdValue;
    ///
    /// let x = AdValue::forward(2_i32, 3_i32);
    /// let y = x.map(|v| v as f64);
    /// assert_eq!(y.primal_ref(), &2.0_f64);
    /// assert_eq!(y.tangent_ref(), Some(&3.0_f64));
    /// ```
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> AdValue<U> {
        match self {
            Self::Primal(value) => AdValue::Primal(f(value)),
            Self::Forward { primal, tangent } => AdValue::Forward {
                primal: f(primal),
                tangent: f(tangent),
            },
            Self::Reverse {
                primal,
                node,
                tape,
                tangent,
            } => AdValue::Reverse {
                primal: f(primal),
                node,
                tape,
                tangent: tangent.map(f),
            },
        }
    }
}

impl<T> From<T> for AdValue<T> {
    fn from(value: T) -> Self {
        Self::Primal(value)
    }
}
