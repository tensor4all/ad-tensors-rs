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
/// let tape = TapeId(1);
/// assert_eq!(tape.0, 1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TapeId(pub u64);

/// Plain numeric value wrapper.
///
/// # Examples
///
/// ```rust
/// use ad_tensors_rs::Primal;
///
/// let x = Primal { value: 2.0_f64 };
/// assert_eq!(x.value, 2.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Primal<T> {
    pub value: T,
}

impl<T> From<T> for Primal<T> {
    fn from(value: T) -> Self {
        Self { value }
    }
}

/// Forward-mode dual value wrapper.
///
/// # Examples
///
/// ```rust
/// use ad_tensors_rs::Dual;
///
/// let x = Dual {
///     primal: 3.0_f64,
///     tangent: 1.0_f64,
/// };
/// assert_eq!(x.primal, 3.0);
/// assert_eq!(x.tangent, 1.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Dual<T> {
    pub primal: T,
    pub tangent: T,
}

impl<T> From<(T, T)> for Dual<T> {
    fn from(value: (T, T)) -> Self {
        Self {
            primal: value.0,
            tangent: value.1,
        }
    }
}

/// Reverse-mode tracked value wrapper.
///
/// # Examples
///
/// ```rust
/// use ad_tensors_rs::{NodeId, TapeId, Tracked};
///
/// let x = Tracked {
///     primal: 4.0_f64,
///     node: NodeId(10),
///     tape: TapeId(2),
///     tangent: Some(0.5),
/// };
/// assert_eq!(x.node.0, 10);
/// assert_eq!(x.tape.0, 2);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Tracked<T> {
    pub primal: T,
    pub node: NodeId,
    pub tape: TapeId,
    pub tangent: Option<T>,
}

impl<T> From<(T, NodeId, TapeId)> for Tracked<T> {
    fn from(value: (T, NodeId, TapeId)) -> Self {
        Self {
            primal: value.0,
            node: value.1,
            tape: value.2,
            tangent: None,
        }
    }
}

impl<T> From<(T, NodeId, TapeId, Option<T>)> for Tracked<T> {
    fn from(value: (T, NodeId, TapeId, Option<T>)) -> Self {
        Self {
            primal: value.0,
            node: value.1,
            tape: value.2,
            tangent: value.3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primal_into() {
        let p: Primal<f64> = 2.5_f64.into();
        assert_eq!(p.value, 2.5);
    }

    #[test]
    fn dual_into() {
        let d: Dual<f64> = (3.0_f64, 1.0_f64).into();
        assert_eq!(d.primal, 3.0);
        assert_eq!(d.tangent, 1.0);
    }

    #[test]
    fn tracked_into() {
        let t0: Tracked<f64> = (5.0_f64, NodeId(3), TapeId(2)).into();
        assert_eq!(t0.tangent, None);

        let t1: Tracked<f64> = (5.0_f64, NodeId(3), TapeId(2), Some(0.25)).into();
        assert_eq!(t1.tangent, Some(0.25));
    }
}
