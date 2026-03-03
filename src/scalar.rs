use num_complex::Complex64;

use crate::mode::{Dual, NodeId, Tracked};

/// Base scalar domain used by the AD wrappers.
///
/// # Examples
///
/// ```rust
/// use ad_tensors_rs::BaseScalar;
/// use num_complex::Complex64;
///
/// let a = BaseScalar::F64(1.5);
/// let b = BaseScalar::C64(Complex64::new(2.0, -0.5));
/// assert!(matches!(a, BaseScalar::F64(_)));
/// assert!(matches!(b, BaseScalar::C64(_)));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BaseScalar {
    F64(f64),
    C64(Complex64),
}

impl From<f64> for BaseScalar {
    fn from(value: f64) -> Self {
        Self::F64(value)
    }
}

impl From<Complex64> for BaseScalar {
    fn from(value: Complex64) -> Self {
        Self::C64(value)
    }
}

/// Runtime scalar preserving AD mode metadata.
///
/// # Examples
///
/// ```rust
/// use ad_tensors_rs::{AnyScalar, BaseScalar, NodeId};
///
/// let primal = AnyScalar::Primal(BaseScalar::F64(3.0));
/// let dual = AnyScalar::Dual {
///     primal: BaseScalar::F64(3.0),
///     tangent: BaseScalar::F64(1.0),
/// };
/// let tracked = AnyScalar::Tracked {
///     primal: BaseScalar::F64(3.0),
///     node: NodeId(42),
///     tangent: None,
/// };
///
/// assert!(matches!(primal, AnyScalar::Primal(_)));
/// assert!(matches!(dual, AnyScalar::Dual { .. }));
/// assert!(matches!(tracked, AnyScalar::Tracked { .. }));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum AnyScalar {
    Primal(BaseScalar),
    Dual {
        primal: BaseScalar,
        tangent: BaseScalar,
    },
    Tracked {
        primal: BaseScalar,
        node: NodeId,
        tangent: Option<BaseScalar>,
    },
}

impl<T> From<T> for AnyScalar
where
    T: Into<BaseScalar>,
{
    fn from(value: T) -> Self {
        Self::Primal(value.into())
    }
}

impl<T> From<Dual<T>> for AnyScalar
where
    T: Into<BaseScalar>,
{
    fn from(value: Dual<T>) -> Self {
        Self::Dual {
            primal: value.primal.into(),
            tangent: value.tangent.into(),
        }
    }
}

impl<T> From<Tracked<T>> for AnyScalar
where
    T: Into<BaseScalar>,
{
    fn from(value: Tracked<T>) -> Self {
        Self::Tracked {
            primal: value.primal.into(),
            node: value.node,
            tangent: value.tangent.map(Into::into),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_scalar_into() {
        let x: BaseScalar = 1.0_f64.into();
        let z: BaseScalar = Complex64::new(2.0, -0.5).into();
        assert!(matches!(x, BaseScalar::F64(_)));
        assert!(matches!(z, BaseScalar::C64(_)));
    }

    #[test]
    fn any_scalar_into() {
        let p0: AnyScalar = 2.0_f64.into();
        let p1: AnyScalar = Complex64::new(3.0, 1.0).into();
        assert!(matches!(p0, AnyScalar::Primal(BaseScalar::F64(_))));
        assert!(matches!(p1, AnyScalar::Primal(BaseScalar::C64(_))));

        let dual: AnyScalar = Dual {
            primal: BaseScalar::F64(1.0),
            tangent: BaseScalar::F64(0.1),
        }
        .into();
        assert!(matches!(dual, AnyScalar::Dual { .. }));

        let tracked: AnyScalar = Tracked {
            primal: BaseScalar::F64(1.0),
            node: NodeId(5),
            tape: crate::TapeId(6),
            tangent: Some(BaseScalar::F64(0.2)),
        }
        .into();
        assert!(matches!(tracked, AnyScalar::Tracked { .. }));
    }
}
