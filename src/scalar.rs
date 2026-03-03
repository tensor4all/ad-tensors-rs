use num_complex::{Complex32, Complex64};

use crate::mode::{Dual, NodeId, Tracked};

/// Compile-time scalar boundary for values representable as [`BaseScalar`].
///
/// This keeps generic APIs ergonomic while preserving the runtime-mode
/// representation (`BaseScalar` / `AnyScalar`) used by the AD layer.
///
/// # Examples
///
/// ```rust
/// use ad_tensors_rs::{AnyScalar, BaseScalarLike};
/// use num_complex::{Complex32, Complex64};
///
/// fn lift<T: BaseScalarLike>(x: T) -> AnyScalar {
///     AnyScalar::from(x)
/// }
///
/// let a0 = lift(1.0_f32);
/// let a = lift(1.0_f64);
/// let b0 = lift(Complex32::new(1.0, -0.25));
/// let b = lift(Complex64::new(2.0, 0.5));
/// assert!(matches!(a0, AnyScalar::Primal(_)));
/// assert!(matches!(a, AnyScalar::Primal(_)));
/// assert!(matches!(b0, AnyScalar::Primal(_)));
/// assert!(matches!(b, AnyScalar::Primal(_)));
/// ```
pub trait BaseScalarLike: Clone {
    fn into_base_scalar(self) -> BaseScalar;
}

/// Base scalar domain used by the AD wrappers.
///
/// # Examples
///
/// ```rust
/// use ad_tensors_rs::BaseScalar;
/// use num_complex::{Complex32, Complex64};
///
/// let a0 = BaseScalar::F32(1.25);
/// let a = BaseScalar::F64(1.5);
/// let b0 = BaseScalar::C32(Complex32::new(0.5, 0.25));
/// let b = BaseScalar::C64(Complex64::new(2.0, -0.5));
/// assert!(matches!(a0, BaseScalar::F32(_)));
/// assert!(matches!(a, BaseScalar::F64(_)));
/// assert!(matches!(b0, BaseScalar::C32(_)));
/// assert!(matches!(b, BaseScalar::C64(_)));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BaseScalar {
    F32(f32),
    F64(f64),
    C32(Complex32),
    C64(Complex64),
}

impl BaseScalarLike for BaseScalar {
    fn into_base_scalar(self) -> BaseScalar {
        self
    }
}

impl From<f32> for BaseScalar {
    fn from(value: f32) -> Self {
        Self::F32(value)
    }
}

impl BaseScalarLike for f32 {
    fn into_base_scalar(self) -> BaseScalar {
        BaseScalar::F32(self)
    }
}

impl From<f64> for BaseScalar {
    fn from(value: f64) -> Self {
        Self::F64(value)
    }
}

impl BaseScalarLike for f64 {
    fn into_base_scalar(self) -> BaseScalar {
        BaseScalar::F64(self)
    }
}

impl From<Complex32> for BaseScalar {
    fn from(value: Complex32) -> Self {
        Self::C32(value)
    }
}

impl BaseScalarLike for Complex32 {
    fn into_base_scalar(self) -> BaseScalar {
        BaseScalar::C32(self)
    }
}

impl From<Complex64> for BaseScalar {
    fn from(value: Complex64) -> Self {
        Self::C64(value)
    }
}

impl BaseScalarLike for Complex64 {
    fn into_base_scalar(self) -> BaseScalar {
        BaseScalar::C64(self)
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
    T: BaseScalarLike,
{
    fn from(value: T) -> Self {
        Self::Primal(value.into_base_scalar())
    }
}

impl<T> From<Dual<T>> for AnyScalar
where
    T: BaseScalarLike,
{
    fn from(value: Dual<T>) -> Self {
        Self::Dual {
            primal: value.primal.into_base_scalar(),
            tangent: value.tangent.into_base_scalar(),
        }
    }
}

impl<T> From<Tracked<T>> for AnyScalar
where
    T: BaseScalarLike,
{
    fn from(value: Tracked<T>) -> Self {
        Self::Tracked {
            primal: value.primal.into_base_scalar(),
            node: value.node,
            tangent: value.tangent.map(BaseScalarLike::into_base_scalar),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_scalar_into() {
        let x0: BaseScalar = 1.0_f32.into();
        let x: BaseScalar = 1.0_f64.into();
        let z0: BaseScalar = Complex32::new(1.0, 0.5).into();
        let z: BaseScalar = Complex64::new(2.0, -0.5).into();
        assert!(matches!(x0, BaseScalar::F32(_)));
        assert!(matches!(x, BaseScalar::F64(_)));
        assert!(matches!(z0, BaseScalar::C32(_)));
        assert!(matches!(z, BaseScalar::C64(_)));
    }

    #[test]
    fn any_scalar_into() {
        let p00: AnyScalar = 2.0_f32.into();
        let p0: AnyScalar = 2.0_f64.into();
        let p10: AnyScalar = Complex32::new(3.0, 0.25).into();
        let p1: AnyScalar = Complex64::new(3.0, 1.0).into();
        assert!(matches!(p00, AnyScalar::Primal(BaseScalar::F32(_))));
        assert!(matches!(p0, AnyScalar::Primal(BaseScalar::F64(_))));
        assert!(matches!(p10, AnyScalar::Primal(BaseScalar::C32(_))));
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

    #[test]
    fn generic_scalar_like() {
        fn lift<T: BaseScalarLike>(value: T) -> AnyScalar {
            value.into()
        }

        let a0 = lift(1.0_f32);
        let a = lift(1.0_f64);
        let b0 = lift(Complex32::new(2.0, 0.75));
        let b = lift(Complex64::new(1.0, -0.25));
        assert!(matches!(a0, AnyScalar::Primal(BaseScalar::F32(_))));
        assert!(matches!(a, AnyScalar::Primal(BaseScalar::F64(_))));
        assert!(matches!(b0, AnyScalar::Primal(BaseScalar::C32(_))));
        assert!(matches!(b, AnyScalar::Primal(BaseScalar::C64(_))));
    }
}
