use num_complex::Complex64;

use crate::mode::NodeId;

/// Base scalar domain used by the AD wrappers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BaseScalar {
    F64(f64),
    C64(Complex64),
}

/// Runtime scalar preserving AD mode metadata.
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
