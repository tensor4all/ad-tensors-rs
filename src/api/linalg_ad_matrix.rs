use super::*;

/// Builder for AD inverse.
/// # Examples
///
/// ```ignore
/// // Construct `InvAdBuilder` via its corresponding operation constructor.
/// ```
pub struct InvAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> InvAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes AD inverse.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad(
            "inv_ad",
            self.tensor,
            |ctx, t| tenferro_linalg::inv::<T>(ctx, t).map_err(Error::from),
            |ctx, t, dt| tenferro_linalg::inv_frule::<T>(ctx, t, dt),
        )
    }
}

/// Creates an AD inv builder.
/// # Examples
///
/// ```ignore
/// let _ = inv_ad(/* ... */);
/// ```
pub fn inv_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> InvAdBuilder<'a, T> {
    InvAdBuilder { tensor }
}

/// Builder for AD det.
/// # Examples
///
/// ```ignore
/// // Construct `DetAdBuilder` via its corresponding operation constructor.
/// ```
pub struct DetAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> DetAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes AD determinant.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad(
            "det_ad",
            self.tensor,
            |ctx, t| tenferro_linalg::det::<T>(ctx, t).map_err(Error::from),
            |ctx, t, dt| tenferro_linalg::det_frule::<T>(ctx, t, dt),
        )
    }
}

/// Creates an AD det builder.
/// # Examples
///
/// ```ignore
/// let _ = det_ad(/* ... */);
/// ```
pub fn det_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> DetAdBuilder<'a, T> {
    DetAdBuilder { tensor }
}

/// Builder for AD slogdet.
/// # Examples
///
/// ```ignore
/// // Construct `SlogdetAdBuilder` via its corresponding operation constructor.
/// ```
pub struct SlogdetAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> SlogdetAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes AD slogdet.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdSlogdetResult<T>> {
        let operands = [self.tensor];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);

        let (primal, tangent) = if needs_tangent {
            let dt = self
                .tensor
                .tangent()
                .cloned()
                .unwrap_or_else(|| zero_like(self.tensor.primal()));
            let (p, d) = with_cpu_runtime("slogdet_ad", |ctx| {
                tenferro_linalg::slogdet_frule::<T>(ctx, self.tensor.primal(), &dt)
                    .map_err(Error::from)
            })?;
            (p, Some(d))
        } else {
            (
                with_cpu_runtime("slogdet_ad", |ctx| {
                    tenferro_linalg::slogdet::<T>(ctx, self.tensor.primal()).map_err(Error::from)
                })?,
                None,
            )
        };

        let (dsign, dlogabsdet) = if let Some(d) = tangent {
            (Some(d.sign), Some(d.logabsdet))
        } else {
            (None, None)
        };

        Ok(AdSlogdetResult {
            sign: wrap_ad_output("slogdet_ad", &operands, primal.sign, dsign, 1)?,
            logabsdet: wrap_ad_output("slogdet_ad", &operands, primal.logabsdet, dlogabsdet, 2)?,
        })
    }
}

/// Creates an AD slogdet builder.
/// # Examples
///
/// ```ignore
/// let _ = slogdet_ad(/* ... */);
/// ```
pub fn slogdet_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> SlogdetAdBuilder<'a, T> {
    SlogdetAdBuilder { tensor }
}

/// Builder for AD eig.
/// # Examples
///
/// ```ignore
/// // Construct `EigAdBuilder` via its corresponding operation constructor.
/// ```
pub struct EigAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> EigAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T, Complex = Complex<T>> + Float + CpuLinalgScalar,
    Complex<T>: Scalar,
{
    /// Executes AD eig.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdEigResult<T>> {
        let operands = [self.tensor];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);

        let (primal, tangent) = if needs_tangent {
            let dt = self
                .tensor
                .tangent()
                .cloned()
                .unwrap_or_else(|| zero_like(self.tensor.primal()));
            let (p, d) = with_cpu_runtime("eig_ad", |ctx| {
                tenferro_linalg::eig_frule::<T>(ctx, self.tensor.primal(), &dt).map_err(Error::from)
            })?;
            (p, Some(d))
        } else {
            (
                with_cpu_runtime("eig_ad", |ctx| {
                    tenferro_linalg::eig::<T, CpuContext>(ctx, self.tensor.primal())
                        .map_err(Error::from)
                })?,
                None,
            )
        };

        let (dvalues, dvectors) = if let Some(d) = tangent {
            (Some(d.values), Some(d.vectors))
        } else {
            (None, None)
        };

        Ok(AdEigResult {
            values: wrap_ad_output("eig_ad", &operands, primal.values, dvalues, 1)?,
            vectors: wrap_ad_output("eig_ad", &operands, primal.vectors, dvectors, 2)?,
        })
    }
}

/// Creates an AD eig builder.
/// # Examples
///
/// ```ignore
/// let _ = eig_ad(/* ... */);
/// ```
pub fn eig_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> EigAdBuilder<'a, T> {
    EigAdBuilder { tensor }
}

/// Builder for AD pinv.
/// # Examples
///
/// ```ignore
/// // Construct `PinvAdBuilder` via its corresponding operation constructor.
/// ```
pub struct PinvAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
    rcond: Option<f64>,
}

impl<'a, T> PinvAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Sets rcond.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.rcond(1e-12);
    /// ```
    pub fn rcond(mut self, rcond: f64) -> Self {
        self.rcond = Some(rcond);
        self
    }

    /// Executes AD pinv.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad(
            "pinv_ad",
            self.tensor,
            |ctx, t| tenferro_linalg::pinv::<T>(ctx, t, self.rcond).map_err(Error::from),
            |ctx, t, dt| tenferro_linalg::pinv_frule::<T>(ctx, t, dt, self.rcond),
        )
    }
}

/// Creates an AD pinv builder.
/// # Examples
///
/// ```ignore
/// let _ = pinv_ad(/* ... */);
/// ```
pub fn pinv_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> PinvAdBuilder<'a, T> {
    PinvAdBuilder {
        tensor,
        rcond: None,
    }
}

/// Builder for AD matrix exponential.
/// # Examples
///
/// ```ignore
/// // Construct `MatrixExpAdBuilder` via its corresponding operation constructor.
/// ```
pub struct MatrixExpAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> MatrixExpAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes AD matrix exponential.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad(
            "matrix_exp_ad",
            self.tensor,
            |ctx, t| tenferro_linalg::matrix_exp::<T>(ctx, t).map_err(Error::from),
            |ctx, t, dt| tenferro_linalg::matrix_exp_frule::<T>(ctx, t, dt),
        )
    }
}

/// Creates an AD matrix_exp builder.
/// # Examples
///
/// ```ignore
/// let _ = matrix_exp_ad(/* ... */);
/// ```
pub fn matrix_exp_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> MatrixExpAdBuilder<'a, T> {
    MatrixExpAdBuilder { tensor }
}

/// Builder for AD solve_triangular.
/// # Examples
///
/// ```ignore
/// // Construct `SolveTriangularAdBuilder` via its corresponding operation constructor.
/// ```
pub struct SolveTriangularAdBuilder<'a, T: Scalar> {
    a: &'a AdTensor<T>,
    b: &'a AdTensor<T>,
    upper: bool,
}

impl<'a, T> SolveTriangularAdBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Sets whether the matrix is upper triangular.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.upper(true);
    /// ```
    pub fn upper(mut self, upper: bool) -> Self {
        self.upper = upper;
        self
    }

    /// Executes AD triangular solve.
    ///
    /// Forward/reverse inputs are currently unsupported.
    ///
    /// Upstream tracking issue: `tensor4all/tenferro-rs#257`.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        let operands = [self.a, self.b];
        if has_forward(&operands) || has_reverse(&operands) {
            return Err(Error::UnsupportedAdOp {
                op: "solve_triangular_ad",
            });
        }

        with_cpu_runtime("solve_triangular_ad", |ctx| {
            let out = tenferro_linalg::solve_triangular::<T, CpuContext>(
                ctx,
                self.a.primal(),
                self.b.primal(),
                self.upper,
            )
            .map_err(Error::from)?;
            Ok(AdTensor::new_primal(out))
        })
    }
}

/// Creates an AD solve_triangular builder.
/// # Examples
///
/// ```ignore
/// let _ = solve_triangular_ad(/* ... */);
/// ```
pub fn solve_triangular_ad<'a, T: Scalar>(
    a: &'a AdTensor<T>,
    b: &'a AdTensor<T>,
) -> SolveTriangularAdBuilder<'a, T> {
    SolveTriangularAdBuilder { a, b, upper: true }
}

/// Builder for AD norm.
/// # Examples
///
/// ```ignore
/// // Construct `NormAdBuilder` via its corresponding operation constructor.
/// ```
pub struct NormAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
    kind: NormKind,
}

impl<'a, T> NormAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Sets norm kind.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.kind(kind);
    /// ```
    pub fn kind(mut self, kind: NormKind) -> Self {
        self.kind = kind;
        self
    }

    /// Executes AD norm.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad(
            "norm_ad",
            self.tensor,
            |ctx, t| tenferro_linalg::norm::<T>(ctx, t, self.kind).map_err(Error::from),
            |ctx, t, dt| tenferro_linalg::norm_frule::<T>(ctx, t, dt, self.kind),
        )
    }
}

/// Creates an AD norm builder.
/// # Examples
///
/// ```ignore
/// let _ = norm_ad(/* ... */);
/// ```
pub fn norm_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> NormAdBuilder<'a, T> {
    NormAdBuilder {
        tensor,
        kind: NormKind::Fro,
    }
}
