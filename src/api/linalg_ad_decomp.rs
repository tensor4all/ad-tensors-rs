use super::*;

/// Builder for AD SVD.
/// # Examples
///
/// ```ignore
/// // Construct `SvdAdBuilder` via its corresponding operation constructor.
/// ```
pub struct SvdAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
    options: Option<&'a SvdOptions>,
}

impl<'a, T> SvdAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Sets optional SVD options.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.options(&options);
    /// ```
    pub fn options(mut self, options: &'a SvdOptions) -> Self {
        self.options = Some(options);
        self
    }

    /// Executes AD SVD.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdSvdResult<T>> {
        let operands = [self.tensor];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);

        let (primal, tangent) = if needs_tangent {
            let dt = self
                .tensor
                .tangent()
                .cloned()
                .unwrap_or_else(|| zero_like(self.tensor.primal()));
            let (p, d) = with_cpu_runtime("svd_ad", |ctx| {
                tenferro_linalg::svd_frule::<T>(ctx, self.tensor.primal(), &dt, self.options)
                    .map_err(Error::from)
            })?;
            (p, Some(d))
        } else {
            (
                with_cpu_runtime("svd_ad", |ctx| {
                    tenferro_linalg::svd::<T, CpuContext>(ctx, self.tensor.primal(), self.options)
                        .map_err(Error::from)
                })?,
                None,
            )
        };

        let (du, ds, dvt) = if let Some(d) = tangent {
            (Some(d.u), Some(d.s), Some(d.vt))
        } else {
            (None, None, None)
        };

        Ok(AdSvdResult {
            u: wrap_ad_output("svd_ad", &operands, primal.u, du, 1)?,
            s: wrap_ad_output("svd_ad", &operands, primal.s, ds, 2)?,
            vt: wrap_ad_output("svd_ad", &operands, primal.vt, dvt, 3)?,
        })
    }
}

/// Creates an AD SVD builder.
/// # Examples
///
/// ```ignore
/// let _ = svd_ad(/* ... */);
/// ```
pub fn svd_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> SvdAdBuilder<'a, T> {
    SvdAdBuilder {
        tensor,
        options: None,
    }
}

/// Builder for AD QR.
/// # Examples
///
/// ```ignore
/// // Construct `QrAdBuilder` via its corresponding operation constructor.
/// ```
pub struct QrAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> QrAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes AD QR.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdQrResult<T>> {
        let operands = [self.tensor];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);

        let (primal, tangent) = if needs_tangent {
            let dt = self
                .tensor
                .tangent()
                .cloned()
                .unwrap_or_else(|| zero_like(self.tensor.primal()));
            let (p, d) = with_cpu_runtime("qr_ad", |ctx| {
                tenferro_linalg::qr_frule::<T>(ctx, self.tensor.primal(), &dt).map_err(Error::from)
            })?;
            (p, Some(d))
        } else {
            (
                with_cpu_runtime("qr_ad", |ctx| {
                    tenferro_linalg::qr::<T, CpuContext>(ctx, self.tensor.primal())
                        .map_err(Error::from)
                })?,
                None,
            )
        };

        let (dq, dr) = if let Some(d) = tangent {
            (Some(d.q), Some(d.r))
        } else {
            (None, None)
        };

        Ok(AdQrResult {
            q: wrap_ad_output("qr_ad", &operands, primal.q, dq, 1)?,
            r: wrap_ad_output("qr_ad", &operands, primal.r, dr, 2)?,
        })
    }
}

/// Creates an AD QR builder.
/// # Examples
///
/// ```ignore
/// let _ = qr_ad(/* ... */);
/// ```
pub fn qr_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> QrAdBuilder<'a, T> {
    QrAdBuilder { tensor }
}

/// Builder for AD LU.
/// # Examples
///
/// ```ignore
/// // Construct `LuAdBuilder` via its corresponding operation constructor.
/// ```
pub struct LuAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
    pivot: LuPivot,
}

impl<'a, T> LuAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Sets LU pivot policy.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.pivot(pivot);
    /// ```
    pub fn pivot(mut self, pivot: LuPivot) -> Self {
        self.pivot = pivot;
        self
    }

    /// Executes AD LU.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdLuResult<T>> {
        let operands = [self.tensor];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);

        let (primal, tangent) = if needs_tangent {
            let dt = self
                .tensor
                .tangent()
                .cloned()
                .unwrap_or_else(|| zero_like(self.tensor.primal()));
            let (p, d) = with_cpu_runtime("lu_ad", |ctx| {
                tenferro_linalg::lu_frule::<T>(ctx, self.tensor.primal(), &dt, self.pivot)
                    .map_err(Error::from)
            })?;
            (p, Some(d))
        } else {
            (
                with_cpu_runtime("lu_ad", |ctx| {
                    tenferro_linalg::lu::<T, CpuContext>(ctx, self.tensor.primal(), self.pivot)
                        .map_err(Error::from)
                })?,
                None,
            )
        };

        let (dl, du) = if let Some(d) = tangent {
            (Some(d.l), Some(d.u))
        } else {
            (None, None)
        };

        Ok(AdLuResult {
            p: primal.p,
            l: wrap_ad_output("lu_ad", &operands, primal.l, dl, 1)?,
            u: wrap_ad_output("lu_ad", &operands, primal.u, du, 2)?,
        })
    }
}

/// Creates an AD LU builder.
/// # Examples
///
/// ```ignore
/// let _ = lu_ad(/* ... */);
/// ```
pub fn lu_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> LuAdBuilder<'a, T> {
    LuAdBuilder {
        tensor,
        pivot: LuPivot::Partial,
    }
}

/// Builder for AD eigen decomposition.
/// # Examples
///
/// ```ignore
/// // Construct `EigenAdBuilder` via its corresponding operation constructor.
/// ```
pub struct EigenAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> EigenAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes AD eigen decomposition.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdEigenResult<T>> {
        let operands = [self.tensor];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);

        let (primal, tangent) = if needs_tangent {
            let dt = self
                .tensor
                .tangent()
                .cloned()
                .unwrap_or_else(|| zero_like(self.tensor.primal()));
            let (p, d) = with_cpu_runtime("eigen_ad", |ctx| {
                tenferro_linalg::eigen_frule::<T>(ctx, self.tensor.primal(), &dt)
                    .map_err(Error::from)
            })?;
            (p, Some(d))
        } else {
            (
                with_cpu_runtime("eigen_ad", |ctx| {
                    tenferro_linalg::eigen::<T, CpuContext>(ctx, self.tensor.primal())
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

        Ok(AdEigenResult {
            values: wrap_ad_output("eigen_ad", &operands, primal.values, dvalues, 1)?,
            vectors: wrap_ad_output("eigen_ad", &operands, primal.vectors, dvectors, 2)?,
        })
    }
}

/// Creates an AD eigen builder.
/// # Examples
///
/// ```ignore
/// let _ = eigen_ad(/* ... */);
/// ```
pub fn eigen_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> EigenAdBuilder<'a, T> {
    EigenAdBuilder { tensor }
}

/// Builder for AD least squares.
/// # Examples
///
/// ```ignore
/// // Construct `LstsqAdBuilder` via its corresponding operation constructor.
/// ```
pub struct LstsqAdBuilder<'a, T: Scalar> {
    a: &'a AdTensor<T>,
    b: &'a AdTensor<T>,
}

impl<'a, T> LstsqAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes AD least squares.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdLstsqResult<T>> {
        let operands = [self.a, self.b];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);

        let (primal, tangent) = if needs_tangent {
            let da = self
                .a
                .tangent()
                .cloned()
                .unwrap_or_else(|| zero_like(self.a.primal()));
            let db = self
                .b
                .tangent()
                .cloned()
                .unwrap_or_else(|| zero_like(self.b.primal()));
            let (p, d) = with_cpu_runtime("lstsq_ad", |ctx| {
                tenferro_linalg::lstsq_frule::<T>(ctx, self.a.primal(), self.b.primal(), &da, &db)
                    .map_err(Error::from)
            })?;
            (p, Some(d))
        } else {
            (
                with_cpu_runtime("lstsq_ad", |ctx| {
                    tenferro_linalg::lstsq::<T>(ctx, self.a.primal(), self.b.primal())
                        .map_err(Error::from)
                })?,
                None,
            )
        };

        let (dx, dresidual) = if let Some(d) = tangent {
            (Some(d.x), Some(d.residual))
        } else {
            (None, None)
        };

        Ok(AdLstsqResult {
            x: wrap_ad_output("lstsq_ad", &operands, primal.x, dx, 1)?,
            residual: wrap_ad_output("lstsq_ad", &operands, primal.residual, dresidual, 2)?,
        })
    }
}

/// Creates an AD lstsq builder.
/// # Examples
///
/// ```ignore
/// let _ = lstsq_ad(/* ... */);
/// ```
pub fn lstsq_ad<'a, T: Scalar>(a: &'a AdTensor<T>, b: &'a AdTensor<T>) -> LstsqAdBuilder<'a, T> {
    LstsqAdBuilder { a, b }
}

/// Builder for AD Cholesky.
/// # Examples
///
/// ```ignore
/// // Construct `CholeskyAdBuilder` via its corresponding operation constructor.
/// ```
pub struct CholeskyAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> CholeskyAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes AD Cholesky.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad(
            "cholesky_ad",
            self.tensor,
            |ctx, t| tenferro_linalg::cholesky::<T, CpuContext>(ctx, t).map_err(Error::from),
            |ctx, t, dt| tenferro_linalg::cholesky_frule::<T>(ctx, t, dt),
        )
    }
}

/// Creates an AD cholesky builder.
/// # Examples
///
/// ```ignore
/// let _ = cholesky_ad(/* ... */);
/// ```
pub fn cholesky_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> CholeskyAdBuilder<'a, T> {
    CholeskyAdBuilder { tensor }
}

/// Builder for AD solve.
/// # Examples
///
/// ```ignore
/// // Construct `SolveAdBuilder` via its corresponding operation constructor.
/// ```
pub struct SolveAdBuilder<'a, T: Scalar> {
    a: &'a AdTensor<T>,
    b: &'a AdTensor<T>,
}

impl<'a, T> SolveAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes AD solve.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_binary_tensor_ad(
            "solve_ad",
            self.a,
            self.b,
            |ctx, a, b| tenferro_linalg::solve::<T, CpuContext>(ctx, a, b).map_err(Error::from),
            |ctx, a, b, da, db| tenferro_linalg::solve_frule::<T>(ctx, a, b, da, db),
        )
    }
}

/// Creates an AD solve builder.
/// # Examples
///
/// ```ignore
/// let _ = solve_ad(/* ... */);
/// ```
pub fn solve_ad<'a, T: Scalar>(a: &'a AdTensor<T>, b: &'a AdTensor<T>) -> SolveAdBuilder<'a, T> {
    SolveAdBuilder { a, b }
}
