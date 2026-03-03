use super::*;

/// Builder for SVD.
/// # Examples
///
/// ```ignore
/// // Construct `SvdBuilder` via its corresponding operation constructor.
/// ```
pub struct SvdBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    options: Option<&'a SvdOptions>,
}

impl<'a, T> SvdBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
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

    /// Executes SVD.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<SvdResult<T, T::Real>> {
        with_cpu_runtime("svd", |ctx| {
            tenferro_linalg::svd::<T, CpuContext>(ctx, self.tensor, self.options)
                .map_err(Error::from)
        })
    }
}

/// Creates an SVD builder.
/// # Examples
///
/// ```ignore
/// let _ = svd(/* ... */);
/// ```
pub fn svd<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> SvdBuilder<'a, T> {
    SvdBuilder {
        tensor,
        options: None,
    }
}

/// Builder for QR.
/// # Examples
///
/// ```ignore
/// // Construct `QrBuilder` via its corresponding operation constructor.
/// ```
pub struct QrBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> QrBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes QR.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<QrResult<T>> {
        with_cpu_runtime("qr", |ctx| {
            tenferro_linalg::qr::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates a QR builder.
/// # Examples
///
/// ```ignore
/// let _ = qr(/* ... */);
/// ```
pub fn qr<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> QrBuilder<'a, T> {
    QrBuilder { tensor }
}

/// Builder for LU.
/// # Examples
///
/// ```ignore
/// // Construct `LuBuilder` via its corresponding operation constructor.
/// ```
pub struct LuBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    pivot: LuPivot,
}

impl<'a, T> LuBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Sets LU pivoting policy.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.pivot(pivot);
    /// ```
    pub fn pivot(mut self, pivot: LuPivot) -> Self {
        self.pivot = pivot;
        self
    }

    /// Executes LU.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<LuResult<T>> {
        with_cpu_runtime("lu", |ctx| {
            tenferro_linalg::lu::<T, CpuContext>(ctx, self.tensor, self.pivot).map_err(Error::from)
        })
    }
}

/// Creates an LU builder.
/// # Examples
///
/// ```ignore
/// let _ = lu(/* ... */);
/// ```
pub fn lu<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> LuBuilder<'a, T> {
    LuBuilder {
        tensor,
        pivot: LuPivot::Partial,
    }
}

/// Builder for eigen decomposition.
/// # Examples
///
/// ```ignore
/// // Construct `EigenBuilder` via its corresponding operation constructor.
/// ```
pub struct EigenBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> EigenBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes eigen decomposition.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<EigenResult<T, T::Real>> {
        with_cpu_runtime("eigen", |ctx| {
            tenferro_linalg::eigen::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates an eigen builder.
/// # Examples
///
/// ```ignore
/// let _ = eigen(/* ... */);
/// ```
pub fn eigen<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> EigenBuilder<'a, T> {
    EigenBuilder { tensor }
}

/// Builder for least squares solve.
/// # Examples
///
/// ```ignore
/// // Construct `LstsqBuilder` via its corresponding operation constructor.
/// ```
pub struct LstsqBuilder<'a, T: LinalgScalar> {
    a: &'a Tensor<T>,
    b: &'a Tensor<T>,
}

impl<'a, T> LstsqBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes least squares.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<LstsqResult<T>> {
        with_cpu_runtime("lstsq", |ctx| {
            tenferro_linalg::lstsq::<T>(ctx, self.a, self.b).map_err(Error::from)
        })
    }
}

/// Creates an lstsq builder.
/// # Examples
///
/// ```ignore
/// let _ = lstsq(/* ... */);
/// ```
pub fn lstsq<'a, T: LinalgScalar>(a: &'a Tensor<T>, b: &'a Tensor<T>) -> LstsqBuilder<'a, T> {
    LstsqBuilder { a, b }
}

/// Builder for Cholesky.
/// # Examples
///
/// ```ignore
/// // Construct `CholeskyBuilder` via its corresponding operation constructor.
/// ```
pub struct CholeskyBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> CholeskyBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes Cholesky.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_cpu_runtime("cholesky", |ctx| {
            tenferro_linalg::cholesky::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates a cholesky builder.
/// # Examples
///
/// ```ignore
/// let _ = cholesky(/* ... */);
/// ```
pub fn cholesky<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> CholeskyBuilder<'a, T> {
    CholeskyBuilder { tensor }
}

/// Builder for solve.
/// # Examples
///
/// ```ignore
/// // Construct `SolveBuilder` via its corresponding operation constructor.
/// ```
pub struct SolveBuilder<'a, T: LinalgScalar> {
    a: &'a Tensor<T>,
    b: &'a Tensor<T>,
}

impl<'a, T> SolveBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes linear solve.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_cpu_runtime("solve", |ctx| {
            tenferro_linalg::solve::<T, CpuContext>(ctx, self.a, self.b).map_err(Error::from)
        })
    }
}

/// Creates a solve builder.
/// # Examples
///
/// ```ignore
/// let _ = solve(/* ... */);
/// ```
pub fn solve<'a, T: LinalgScalar>(a: &'a Tensor<T>, b: &'a Tensor<T>) -> SolveBuilder<'a, T> {
    SolveBuilder { a, b }
}
