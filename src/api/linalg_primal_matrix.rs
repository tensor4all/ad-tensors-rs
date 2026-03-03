use super::*;

/// Builder for matrix inverse.
/// # Examples
///
/// ```ignore
/// // Construct `InvBuilder` via its corresponding operation constructor.
/// ```
pub struct InvBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> InvBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes matrix inverse.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_cpu_runtime("inv", |ctx| {
            tenferro_linalg::inv::<T>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates an inv builder.
/// # Examples
///
/// ```ignore
/// let _ = inv(/* ... */);
/// ```
pub fn inv<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> InvBuilder<'a, T> {
    InvBuilder { tensor }
}

/// Builder for determinant.
/// # Examples
///
/// ```ignore
/// // Construct `DetBuilder` via its corresponding operation constructor.
/// ```
pub struct DetBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> DetBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes determinant.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_cpu_runtime("det", |ctx| {
            tenferro_linalg::det::<T>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates a det builder.
/// # Examples
///
/// ```ignore
/// let _ = det(/* ... */);
/// ```
pub fn det<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> DetBuilder<'a, T> {
    DetBuilder { tensor }
}

/// Builder for slogdet.
/// # Examples
///
/// ```ignore
/// // Construct `SlogdetBuilder` via its corresponding operation constructor.
/// ```
pub struct SlogdetBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> SlogdetBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes slogdet.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<SlogdetResult<T>> {
        with_cpu_runtime("slogdet", |ctx| {
            tenferro_linalg::slogdet::<T>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates an slogdet builder.
/// # Examples
///
/// ```ignore
/// let _ = slogdet(/* ... */);
/// ```
pub fn slogdet<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> SlogdetBuilder<'a, T> {
    SlogdetBuilder { tensor }
}

/// Builder for general eigendecomposition.
/// # Examples
///
/// ```ignore
/// // Construct `EigBuilder` via its corresponding operation constructor.
/// ```
pub struct EigBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> EigBuilder<'a, T>
where
    T: LinalgScalar<Real = T, Complex = Complex<T>> + Float + CpuLinalgScalar,
{
    /// Executes eig.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<EigResult<T>> {
        with_cpu_runtime("eig", |ctx| {
            tenferro_linalg::eig::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates an eig builder.
/// # Examples
///
/// ```ignore
/// let _ = eig(/* ... */);
/// ```
pub fn eig<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> EigBuilder<'a, T> {
    EigBuilder { tensor }
}

/// Builder for pseudoinverse.
/// # Examples
///
/// ```ignore
/// // Construct `PinvBuilder` via its corresponding operation constructor.
/// ```
pub struct PinvBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    rcond: Option<f64>,
}

impl<'a, T> PinvBuilder<'a, T>
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

    /// Executes pseudoinverse.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_cpu_runtime("pinv", |ctx| {
            tenferro_linalg::pinv::<T>(ctx, self.tensor, self.rcond).map_err(Error::from)
        })
    }
}

/// Creates a pinv builder.
/// # Examples
///
/// ```ignore
/// let _ = pinv(/* ... */);
/// ```
pub fn pinv<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> PinvBuilder<'a, T> {
    PinvBuilder {
        tensor,
        rcond: None,
    }
}

/// Builder for matrix exponential.
/// # Examples
///
/// ```ignore
/// // Construct `MatrixExpBuilder` via its corresponding operation constructor.
/// ```
pub struct MatrixExpBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> MatrixExpBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes matrix exponential.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_cpu_runtime("matrix_exp", |ctx| {
            tenferro_linalg::matrix_exp::<T>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates a matrix_exp builder.
/// # Examples
///
/// ```ignore
/// let _ = matrix_exp(/* ... */);
/// ```
pub fn matrix_exp<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> MatrixExpBuilder<'a, T> {
    MatrixExpBuilder { tensor }
}

/// Builder for triangular solve.
/// # Examples
///
/// ```ignore
/// // Construct `SolveTriangularBuilder` via its corresponding operation constructor.
/// ```
pub struct SolveTriangularBuilder<'a, T: LinalgScalar> {
    a: &'a Tensor<T>,
    b: &'a Tensor<T>,
    upper: bool,
}

impl<'a, T> SolveTriangularBuilder<'a, T>
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

    /// Executes triangular solve.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_cpu_runtime("solve_triangular", |ctx| {
            tenferro_linalg::solve_triangular::<T, CpuContext>(ctx, self.a, self.b, self.upper)
                .map_err(Error::from)
        })
    }
}

/// Creates a solve_triangular builder.
/// # Examples
///
/// ```ignore
/// let _ = solve_triangular(/* ... */);
/// ```
pub fn solve_triangular<'a, T: LinalgScalar>(
    a: &'a Tensor<T>,
    b: &'a Tensor<T>,
) -> SolveTriangularBuilder<'a, T> {
    SolveTriangularBuilder { a, b, upper: true }
}

/// Builder for norm.
/// # Examples
///
/// ```ignore
/// // Construct `NormBuilder` via its corresponding operation constructor.
/// ```
pub struct NormBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    kind: NormKind,
}

impl<'a, T> NormBuilder<'a, T>
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

    /// Executes norm.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_cpu_runtime("norm", |ctx| {
            tenferro_linalg::norm::<T>(ctx, self.tensor, self.kind).map_err(Error::from)
        })
    }
}

/// Creates a norm builder.
/// # Examples
///
/// ```ignore
/// let _ = norm(/* ... */);
/// ```
pub fn norm<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> NormBuilder<'a, T> {
    NormBuilder {
        tensor,
        kind: NormKind::Fro,
    }
}
