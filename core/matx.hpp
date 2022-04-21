#pragma once
#include <initializer_list>

namespace tt
{
    template <typename _Tp>
    class DataType
    {
    };

    template <typename _Tp, int cn>
    class Vec;

    struct Matx_Addop
    {
        Matx_Addop() {}
        Matx_Addop(const Matx_Addop &) {}
    };
    struct Matx_SubOp
    {
        Matx_SubOp() {}
        Matx_SubOp(const Matx_SubOp &) {}
    };
    struct Matx_ScaleOp
    {
        Matx_ScaleOp() {}
        Matx_ScaleOp(const Matx_ScaleOp &) {}
    };
    struct Matx_MulOp
    {
        Matx_MulOp() {}
        Matx_MulOp(const Matx_MulOp &) {}
    };
    struct Matx_DivOp
    {
        Matx_DivOp() {}
        Matx_DivOp(const Matx_DivOp &) {}
    };
    struct Matx_MatMulOp
    {
        Matx_MatMulOp() {}
        Matx_MatMulOp(const Matx_MatMulOp &) {}
    };
    struct Matx_TOp
    {
        Matx_TOp() {}
        Matx_TOp(const Matx_TOp &) {}
    };

    //! matrix decomposition types
    enum DecompTypes
    {
        /** Gaussian elimination with the optimal pivot element chosen. */
        DECOMP_LU = 0,
        /** singular value decomposition (SVD) method; the system can be over-defined and/or the matrix
        src1 can be singular */
        DECOMP_SVD = 1,
        /** eigenvalue decomposition; the matrix src1 must be symmetrical */
        DECOMP_EIG = 2,
        /** Cholesky \f$LL^T\f$ factorization; the matrix src1 must be symmetrical and positively
        defined */
        DECOMP_CHOLESKY = 3,
        /** QR factorization; the system can be over-defined and/or the matrix src1 can be singular */
        DECOMP_QR = 4,
        /** while all the previous flags are mutually exclusive, this flag can be used together with
        any of the previous; it means that the normal equations
        \f$\texttt{src1}^T\cdot\texttt{src1}\cdot\texttt{dst}=\texttt{src1}^T\texttt{src2}\f$ are
        solved instead of the original system
        \f$\texttt{src1}\cdot\texttt{dst}=\texttt{src2}\f$ */
        DECOMP_NORMAL = 16
    };

    template <typename _Tp, int m, int n>
    class Matx
    {
    public:
        enum
        {
            rows = m,
            cols = n,
            channels = rows * cols,
            shortdim = ((m < n) ? m : n)
        };

        using value_type = _Tp;
        using mat_type = Matx<_Tp, m, n>;
        using diag_type = Matx<_Tp, shortdim, 1>;

        //! default constructor
        Matx();

        explicit Matx(_Tp v0);                                                                //!< 1x1 matrix
        Matx(_Tp v0, _Tp v1);                                                                 //!< 1x2 or 2x1 matrix
        Matx(_Tp v0, _Tp v1, _Tp v2);                                                         //!< 1x3 or 3x1 matrix
        Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3);                                                 //!< 1x4, 2x2 or 4x1 matrix
        Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4);                                         //!< 1x5 or 5x1 matrix
        Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5);                                 //!< 1x6, 2x3, 3x2 or 6x1 matrix
        Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6);                         //!< 1x7 or 7x1 matrix
        Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7);                 //!< 1x8, 2x4, 4x2 or 8x1 matrix
        Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8);         //!< 1x9, 3x3 or 9x1 matrix
        Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9); //!< 1x10, 2x5 or 5x2 or 10x1 matrix
        Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3,
             _Tp v4, _Tp v5, _Tp v6, _Tp v7,
             _Tp v8, _Tp v9, _Tp v10, _Tp v11); //!< 1x12, 2x6, 3x4, 4x3, 6x2 or 12x1 matrix
        Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3,
             _Tp v4, _Tp v5, _Tp v6, _Tp v7,
             _Tp v8, _Tp v9, _Tp v10, _Tp v11,
             _Tp v12, _Tp v13); //!< 1x14, 2x7, 7x2 or 14x1 matrix
        Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3,
             _Tp v4, _Tp v5, _Tp v6, _Tp v7,
             _Tp v8, _Tp v9, _Tp v10, _Tp v11,
             _Tp v12, _Tp v13, _Tp v14, _Tp v15); //!< 1x16, 4x4 or 16x1 matrix
        explicit Matx(const _Tp *vals);           //!< initialize from a plain array

        Matx(std::initializer_list<_Tp>); //!< initialize from an initializer list

        [[nodiscard]] static Matx all(_Tp alpha);
        [[nodiscard]] static Matx zeros();
        [[nodiscard]] static Matx ones();
        [[nodiscard]] static Matx eye();
        [[nodiscard]] static Matx diag(const diag_type &d);

        /**
         * @brief Generates uniformly distributed random numbers
         *
         * @param a Range boundary.
         * @param b The other range boundary (boundaries don't have to be ordered, the lower boundary is inclusive,
         *          the upper one is exclusive).
         * @return Matx
         */
        [[nodiscard]] static Matx randu(_Tp a, _Tp b);

        /**
         * @brief Generates normally distributed random numbers
         *
         * @param a Mean value.
         * @param b Standard deviation.
         * @return Matx
         */
        [[nodiscard]] static Matx randn(_Tp a, _Tp b);

        //! dot product computed with the default precision
        _Tp dot(const Matx<_Tp, m, n> &v) const;

        //! dot product computed in double-precision arithmetics
        double ddot(const Matx<_Tp, m, n> &v) const;

        //! conversion to another data type
        template <typename T2>
        operator Matx<T2, m, n>() const;

        //! change the matrix shape
        template <int m1, int n1>
        Matx<_Tp, m1, n1> reshape() const;

        //! extract part of the matrix
        template <int m1, int n1>
        Matx<_Tp, m1, n1> get_minor(int base_row, int base_col) const;

        //! extract the matrix row
        Matx<_Tp, 1, n> row(int i) const;

        //! extract the matrix col
        Matx<_Tp, m, 1> col(int i) const;

        //! extract the matrix diagonal
        diag_type diag() const;

        //! transpose the matrix
        Matx<_Tp, n, m> t() const;

        //! invert the matrix
        Matx<_Tp, m, n> inv(int method = DECOMP_LU, bool *p_is_ok = NULL) const;

        //! solve linear system
        template <int l>
        Matx<_Tp, n, l> solve(const Matx<_Tp, m, l> &rhs, int flag = DECOMP_LU) const;
        Vec<_Tp, n> solve(const Vec<_Tp, m> &rhs, int method) const;

        //! multiply tow matrices element-wise
        Matx<_Tp, m, n> mul(const Matx<_Tp, m, n> &a) const;

        //! divide tow matrices element-wise
        Matx<_Tp, m, n> div(const Matx<_Tp, m, n> &a) const;

        //! element access
        const _Tp &operator()(int row, int col) const;
        _Tp &operator()(int row, int col) const;

        //! 1D element access
        const _Tp &operator()(int i) const;
        _Tp &operator()(int i) const;

        Matx(const Matx<_Tp, m, n> &a, const Matx<_Tp, m, n> &b, Matx_AddOp);
        Matx(const Matx<_Tp, m, n> &a, const Matx<_Tp, m, n> &b, Matx_SubOp);
        template <typename _T2>
        Matx(const Matx<_Tp, m, n> &a, _T2 alpha, Matx_ScaleOp);
        Matx(const Matx<_Tp, m, n> &a, const Matx<_Tp, m, n> &b, Matx_MulOp);
        Matx(const Matx<_Tp, m, n> &a, const Matx<_Tp, m, n> &b, Matx_DivOp);
        template <int l>
        Matx(const Matx<_Tp, m, l> &a, const Matx<_Tp, l, n> &b, Matx_MatMulOp);
        Matx(const Matx<_Tp, m, n> &a, Matx_TOp);

        _Tp val[m * n]; //< matrix elements
    };

    /*
 Utility methods
*/
    template <typename _Tp, int m>
    static double determinant(const Matx<_Tp, m, m> &a);
    template <typename _Tp, int m, int n>
    static double trace(const Matx<_Tp, m, n> &a);
    template <typename _Tp, int m, int n>
    static double norm(const Matx<_Tp, m, n> &M);
    template <typename _Tp, int m, int n>
    static double norm(const Matx<_Tp, m, n> &M, int normType);

    template<typename _Tp, int cn>
    class Vec : public Matx<_Tp, cn, 1>
    {
    public:
        typedef _Tp value_type;
        enum
        {
            channels = cn,
#ifdef OPENCV_TRAITS_ENABLE_DEPRECATED
            depth = Matx<_Tp, cn, 1>::depth,
            type = CV_MAKETYPE(depth, channels),
#endif
            _dummy_enum_finalizer = 0
        };

        //! default constructor
        Vec();

        Vec(_Tp v0);                                                                                                             //!< 1-element vector constructor
        Vec(_Tp v0, _Tp v1);                                                                                                     //!< 2-element vector constructor
        Vec(_Tp v0, _Tp v1, _Tp v2);                                                                                             //!< 3-element vector constructor
        Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3);                                                                                     //!< 4-element vector constructor
        Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4);                                                                             //!< 5-element vector constructor
        Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5);                                                                     //!< 6-element vector constructor
        Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6);                                                             //!< 7-element vector constructor
        Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7);                                                     //!< 8-element vector constructor
        Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8);                                             //!< 9-element vector constructor
        Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9);                                     //!< 10-element vector constructor
        Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9, _Tp v10, _Tp v11, _Tp v12, _Tp v13); //!< 14-element vector constructor
        explicit Vec(const _Tp *values);

        Vec(std::initializer_list<_Tp>);

        Vec(const Vec<_Tp, cn> &v);

        static Vec all(_Tp alpha);

        //! per-element multiplication
        Vec mul(const Vec<_Tp, cn> &v) const;

        //! conjugation (makes sense for complex numbers and quaternions)
        Vec conj() const;

        /*!
          cross product of the two 3D vectors.

          For other dimensionalities the exception is raised
        */
        Vec cross(const Vec &v) const;
        //! conversion to another data type
        template <typename T2>
        operator Vec<T2, cn>() const;

        /*! element access */
        const _Tp &operator[](int i) const;
        _Tp &operator[](int i);
        const _Tp &operator()(int i) const;
        _Tp &operator()(int i);

#ifdef CV_CXX11
        Vec<_Tp, cn> &operator=(const Vec<_Tp, cn> &rhs) = default;
#endif

        Vec(const Matx<_Tp, cn, 1> &a, const Matx<_Tp, cn, 1> &b, Matx_AddOp);
        Vec(const Matx<_Tp, cn, 1> &a, const Matx<_Tp, cn, 1> &b, Matx_SubOp);
        template <typename _T2>
        Vec(const Matx<_Tp, cn, 1> &a, _T2 alpha, Matx_ScaleOp);
    };
}