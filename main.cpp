#include <iostream>
#include <assert.h>
#include <chrono>
#include <opencv2/core.hpp>
#include "core/matx.hpp"

template <typename _Tp>
static inline _Tp *alignPtr(_Tp *ptr, int n = (int)sizeof(_Tp))
{
    assert((n & (n - 1)) == 0); // n is a power of 2
    return (_Tp *)(((size_t)ptr + n - 1) & -n);
}

void *fastMalloc(size_t size)
{
    uint8_t *udata = (uint8_t *)malloc(size + sizeof(void *) + 64);
    if (!udata)
        return nullptr;
    uint8_t **adata = alignPtr((uint8_t **)udata + 1, 64);
    adata[-1] = udata;
    return adata;
}

void fastFree(void *ptr)
{
    if (ptr)
    {
        uint8_t *udata = ((uint8_t **)ptr)[-1];
        assert(udata < (uint8_t *)ptr && ((uint8_t *)ptr - udata) <= (ptrdiff_t)(sizeof(void *) + 64));
        free(udata);
    }
}

static inline size_t alignSize(size_t sz, int n)
{
    assert((n & (n - 1)) == 0); // n is a power of 2
    return (sz + n - 1) & -n;
}

template <typename _Tp>
class Allocator
{
public:
    using value_type = _Tp;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using reference = value_type &;
    using const_reference = const value_type &;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    template <typename U>
    class rebind
    {
        using other = Allocator<U>;
    };

    explicit Allocator() {}
    ~Allocator() {}
    explicit Allocator(Allocator const &) {}
    template <typename U>
    explicit Allocator(Allocator<U> const &) {}

    pointer address(reference r) { return &r; }
    const_pointer address(const_reference r) { return &r; }

    pointer allocate(size_type count, const void * = 0)
    {
        return reinterpret_cast<pointer>(fastMalloc(count * sizeof(_Tp)));
    }
    void deallocate(pointer p, size_type) { fastFree(p); }

    void construct(pointer p, const _Tp &v) { new (static_cast<void *>(p)) _Tp(v); }
    void destroy(pointer p) { p->~_Tp(); }

    size_type max_size() const { return std::max(_Tp(static_cast<_Tp>(-1) / sizeof(_Tp)), _Tp(1)); }
};

int main(int, char **)
{
    cv::Mat m = (cv::Mat_<int>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
    std::cout << m << std::endl;

    tt::Matx<int, 3, 1> m1(1, 2, 3);
    tt::Matx<int, 3, 1> m2 = {1, 2, 3};

    tt::Matx<int, 3, 3> m3 = tt::Matx<int, 3, 3>::eye();
    for (int i = 0; i < m3.rows; i++)
    {
        for (int j = 0; j < m3.cols; j++)
        {
            std::cout << m3(i, j);
        }
        std::cout << std::endl;
    }

    m3 = tt::Matx<int, 3, 3>::all(1);
    for (int i = 0; i < m3.rows; i++)
    {
        for (int j = 0; j < m3.cols; j++)
        {
            std::cout << m3(i, j);
        }
        std::cout << std::endl;
    }

    tt::Matx<int, 3, 3>::diag_type d = { 1, 2, 3 };
    m3 = tt::Matx<int, 3, 3>::diag(d);
    for (int i = 0; i < m3.rows; i++)
    {
        for (size_t j = 0; j < m3.cols; j++)
        {
            std::cout << m3(i, j);
        }
        std::cout << std::endl;
    }

    auto row = m3.row(1);
    for (int j = 0; j < row.cols; j++)
    {
        std::cout << row(0, j);
    }
    auto col = m3.col(0);
    for (int j = 0; j < col.rows; j++)
    {
        std::cout << col(j, 0);
    }
    std::cout << std::endl;
    std::cout << m3.ddot(m3) << std::endl;

    auto m5 = m3.reshape<9, 1>();

    auto m6 = m3.get_minor<2, 2>(1, 1);
    for (int j = 0; j < m6.channels; j++)
    {
        std::cout << m6.val[j];
    }
    std::cout << std::endl;

    auto m7 = cv::Matx<int, 10, 10>::randu(0, 20);
    std::cout << m7 << std::endl;

    auto m8 = cv::Matx<int, 10, 10>::randn(10, 1);
    std::cout << m8 << std::endl;
}
