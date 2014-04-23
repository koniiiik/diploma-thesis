#include <exception>
#include <NTL/LLL.h>
#include <vector>

using namespace std;
using namespace NTL;

/*
** LLL solver for knapsack using NTL.
**
** Input:
** n m
** a[0] a[1] ... a[n]
**
** Output:
** b[0] b[1] ... b[n] first_attempt
**
** Returns 47 if no solution found.
*/


class SolutionNotFound : public std::exception
{
};


/*
** Initialize the base matrix. Top left n×n matrix is identity, last
** column is -a[i], last vector's only nonzero element is M in the
** last column.
*/
void init_base_matrix(int n, const std::vector<NTL::ZZ> &a, const NTL::ZZ &m,
                      NTL::mat_ZZ &mat)
{
    mat.SetDims(n+1, n+1);
    for (int i = 0; i < n+1; ++i)
    {
        for (int j = 0; j < n+1; ++j)
        {
            mat[i][j] = 0;
        }
        if (i < n)
        {
            mat[i][i] = 1;
            mat[i][n] = -a[i];
        }
        else
        {
            mat[i][i] = m;
        }
    }
}


void print_matrix(const NTL::mat_ZZ &mat)
{
    for (int i = 0; i < mat.NumRows(); ++i)
    {
        for (int j = 0; j < mat.NumCols(); ++j)
        {
            std::cerr << mat[i][j] << ' ';
        }
        std::cerr << std::endl;
    }
}


int main()
{
    int n;
    NTL::ZZ m, total;
    total = 0;
    std::cin >> n >> m;

    std::vector<NTL::ZZ> a(n);
    for (int i = 0; i < n; ++i)
    {
        std::cin >> a[i];
        total += a[i];
    }

    std::vector<NTL::ZZ> m_opts = {m, total - m};
    for (NTL::ZZ curr_m : m_opts)
    {
        NTL::mat_ZZ mat;
        init_base_matrix(n, a, curr_m, mat);

        std::cerr << "Original matrix:" << std::endl;
        print_matrix(mat);

        NTL::ZZ det;
        NTL::LLL(det, mat);

        std::cerr << "Reduced matrix:" << std::endl;
        print_matrix(mat);

        for (int i = 0; i < n+1; ++i)
        {
            cerr << "trying row " << i << std::endl;
            try
            {
                NTL::ZZ lambda, selected_sum;
                for (int j = 0; j < n; ++j)
                {
                    if (lambda == 0 && mat[i][j] > 0 && mat[i][j] <= n)
                    {
                        lambda = mat[i][j];
                    }
                    if ((lambda != 0 && mat[i][j] > 0 && mat[i][j] != lambda)
                            || mat[i][j] < 0 || mat[i][j] > n)
                    {
                        cerr << lambda << ' ' << mat[i][j] << endl;
                        throw SolutionNotFound();
                    }
                    if (mat[i][j] != 0)
                    {
                        selected_sum += a[j];
                    }
                }
                if (selected_sum == curr_m)
                {
                    for (int j = 0; j < n; ++j)
                    {
                        std::cout << mat[i][j] / lambda << ' ';
                    }
                    std::cout << (int)(curr_m == m) << std::endl;
                    return 0;
                }
                else
                {
                    cerr << "sum error: " << selected_sum << ", expected "
                        << curr_m << endl;
                }
            }
            catch (SolutionNotFound e)
            {
                cerr << "mismatch" << endl;
            }
        }
    }
    return 47;
}
