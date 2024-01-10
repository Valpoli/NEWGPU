#include "ex2.h"

int main()
{
    {
        std::cout << "Test 1" << std::endl;
        // N = M = BLOCK_SIZE = 8
        const std::vector<int> A = {
            1, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 1};
        const std::vector<int> b = {1, 2, 3, 4, 5, 6, 7, 8};
        const std::vector<int> c_sol = {1, 2, 3, 4, 5, 6, 7, 8};
        const std::vector<int> c = matvecmul2(A, b);
        if(c != c_sol)
        {
            std::cout << "Error, expected:" << std::endl;
            print(c_sol);
            std::cout << "got:" << std::endl;
            print(c);
        }
        else
        {
            std::cout << "Ok" << std::endl;
        }
    }
    {
        std::cout << "Test 2" << std::endl;
        // N =  8 = BLOCK_SIZE
        // M = 16 = BLOCK_SIZE * 2
        const std::vector<int> A = {
            1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1};
        const std::vector<int> b = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        const std::vector<int> c_sol = {10, 12, 14, 16, 18, 20, 22, 24};
        const std::vector<int> c = matvecmul2(A, b);
        if(c != c_sol)
        {
            std::cout << "Error, expected:" << std::endl;
            print(c_sol);
            std::cout << "got:" << std::endl;
            print(c);
        }
        else
        {
            std::cout << "Ok" << std::endl;
        }
    }

    return 0;
}
