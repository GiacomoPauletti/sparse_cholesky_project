#include "cholesky.h"
#include <iostream>

CholeskyTree::CholeskyTree() 
: parentship{0}
{
}

CholeskyTree::CholeskyTree(int num_nodes) 
: parentship(num_nodes, 0)
{
}

int& CholeskyTree::operator[](int index) {
    return this->parentship[index];
}

int& CholeskyTree::parent(int index) {
    return this->parentship[index];
}