#include <Eigen.h>
#include <iostream>

namespace factorizedEightPoint
{

using namespace Eigen;

void estimateFundamentalMatrix(std::vector<Vector2f>);
void fetchRotationTranslation(Matrix3f);

}