#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H

#include <iostream>
#include <fstream>
#include <array>
#include <vector>

#include "Eigen.h"
#include "Utils.h"

/***
 *  Reconstruction with .off format mesh files
 *  Reference 3D Scanning and Motion capture SS22 TUM Lecture materials
 * ***/
struct Vertex {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// position stored as 4 floats (4th component is supposed to be 1.0)
	Vector4f position;
	// color stored as 4 unsigned char
	Vector4uc color;
};

class Mesh {
	public:
		static bool writeMesh(Vertex* vertices, int width, int height, const std::string& filename);

	private:
		// Checks for the given vertices, if the triangle formed by them is valid.
		// This is only the case if every edge is smaller than some threshold value.
		static bool valid_triangle(Vertex *vertices, int i0, int i1, int i2, float edgeThreshold) {
			return (vertices[i0].position - vertices[i1].position).norm() < edgeThreshold
				&& (vertices[i0].position - vertices[i2].position).norm() < edgeThreshold
				&& (vertices[i1].position - vertices[i2].position).norm() < edgeThreshold;
		}
};

#endif