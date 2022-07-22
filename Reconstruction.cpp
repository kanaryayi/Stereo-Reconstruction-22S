#include "Reconstruction.h"

std::ostream &operator<<(std::ostream &os, const Vertex &v) {
	if (v.position[0] == MINF || v.position[1] == MINF || v.position[2] == MINF) {
		os << "0.0 0.0 0.0 0 0 0\n";
	} else {

		os	<< v.position[0] << " " 
			<< v.position[1] << " " 
			<< v.position[2] << " "
			<< (int)v.color[0] << " "
			<< (int)v.color[1] << " "
		   	<< (int)v.color[2] << "\n";
	}

	return os;
}

bool Mesh::writeMesh(Vertex* vertices, int width, int height, const std::string& filename) {
	float edgeThreshold = 0.01f; // 1cm
	unsigned int nVertices = width * height;

	// Maximum number of faces
	// One is missing (rightmost column, bottom row)
	// and for each vertex, there are 2 triangles
	// 		=> hence: (width - 1) * (height - 1) * 2
	unsigned nFaces = (width - 1) * (height - 1) * 2;

	std::vector<Eigen::Vector3i> faces{nFaces};
	int faceCount = 0;

	int idx = 0;
	for (int x = 0; x < (width - 1); x++) {
		for (int y = 0; y < (height - 1); y++, idx++) {
			int current = idx;
			int bottom = idx + width;
			int right = idx + 1;
			int diag = idx + width + 1;

			// Left upper face
			if (Mesh::valid_triangle(vertices, current, bottom, right, edgeThreshold)) {
				faces[faceCount++] = Vector3i{current, bottom, right};
			}

			// Right lower face
			if (Mesh::valid_triangle(vertices, bottom, diag, right, edgeThreshold)) {
				faces[faceCount++] = Vector3i{bottom, diag, right};
			}
		}
	}

	nFaces = faceCount;

	// Write off file
	std::ofstream outFile(filename);
	if (!outFile.is_open())
		return false;

	// write header
	outFile << "COFF" << std::endl;
	outFile << "# numVertices numFaces numEdges" << std::endl;
	outFile << nVertices << " " << nFaces << " 0" << std::endl;

	for (int i = 0; i < width * height; i++) {
		outFile << vertices[i];
	}

	
	for (int i = 0; i < nFaces; i++) {
		outFile << "3 " 
				<< faces[i][0] << " "
				<< faces[i][1] << " "
				<< faces[i][2] << "\n";
	}

	outFile.close();
	return true;
}