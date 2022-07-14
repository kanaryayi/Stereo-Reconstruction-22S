#include "Reconstruction.h"
/***
 *  Reconstruction with .off format mesh files
 *  Reference 3D Scanning and Motion capture SS22 TUM Lecture materials
 * ***/
struct Vertex
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// position stored as 4 floats (4th component is supposed to be 1.0)
	Vector4f position;
	// color stored as 4 unsigned char
	Vector4uc color;
};
double euclidianDist(Vertex a, Vertex b)
{
	double dist = abs(((Vector4f)(a.position-b.position)).norm());
	//std::cout << dist << " " << a.position.transpose() << " "<< b.position.transpose() << " " << std::endl;
	return dist;
} 
bool isTriangle(Vertex& a, Vertex b, Vertex c, double edgeThreshold)
{
	double dist;
	
	//std::cout << a.position.transpose() << b.position.transpose() << c.position.transpose() << std::endl; 
	if (a.position[0] == MINF or a.position[1] == MINF or a.position[2] == MINF or a.position[2] == 0) // point is not valid
	{
		a.position = Vector4f(0,0,0,0);
		a.color = Vector4uc(0,0,0,0);
		return false;
	}
	if (b.position[0] == MINF or b.position[1] == MINF or b.position[2] == MINF or b.position[2] == 0) return false; // point is not valid
    if (c.position[0] == MINF or c.position[1] == MINF or c.position[2] == MINF or c.position[2] == 0) return false; // point is not valid
	else // valid
	{
		dist= euclidianDist(a,b);
		//std::cout << dist << std::endl;
		if(dist >= edgeThreshold ) // distance over the threshold
		{
			return false;
		}
		dist = euclidianDist(b,c);
		//std::cout << dist << std::endl;
		if(dist >= edgeThreshold ) // distance over the threshold
		{
			return false;
		}	
		dist = euclidianDist(a,c);
		//std::cout << dist << std::endl;
		if(dist >= edgeThreshold ) // distance over the threshold
		{
			return false;
		}
		//std::cout << a.position.transpose() << b.position.transpose() << c.position.transpose() << std::endl; 
	} 
	return true;
}
bool WriteMesh(Vertex* vertices, unsigned int width, unsigned int height)
{
	float edgeThreshold = 0.01f; // 1cm
	
	// Each grid is processed individually with the two triangles it has. 
	std::vector<unsigned int*> faces;

	for(int j = 0; j < height; j++)
	{
		unsigned int idxj = j;
		for(int i = 0; i < width; i++)
		{
			unsigned int idxi = i;
			if (j == height-1 or i == width -1) // point is not valid
			{
				Vertex *lastVertex = &vertices[idxj * width + idxi];
				lastVertex->position[0] = 0;
				lastVertex->position[1] = 0;
				lastVertex->position[2] = 0;
				lastVertex->position[3] = 0;	
				continue;
			}
			// first triangle
			unsigned int *face = new unsigned int[3];
			face[0] = idxj * width + idxi;
			face[1] = (idxj + 1) * width + idxi;
			face[2] = idxj * width + idxi + 1;
			if(isTriangle(vertices[face[0]], vertices[face[1]], vertices[face[2]], (double) edgeThreshold))	
			{
				faces.push_back(face);
			}

			// second triangle
			unsigned int *face2 = new unsigned int[3];
			face2[0] = (idxj + 1) * width + idxi;
			face2[1] = (idxj + 1) * width + idxi + 1;
			face2[2] = idxj * width + idxi + 1;
			if(isTriangle(vertices[face2[1]], vertices[face2[0]], vertices[face2[2]], (double) edgeThreshold))
			{
				faces.push_back(face2);
			}		
		}
	}
	
	// number of vertices
	unsigned int nVertices = width * height;

	// number of valid faces
	unsigned nFaces = faces.size();


	// Write off file
	std::ofstream outFile("reconst.off");
	if (!outFile.is_open()) return false;

	// write header
	outFile << "COFF" << std::endl;

	outFile << "# numVertices numFaces numEdges" << std::endl;

	outFile << nVertices << " " << nFaces << " 0" << std::endl;

	// save vertices
	outFile << "# list of vertices" << std::endl << "# X Y Z R G B A" << std::endl;
	for(int i = 0; i<nVertices; i++)
	{
		Vector4f pos = vertices[i].position;
		Vector4uc color = vertices[i].color;
		outFile << pos[0] << " " << pos[1] << " " << pos[2];
		outFile << " " << (unsigned int) color[0]<< " " << (unsigned int) color[1]<< " " << (unsigned int) color[2]<< " " << (unsigned int) color[3] <<std::endl;	
	}
	
	// TODO: save valid faces
	outFile << "# list of faces" << std::endl;
	outFile << "# nVerticesPerFace idx0 idx1 idx2 ..." << std::endl;

	for(int i = 0; i<nFaces; i++)
	{
		outFile << "3 " << faces[i][0] << " " << faces[i][1] << " "<< faces[i][2] << std::endl;
		delete[] faces[i];
	}

	// close file
	outFile.close();

	return true;
}