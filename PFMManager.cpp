#include "PFMManager.h"

cv::Mat PFMManager::loadPFM(const std::string filePath) {
    // Open binary file
    std::ifstream file(filePath.c_str(), std::ios::in | std::ios::binary);

    cv::Mat imagePFM;

    // If file correctly openened
    if (file)
    {
        // Read the type of file plus the 0x0a UNIX return character at the end
        char type[3];
        file.read(type, 3 * sizeof(char));

        // Read the width and height
        unsigned int width(0), height(0);
        file >> width >> height;

        // Read the 0x0a UNIX return character at the end
        char endOfLine;
        file.read(&endOfLine, sizeof(char));

        int numberOfComponents(0);
        // The type gets the number of color channels
        if (type[1] == 'F')
        {
            imagePFM = cv::Mat(height, width, CV_32FC3);
            numberOfComponents = 3;
        }
        else if (type[1] == 'f')
        {
            imagePFM = cv::Mat(height, width, CV_32FC1);
            numberOfComponents = 1;
        }

        // KEEP BYTE ORDER IN MIND
        // ONLY WORKS ON LITTLE ENDIAN
        char byteOrder[4];
        file.read(byteOrder, 4 * sizeof(char));

        // Find the last line return 0x0a before the pixels of the image
        char findReturn = ' ';
        while (findReturn != 0x0a)
        {
            file.read(&findReturn, sizeof(char));
        }

        // Read each RGB colors as 3 floats and store it in the image.
        float *color = new float[numberOfComponents];
        for (unsigned int i = 0; i < height; ++i)
        {
            for (unsigned int j = 0; j < width; ++j)
            {
                file.read((char *)color, numberOfComponents * sizeof(float));

                // In the PFM format the image is upside down
                if (numberOfComponents == 3)
                {
                    // OpenCV stores the color as BGR
                    imagePFM.at<cv::Vec3f>(height - 1 - i, j) = cv::Vec3f(color[2], color[1], color[0]);
                }
                else if (numberOfComponents == 1)
                {
                    // OpenCV stores the color as BGR
                    imagePFM.at<float>(height - 1 - i, j) = color[0];
                }
            }
        }

        delete[] color;

        // Close file
        file.close();
    }
    else
    {
        std::cerr << "Could not open the file : " << filePath << std::endl;
    }

    return imagePFM;
}