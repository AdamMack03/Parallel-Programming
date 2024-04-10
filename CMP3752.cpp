#include <vector>
#include <fstream>
#include <string>

#include "Utils.h"

typedef unsigned char uchar;
typedef unsigned int uint;

struct PGMImage {
    std::string magicNumber;
    std::string comment;
    int width;
    int height;
    int maxGray;
    std::vector<std::vector<char>> pixels;
};

void print_help() {
    std::cerr << "Application usage:" << std::endl;

    std::cerr << "  -p : select platform " << std::endl;
    std::cerr << "  -d : select device" << std::endl;
    std::cerr << "  -l : list all platforms and devices" << std::endl;
    std::cerr << "  -h : print this message" << std::endl;
}

PGMImage readPGM(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    PGMImage image;
    std::string line;

    // Read magic number
    std::getline(file, image.magicNumber);

    // Read comment if present
    std::getline(file, line);
    if (line.find('#') != std::string::npos) {
        image.comment = line.substr(1); // Exclude the '#' character
        std::getline(file, line); // Read next line
    }

    // Read width and height
    std::stringstream ss(line);
    ss >> image.width >> image.height;

    std::cout << image.width << " " << image.height << std::endl;

    // Read maximum gray value
    file >> image.maxGray;

    // Allocate memory for image
    image.pixels.resize(image.height, std::vector<char>(image.width));

    // Read pixel values
    string pixelValues;
    file >> pixelValues;

    int pixelIndex = 0;
    for (int i = 0; i < image.height; ++i) {
        for (int j = 0; j < image.width; ++j) {
            // Store pixel value as char
            image.pixels[i][j] = pixelValues[pixelIndex++];
        }
    }

    file.close();
    return image;
}

void writePGM(const std::string& filename, const std::vector<uchar>& imageData, int width, int height) {
    std::ofstream outFile(filename, std::ios::out | std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Couldn't open output file " << filename << " for writing!" << std::endl;
        return;
    }
    // Write the PGM header
    outFile << "P5\n" << width << " " << height << "\n255\n";

    // Write image data
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            // Ensure the pixel values are within range [0, 255]
            uchar pixelValue = (imageData[i * width + j] > 255) ? 255 : ((imageData[i * width + j] < 0) ? 0 : imageData[i * width + j]);
            outFile << static_cast<char>(pixelValue);
        }
    }

    outFile.close();
    std::cout << "PGM file saved as " << filename << std::endl;
}

int main(int argc, char** argv) {

    int platform_id = 0;
    int device_id = 0;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
        else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
        else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
    }

    try {
        cl::Context context = GetContext(platform_id, device_id);
        std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

        cl::CommandQueue queue(context);

        cl::Program::Sources sources;
        AddSources(sources, "C:\\Users\\adam0\\Desktop\\Uni\\Year 3\\Parallel Programming\\Assignment\\CMP3752\\kernels\\kernel.cl");

        cl::Program program(context, sources);

        try {
            program.build();
        }
        catch (const cl::Error& err) {
            std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            throw err;
        }

        // Get image size
        std::string iFile = "C:\\Users\\adam0\\Desktop\\Uni\\Year 3\\Parallel Programming\\Assignment\\CMP3752\\images\\test.pgm";
        PGMImage image = readPGM(iFile);
        size_t imageSize = image.width * image.height;
        std::vector<uchar> inputImage;
        for (const auto& row : image.pixels) {
            for (const auto& pixel : row) {
                inputImage.push_back(static_cast<uchar>(pixel));
            }
        }
        std::vector<uchar> outputImage(imageSize);

        // Allocate memory buffers on the device
        cl::Buffer inputImageBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uchar) * imageSize, inputImage.data());
        cl::Buffer outputImageBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uchar) * imageSize);
        cl::Buffer histogramBuffer(context, CL_MEM_READ_WRITE, sizeof(uint) * 256);
        cl::Buffer cumulativeHistogramBuffer(context, CL_MEM_READ_WRITE, sizeof(uint) * 256);
        cl::Buffer lutBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uchar) * 256);

        // Set kernel arguments
        cl::Kernel calculateHistogramKernel(program, "calculateHistogram");
        calculateHistogramKernel.setArg(0, inputImageBuffer);
        calculateHistogramKernel.setArg(1, histogramBuffer);
        calculateHistogramKernel.setArg(2, static_cast<cl_uint>(imageSize));

        cl::Kernel calculateCumulativeHistogramKernel(program, "calculateCumulativeHistogram");
        calculateCumulativeHistogramKernel.setArg(0, histogramBuffer);
        calculateCumulativeHistogramKernel.setArg(1, cumulativeHistogramBuffer);

        cl::Kernel normalizeAndScaleCumulativeHistogramKernel(program, "normalizeAndScaleCumulativeHistogram");
        normalizeAndScaleCumulativeHistogramKernel.setArg(0, cumulativeHistogramBuffer);
        normalizeAndScaleCumulativeHistogramKernel.setArg(1, lutBuffer);
        normalizeAndScaleCumulativeHistogramKernel.setArg(2, static_cast<cl_uint>(imageSize));

        cl::Kernel backProjectionKernel(program, "backProjection");
        backProjectionKernel.setArg(0, inputImageBuffer);
        backProjectionKernel.setArg(1, outputImageBuffer);
        backProjectionKernel.setArg(2, lutBuffer);
        backProjectionKernel.setArg(3, static_cast<cl_uint>(imageSize));

        // Write input data to the device
        queue.enqueueWriteBuffer(inputImageBuffer, CL_TRUE, 0, sizeof(uchar) * imageSize, inputImage.data());

        // Execute kernels
        queue.enqueueNDRangeKernel(calculateHistogramKernel, cl::NullRange, cl::NDRange(imageSize), cl::NullRange);
        queue.enqueueNDRangeKernel(calculateCumulativeHistogramKernel, cl::NullRange, cl::NDRange(256), cl::NullRange);
        queue.enqueueNDRangeKernel(normalizeAndScaleCumulativeHistogramKernel, cl::NullRange, cl::NDRange(256), cl::NullRange);
        queue.enqueueNDRangeKernel(backProjectionKernel, cl::NullRange, cl::NDRange(imageSize), cl::NullRange);

        // Declare a vector to store histogram values
        std::vector<uint> histogram(256);

        // Read histogram data from the device
        queue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, sizeof(uint) * 256, histogram.data());

        // Output histogram values
        std::cout << "Histogram Values:" << std::endl;
        for (int i = 0; i < 256; ++i) {
            std::cout << "Intensity " << i << ": " << histogram[i] << std::endl;
        }

        std::ofstream histogramFile("histogram_data.txt");
        for (int i = 0; i < 256; ++i) {
            histogramFile << i << " " << histogram[i] << std::endl;
        }
        histogramFile.close();

        // Repeat for the other two histograms

        // Cumulative Histogram
        std::vector<uint> cumulativeHistogram(256);
        queue.enqueueReadBuffer(cumulativeHistogramBuffer, CL_TRUE, 0, sizeof(uint) * 256, cumulativeHistogram.data());
        std::cout << "Cumulative Histogram Values:" << std::endl;
        for (int i = 0; i < 256; ++i) {
            std::cout << "Intensity " << i << ": " << cumulativeHistogram[i] << std::endl;
        }
        std::ofstream cumulativeHistogramFile("cumulative_histogram_data.txt");
        for (int i = 0; i < 256; ++i) {
            cumulativeHistogramFile << i << " " << cumulativeHistogram[i] << std::endl;
        }
        cumulativeHistogramFile.close();

        //Normalize and scale cumulative histogram
        std::vector<uchar> lut(256);
        queue.enqueueReadBuffer(lutBuffer, CL_TRUE, 0, sizeof(uchar) * 256, lut.data());
        std::ofstream lutFile("lut_data.txt");
        for (int i = 0; i < 256; ++i) {
            lutFile << i << " " << static_cast<int>(lut[i]) << std::endl;
        }
        lutFile.close();

        std::vector<uchar> outputImageData(imageSize);
        queue.enqueueReadBuffer(outputImageBuffer, CL_TRUE, 0, sizeof(uchar) * imageSize, outputImageData.data());
        writePGM("output.pgm", outputImageData, image.width, image.height);
    }

    catch (cl::Error err) {
        std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
    }

    return 0;
}