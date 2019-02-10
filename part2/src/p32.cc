#include <iostream>
#include <fstream>
#include <mpi.h>
#include <cstring>
#include "complex.h"
#include "input_image.h"
#include "fft.h"
//#include <chrono>

int main(int argc, char **argv)
{
	
    //auto start = std::chrono::system_clock::now();

    if (argc != 4)
    {
        std::cout << "Incorrect usage" << std::endl;
        return -1;
    }

    // declare vars
    InputImage imageObj(argv[2]);
    int length, totalLength;
    Complex *image = nullptr;

    // values from input file
    length = imageObj.get_width();
    totalLength = length * length;
    image = imageObj.get_image_data();

    // start mpi
    int numtasks, rank, rc;
    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS)
    {
        std::cout << "Error starting MPI program. Terminating." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // create new MPI type for Complex
    int count = 2;
    const int arrBlockLength[] = {1, 1};
    const MPI_Aint arrDisplacement[] = {offsetof(Complex, real), offsetof(Complex, imag)};
    const MPI_Datatype arrTypes[] = {MPI_FLOAT, MPI_FLOAT};
    MPI_Datatype MPI_COMPLEX_TEMP, MY_MPI_COMPLEX;
    MPI_Aint lowerBound, extent;
    MPI_Type_create_struct(count, arrBlockLength, arrDisplacement, arrTypes, &MPI_COMPLEX_TEMP);
    MPI_Type_get_extent(MPI_COMPLEX_TEMP, &lowerBound, &extent);
    MPI_Type_create_resized(MPI_COMPLEX_TEMP, lowerBound, extent, &MY_MPI_COMPLEX);
    MPI_Type_commit(&MY_MPI_COMPLEX);

    // if (rank == 0) //debug
    // {
    //     std::cout << "Image length: " << length;
    //     std::cout << ", total length: " << totalLength << std::endl;
    // }

    // printf("Number of tasks: %d My rank: %d\n", numtasks, rank);

    //calculate row fft
    Complex *sendBuffer = (Complex *)malloc(sizeof(Complex) * length);
    int m, row;
    for (m = 0; m < length / numtasks; ++m)
    {
        row = rank + m * numtasks;
        // std::cout << "row: " << row << std::endl;
        std::memcpy(sendBuffer, image + row * length, length * sizeof(Complex));
        fft(sendBuffer, length);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgather(sendBuffer, length, MY_MPI_COMPLEX, image + m * numtasks * length, length, MY_MPI_COMPLEX, MPI_COMM_WORLD);
    }

    // synchronize at end of horizontal FFT
    MPI_Barrier(MPI_COMM_WORLD);

    // transpose for columns
    for (int i = 0; i < length; ++i)
        for (int j = i + 1; j < length; ++j)
            std::swap(image[length * i + j], image[length * j + i]);

    // fft of columns
    int n, column;
    for (n = 0; n < length / numtasks; ++n)
    {
        column = rank + n * numtasks;
        // std::cout << "column: " << row << std::endl;

        std::memcpy(sendBuffer, image + column * length, sizeof(Complex) * length);
        fft(sendBuffer, length);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(sendBuffer, length, MY_MPI_COMPLEX, image + n * numtasks * length, length, MY_MPI_COMPLEX, 0, MPI_COMM_WORLD);
    }

    // send everything to rank 0 and print
    if (rank == 0)
    {
        for (int i = 0; i < length; ++i)
            for (int j = i + 1; j < length; ++j)
                std::swap(image[length * i + j], image[length * j + i]);

	imageObj.save_image_data(argv[3], image, imageObj.get_width(), imageObj.get_height());

        //std::ofstream outFile;
        //outFile.open(argv[3]);

        //for (int i = 0; i < totalLength; ++i)
        //{
        //    outFile << image[i] << ' ';
        //    if ((i + 1) == 16)
        //        outFile << std::endl;
        //}
        //outFile.close();
    }
    // std::cout << "Rank " << rank << " exiting normally" << std::endl;
    free(sendBuffer);

    //if (rank == 0) {
    //    auto end = std::chrono::system_clock::now();
    //    std::chrono::duration<double> elapsed_seconds = end-start;
    //    std::cout << elapsed_seconds.count() << std::endl;
    //}


    MPI_Finalize();

    return 0;
}