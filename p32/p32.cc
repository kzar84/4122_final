// Demonstrate simple MPI program
// George F. Riley, Georgia Tech, Fall 2011

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <iterator>
#include <unistd.h>
#include <mpi.h>

using namespace std;

int main(int argc, char **argv)
{
    double T1temp = atof(argv[1]);
    double T2temp = atof(argv[2]);
    int NumGridPoints = atoi(argv[3]);
    int NumTimesteps = atoi(argv[4]);
    int numtasks, rank, rc;

    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS)
    {
        cout << "Error starting MPI program. Terminating." << endl;
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("Number of tasks= %d My rank= %d\n", numtasks, rank);

    ofstream myfile;
    myfile.open("heat1Doutput.csv");

    int min_cells = NumGridPoints / numtasks;
    int extras = NumGridPoints % numtasks;
    int curr_cells = min_cells;
    if (rank < extras)
    {
        ++curr_cells;
    }

    double *arr1 = new double[curr_cells]();
    double *arr2 = new double[curr_cells]();
    double head = 0;
    double mid = 0;
    double end = 0;
    double buff;

    double *answer = nullptr;
    int *recvCount = nullptr;
    int *displacement = nullptr;

    if (rank == 0)
    {
        answer = new double[NumGridPoints]();
        recvCount = new int[numtasks]();
        displacement = new int[numtasks]();
        for (int i = 0; i < numtasks; i++)
        {
            if (i < extras)
            {
                recvCount[i] = min_cells + 1;
            }
            else
            {
                recvCount[i] = min_cells;
            }
        }

        for (int i = 1; i < numtasks; i++)
        {
            displacement[i] = displacement[i - 1] + recvCount[i - 1];
        }

        for (int i = 0; i < numtasks; i++)
        {
            cout << recvCount[i] << ",";
        }
        cout << endl;

        for (int i = 0; i < numtasks; i++)
        {
            cout << displacement[i] << ",";
        }
        cout << endl;
    }

    while (NumTimesteps > 0)
    {
        cout << "Rank " << rank << " at step: " << NumTimesteps << endl;
        for (int i = 0; i < curr_cells; i++)
        {
            cout << "Rank " << rank << " at cell: " << i << " of " << curr_cells - 1 << ", curr = " << arr1[i] << endl;
            head = 0.25 * arr1[i - 1];
            mid = 0.5 * arr1[i];
            end = 0.25 * arr1[i + 1];

            if (i == 0)
            {
                //mpi receive from rank - 1
                //if rank 0, get t1
                //send to rank + 1

                if (rank == 0)
                {
                    head = 0.25 * T1temp;
                }
                else
                {
                    buff = arr1[i];
                    MPI_Request request;
                    cout << "SEND ----- Rank " << rank << " sending to rank" << rank - 1 << " data: " << buff << endl;
                    rc = MPI_Send(&buff, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
                    if (rc != MPI_SUCCESS)
                    {
                        cout << "Rank " << rank << " send failed, rc " << rc << endl;
                        MPI_Finalize();
                        exit(1);
                    }

                    MPI_Status status;
                    rc = MPI_Recv(&buff, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
                    head = 0.25 * buff;
                    cout << "RECV ----- Rank " << rank << " received from rank " << rank - 1 << " data: " << buff << endl;
                    if (rc != MPI_SUCCESS)
                    {
                        cout << "Rank " << rank << " recv failed, rc " << rc << endl;
                        MPI_Finalize();
                        exit(1);
                    }
                }
            }
            if (i == curr_cells - 1)
            {
                //mpi receive from rank - 1
                //if rank max, get t2
                //send to rank - 1
                if (rank == numtasks - 1 || rank == NumGridPoints - 1)
                {
                    end = 0.25 * T2temp;
                }
                else if (rank < NumGridPoints - 1)
                {
                    buff = arr1[i];
                    MPI_Request request;
                    cout << "SEND ----- Rank " << rank << " sending to rank" << rank + 1 << " data: " << buff << endl;
                    rc = MPI_Send(&buff, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
                    if (rc != MPI_SUCCESS)
                    {
                        cout << "Rank " << rank << " send failed, rc " << rc << endl;
                        MPI_Finalize();
                        exit(1);
                    }
                    MPI_Status status;
                    cout << "WaitRECV - Rank " << rank << " from rank " << rank + 1 << endl;
                    rc = MPI_Recv(&buff, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &status);
                    end = 0.25 * buff;
                    cout << "RECV ----- Rank " << rank << " received from rank " << rank + 1 << " data: " << buff << endl;
                    if (rc != MPI_SUCCESS)
                    {
                        cout << "Rank " << rank << " recv failed, rc " << rc << endl;
                        MPI_Finalize();
                        exit(1);
                    }
                }
            }
            arr2[i] = head + mid + end;

            cout << "arr2[" << i << "]: " << arr2[i] << endl;
        }
        copy(arr2, arr2 + curr_cells, arr1);
        --NumTimesteps;
    }

    MPI_Gatherv(arr2, curr_cells, MPI_DOUBLE, answer, recvCount, displacement, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    delete[] arr1;
    delete[] arr2;
    if (rank == 0)
    {
        int i;
        for (i = 0; i < NumGridPoints - 1; i++)
        {
            myfile << answer[i] << ",";
        }
        myfile << answer[i];

        delete[] answer;
        delete[] recvCount;
        delete[] displacement;
    }

    cout << "Rank " << rank << " exiting normally" << endl;
    MPI_Finalize();
}