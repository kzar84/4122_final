Read FinalProject.pdf for information on the project

building project:

mkdir build

cd build

cmake ..

make

./p31 forward/reverse [INPUTFILE] [OUTPUTFILE]

mpirun -np 8 ./p32 forward/reverse [INPUTFILE] [OUTPUTFILE]

./p33 forward/reverse [INPUTFILE] [OUTPUTFILE]

tests are in ../build

make a new test with make_test.cc
