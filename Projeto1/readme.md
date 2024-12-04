# Implementation - Decision Tree, ID3, Parallel.

# Data-set's objective:

Predict the grade of a student based on:  
- Categorical demographic data.  
- Categorical academic participation data.  
- Categorical social factors data.  
- Categorical lifestyle and style data.  

## Data-set:

1. Data-set Link:
*[Original Data-set - Atif Masih](https://www.kaggle.com/datasets/atifmasih/factors-affecting-university-student-grades)  
2. Folders:
- `StudentData.csv`				-> Original data-set  
- `PrepData/*.csv`				-> Pre-processed data-set files  
- `PrepData/Stdnt_Oversampled_Train/Test.txt`	-> Oversampled data read by the *.cpp files  
3. Data Preprocessing and Oversampling:  
- Read the `Data-set/readme.md` to populate the folders. These were deleted to save storage.  

# Código ID3 original:

*[ID3 original code - Nandor Licker](https://gist.github.com/nandor/d6002cd9a5f674cd550a/718b0f27aa5f14d31db34efc1451a2b5c532997f)  
	
# Rodando o programas:

The oversampled data was converted into text files: `Stdnt_Oversampled_<Test||Train>.txt`.  

Sequential:
- Compile: `g++ -o id3.exe main.cpp`  
- Run: `time ./id3`  

OpenMP:
- Compile: `g++ -o id3_open.exe OpenMP.cpp -fopenmp`  
- Run: `time ./id3_open <threads>`  

MPI:
- Compile: `mpicxx -o id3_mpi.exe mpiopenmp.cpp -fopenmp`  
- Run: `time mpirun -np <processes> ./id3_mpi <processes> <threads>`  