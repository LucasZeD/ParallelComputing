# Implementação e árvore de decisão, ID3, paralela.

# Data-set possui objetivo e prever a nota de um aluno de acordo com:
	- dados demográficos categóricos.
	- dados categóricos de participação acadêmica.
	- dados categóricos de fatores sociais.
	- dados categóricos e estilo de vida.

## Data-set:
	https://www.kaggle.com/datasets/atifmasih/factors-affecting-university-student-grades
	StudentData.csv				-> Original data-set
	PreprocessesData/*.csv			-> data-set pre processado
	PreprocessesData/Oversampled/*.csv	-> data-set com oversample
	PreprocessesData/Oversampled/*.txt	-> Data read by *.cpp files
	basta rodar o .ipynb para popular as pastas, foram apagadas para poupar armazenamento

# Código ID3 original:
	https://gist.github.com/nandor/d6002cd9a5f674cd550a/718b0f27aa5f14d31db34efc1451a2b5c532997f
	
# Rodando o programas:

O data-set oversampled foi transformado nos txts, teste e train.

Sequencial:
	Compilar: g++ -o id3 main.cpp
	Excutar: time ./id3

OpenMP:
	Compilar: g++ -o id3_open OpenMP.cpp -fopenmp
	Executar: time ./id3_open <threads>

MPI:
	Compilar: mpicxx -o id3_mpi mpiopenmp.cpp -fopenmp
	Executar: time mpirun -np <processes> ./id3_mpi <processes> <threads>