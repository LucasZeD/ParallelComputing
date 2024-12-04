/*
Processos   Threads   real      user       sys
1           4         36,257    95,696     6,236
2           2         40,368    117,597    5,356
4           1         41,293    139,947    22,802
*/

#include <cmath>
#include <cstdlib>
#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <omp.h> // Incluindo a biblioteca OpenMP
#include <mpi.h> // Incluindo a biblioteca MPI

using namespace std;

template <size_t N>
class ID3;
template <size_t N>
class ID3Train;

constexpr size_t N = 27;
constexpr float EPS = 1e-7;

template <size_t N>
struct Sample
{
	Sample() {}

	Sample(Sample &&sample)
		: attributes(move(sample.attributes)), clazz(move(sample.clazz)) {}

	void operator=(Sample &&sample)
	{
		attributes = move(sample.attributes);
		clazz = move(sample.clazz);
	}

	array<string, N> attributes;
	string clazz;
};

template <size_t N>
class ID3
{
public:
	string classify(const array<string, N> &sample)
	{
		return root_->classify(sample);
	}

	void print(ostream &os)
	{
		root_->print(os, 0);
	}

private:
	class Node
	{
	public:
		virtual ~Node() {};
		virtual string classify(const array<string, N> &sample) = 0;
		virtual void print(ostream &os, size_t level) = 0;
	};

	class TerminalNode : public Node
	{
	public:
		TerminalNode(const string &clazz)
			: clazz_(clazz) {}

		string classify(const array<string, N> &sample) override
		{
			return clazz_;
		}

		void print(ostream &os, size_t level) override
		{
			for (size_t i = 0; i < level; ++i)
			{
				os << ' ';
			}
			os << clazz_ << endl;
		}

	private:
		string clazz_;
	};

	class InnerNode : public Node
	{
	public:
		InnerNode(
			size_t attribute,
			string &&clazz,
			unordered_map<string, unique_ptr<Node>> &&branches)
			: attribute_(attribute), clazz_(clazz), branches_(move(branches)) {}

		string classify(const array<string, N> &sample) override
		{
			auto it = branches_.find(sample[attribute_]);
			if (it == branches_.end())
			{
				return clazz_;
			}
			else
			{
				return it->second->classify(sample);
			}
		}

		void print(ostream &os, size_t level) override
		{
			auto tabs = [&os, level]
			{
				for (size_t i = 0; i < level; ++i)
				{
					os << ' ';
				}
			};
			for (const auto &branch : branches_)
			{
				tabs();
				os << branch.first << ":" << endl;
				branch.second->print(os, level + 1);
			}
		}

	private:
		size_t attribute_;
		string clazz_;
		unordered_map<string, unique_ptr<Node>> branches_;
	};

	ID3(unique_ptr<Node> &&root)
		: root_(move(root)) {}

	unique_ptr<Node> root_;

	template <size_t M>
	friend class ID3Train;
};

template <size_t N>
class ID3Train
{
private:
	using ID3 = typename ::ID3<N>;
	using Node = typename ::ID3<N>::Node;
	using TerminalNode = typename ::ID3<N>::TerminalNode;
	using InnerNode = typename ::ID3<N>::InnerNode;
	using Sample = typename ::Sample<N>;
	using Iter = typename vector<typename ::Sample<N>>::iterator;

public:
	ID3Train(vector<Sample> &&samples)
		: samples_(move(samples)) {}

	unique_ptr<ID3> train()
	{
		return unique_ptr<ID3>(new ID3(train(samples_.begin(), samples_.end())));
	}

private:
	unique_ptr<Node> train(Iter start, Iter end)
	{
		auto ig = make_pair(0, numeric_limits<float>::min());
		unordered_map<string, size_t> clazzes;
		string maxClazz;

		array<unordered_map<string, unordered_map<string, size_t>>, N> count;

#pragma omp parallel for // Aplicando OpenMP para paralelizar o loop
		for (auto it = start; it < end; ++it)
		{

			{
				for (size_t i = 0; i < N; ++i)
#pragma omp critical // Garantindo que incrementos a 'clazz' sejam feitos de forma segura
				{
					count[i][it->attributes[i]][it->clazz]++;
				}
#pragma omp critical // Garantindo que incrementos a 'clazzes' sejam feitos de forma segura
				clazzes[it->clazz]++;
			}
		}

		auto entropy = 0.0f;
		auto total = end - start;
		for (auto clazz : clazzes)
		{
			auto p = clazz.second / (float)total;
			entropy -= p * log(p) / log(2.0f);

			if (!maxClazz.empty() && clazz.second > clazzes[maxClazz])
			{
				maxClazz = clazz.first;
			}
		}

		if (abs(entropy) <= EPS)
		{
			return make_unique<TerminalNode>(start->clazz);
		}

#pragma omp parallel for // Aplicando OpenMP para paralelizar o loop
		for (size_t i = 0; i < N; ++i)
		{
			auto attribIG = entropy;
			for (auto split : count[i])
			{
				auto setTotal = 0;
				for (auto clazz : split.second)
				{
					setTotal += clazz.second;
				}

				auto setEntropy = 0.0f;
				for (auto clazz : split.second)
				{
					auto p = clazz.second / (float)setTotal;
					setEntropy -= p * log(p) / log(2.0f);
				}
#pragma omp critical // Garantindo que decrementos a 'attribIG' sejam feitos de forma segura
				{
					attribIG -= (float)setTotal / (float)total * setEntropy;
				}
			}
			if (attribIG >= ig.second)
			{
				ig.first = i;
				ig.second = attribIG;
			}
		}

		auto attribIndex = ig.first;
		sort(start, end, [attribIndex](const Sample &a, const Sample &b)
			 { return a.attributes[attribIndex] < b.attributes[attribIndex]; });

		auto setStart = start;
		unordered_map<string, unique_ptr<Node>> nodes;
		for (auto it = start + 1; it != end + 1; ++it)
		{
			if (it < end && it->attributes[attribIndex] == setStart->attributes[attribIndex])
			{
				continue;
			}
			nodes[setStart->attributes[attribIndex]] = train(setStart, it);
			setStart = it;
		}

		return make_unique<InnerNode>(attribIndex, move(maxClazz), move(nodes));
	}

	vector<Sample> samples_;
};

int main(int argc, char **argv)
{

	MPI_Init(&argc, &argv); // Inicializando MPI

	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Obtendo o número total de processos
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Obtendo o rank do processo atual

	vector<Sample<N>> samples;

	int num_threads = omp_get_max_threads(); // Definindo o número padrão de threads como o máximo possível

	if (argc >= 3)
	{
		world_size = atoi(argv[1]);	 // Definindo o número de processos pela linha de comando
		num_threads = atoi(argv[2]); // Permitindo que o usuário defina o número de threads pela linha de comando
	}

	omp_set_num_threads(num_threads); // Configurando o número de threads do OpenMP

	if (world_rank == 0)
	{
		// Processo mestre lê os dados
		//ifstream is("train.txt");
		ifstream is("PrepData/Stdnt_Oversampled_Train.txt");
		while (!is.eof())
		{
			Sample<N> sample;
			for (auto i = 0; i < N; ++i)
			{
				if (!(is >> sample.attributes[i]))
				{
					break;
				}
			}
			if (!(is >> sample.clazz))
			{
				break;
			}
			samples.push_back(move(sample));
		}
	}

	// Transmitir número de amostras para todos os processos
	int num_samples = samples.size();
	MPI_Bcast(&num_samples, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (world_rank != 0)
	{
		// Processo trabalhador aloca memória para as amostras
		samples.resize(num_samples);
	}

	// Transmitir as amostras para todos os processos
	for (auto &sample : samples)
	{
		for (auto &attr : sample.attributes)
		{
			int attr_size;
			if (world_rank == 0)
			{
				attr_size = attr.size();
			}
			MPI_Bcast(&attr_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
			if (world_rank != 0)
			{
				attr.resize(attr_size);
			}
			MPI_Bcast(&attr[0], attr_size, MPI_CHAR, 0, MPI_COMM_WORLD);
		}

		int clazz_size;
		if (world_rank == 0)
		{
			clazz_size = sample.clazz.size();
		}
		MPI_Bcast(&clazz_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (world_rank != 0)
		{
			sample.clazz.resize(clazz_size);
		}
		MPI_Bcast(&sample.clazz[0], clazz_size, MPI_CHAR, 0, MPI_COMM_WORLD);
	}

	auto id3 = ID3Train<N>(move(samples)).train();

	{
		vector<array<string, N>> samples_to_classify;
		//ifstream is("teste.txt");
		ifstream is("PrepData/Stdnt_Oversampled_Test.txt");
		while (!is.eof())
		{
			array<string, N> sample;
			for (auto i = 0; i < N; ++i)
			{
				if (!(is >> sample[i]))
				{
					break;
				}
			}
			if (is)
			{
				samples_to_classify.push_back(sample);
			}
		}

		vector<string> results(samples_to_classify.size());

		// Dividir trabalho entre processos MPI
		int chunk_size = samples_to_classify.size() / world_size;
		int start = world_rank * chunk_size;
		int end = (world_rank == world_size - 1) ? samples_to_classify.size() : start + chunk_size;

		for (int i = start; i < end; ++i)
		{
			results[i] = id3->classify(samples_to_classify[i]);
		}

		// Processo mestre coleta os resultados
		if (world_rank == 0)
		{
			vector<string> final_results(samples_to_classify.size());

			for (int i = 0; i < world_size; ++i)
			{
				if (i != 0)
				{
					int recv_start = i * chunk_size;
					int recv_end = (i == world_size - 1) ? samples_to_classify.size() : recv_start + chunk_size;
					MPI_Status status;
					MPI_Recv(&final_results[recv_start], recv_end - recv_start, MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
				}
				else
				{
					copy(results.begin() + start, results.begin() + end, final_results.begin() + start);
				}
			}

			// Imprime os resultados
			// for (const auto &result : final_results)
			// {
			// 	cout << result << endl;
			// }
		}
		else
		{
			MPI_Send(&results[start], (end - start) * results[0].size(), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
		}
	}

	MPI_Finalize(); // Finalizando MPI
	return EXIT_SUCCESS;
}
