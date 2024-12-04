/*
real      user      sys
31,563    28,416    3,157
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

using namespace std;

/// Forward declarations.
template <size_t N>
class ID3;
template <size_t N>
class ID3Train;

/// Number of attributes in each sample.
constexpr size_t N = 27;
/// Floating point error.
constexpr float EPS = 1e-7;

/**
 * A single sample with attributes & target class.
 */
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

	// Array of attributes of the sample.
	array<string, N> attributes;
	// Class to which the sample belongs.
	string clazz;
};

/**
 * ID3 tree.
 */
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
			: clazz_(clazz)
		{
		}

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

	/// Inner node that makes a decision.
	class InnerNode : public Node
	{
	public:
		InnerNode(
			size_t attribute,
			string &&clazz,
			unordered_map<string, unique_ptr<Node>> &&branches)
			: attribute_(attribute), clazz_(clazz), branches_(move(branches))
		{
		}

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
		: root_(move(root))
	{
	}

	/// Root node of the decision tree.
	unique_ptr<Node> root_;

	/// Only the trainer can construct this class.
	template <size_t M>
	friend class ID3Train;
};

/**
 * ID3 Trainer.
 */
template <size_t N>
class ID3Train
{
private:
	// Some shorthand aliases.
	using ID3 = typename ::ID3<N>;
	using Node = typename ::ID3<N>::Node;
	using TerminalNode = typename ::ID3<N>::TerminalNode;
	using InnerNode = typename ::ID3<N>::InnerNode;
	using Sample = typename ::Sample<N>;
	using Iter = typename vector<typename ::Sample<N>>::iterator;

public:
	ID3Train(vector<Sample> &&samples)
		: samples_(move(samples))
	{
	}

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

		// For each attribute/value pair, compute how many items fall into that category.
		array<unordered_map<string, unordered_map<string, size_t>>, N> count;
		for (auto it = start; it != end; ++it)
		{
			for (size_t i = 0; i < N; ++i)
			{
				count[i][it->attributes[i]][it->clazz]++;
			}
			clazzes[it->clazz]++;
		}

		// Compute the entropy of the current set.
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

		// If set is all classified, return leaf node.
		if (abs(entropy) <= EPS)
		{
			return make_unique<TerminalNode>(start->clazz);
		}

		// Compute the information gain on all possible splits.
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
				attribIG -= (float)setTotal / (float)total * setEntropy;
			}
			if (attribIG >= ig.second)
			{
				ig.first = i;
				ig.second = attribIG;
			}
		}

		// Sort the set by the attribute index ig.first.
		auto attribIndex = ig.first;
		sort(start, end, [attribIndex](const Sample &a, const Sample &b)
			 { return a.attributes[attribIndex] < b.attributes[attribIndex]; });

		// Split the samples by the attributes.
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

/**
 * Entry point of the application.
 */
int main(int argc, char **argv)
{
	vector<Sample<N>>
		samples;

	// Read the samples.
	{
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

	// Train the ID3.
	auto id3 = ID3Train<N>(move(samples)).train();

	// Classify some samples from the file.
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
			if (is) // Verifica se a leitura foi bem-sucedida
			{
				samples_to_classify.push_back(sample);
			}
		}
		vector<string> results(samples_to_classify.size());
		for (size_t i = 0; i < samples_to_classify.size(); ++i)
		{
			results[i] = id3->classify(samples_to_classify[i]);
		} // Imprime os resultados

		return EXIT_SUCCESS;
	}

	return EXIT_SUCCESS;
}
