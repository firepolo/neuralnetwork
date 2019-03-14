#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <functional>

typedef float real;

template<int Inputs, int Layers, int Output = Layers * 2, int Last = Output - 2, int Size = Output + 1>
struct NeuralNetwork
{
	struct Neuron
	{
		real output;
		real error;
	};

	std::vector<Neuron> layers[Size];

	NeuralNetwork()
	{
		layers[0].resize(Inputs);
	}

	void reset()
	{
		std::mt19937 rand((int)std::chrono::high_resolution_clock::now().time_since_epoch().count());
		for (int x = 1; x < Size; x += 2)
		{
			std::vector<Neuron> &w = layers[x];
			for (int y = 0; y < w.size(); ++y) w[y].output = rand() / (real)rand.max();
		}
	}

	void input(int input, real value)
	{
		layers[0][input].output = value;
	}

	void layer(int layer, int size)
	{
		layer <<= 1;
		layers[layer + 1].resize(size * layers[layer].size());
		layers[layer + 2].resize(size);
	}

	real output(int output)
	{
		return layers[Output][output].output;
	}

	void activate(int index, const std::function<real(real)> &function)
	{
		std::vector<Neuron> &i = layers[index];
		std::vector<Neuron> &w = layers[++index];
		std::vector<Neuron> &o = layers[++index];

		for (int x = 0; x < o.size(); ++x)
		{
			real s = 0;
			for (int y = 0; y < i.size(); ++y) s += i[y].output * w[x * i.size() + y].output;
			o[x].output = function(s);
		}
	}

	void propagate()
	{
		for (int x = 0; x < Last; x += 2) activate(x, [](real n) -> real { return 1.0 / (1.0 + exp(-n)); });
		activate(Last, [](real n) -> real { return n > 0.5 ? 1.0 : 0.0; });
	}

	void retropropagate(std::vector<real> &output, real biais)
	{
		std::vector<Neuron> &l = layers[Output];
		for (int x = 0; x < l.size(); ++x) l[x].error = (output[x] - l[x].output) * 0.5 * biais;

		for (int x = Output; x > 1;)
		{
			std::vector<Neuron> &o = layers[x];
			std::vector<Neuron> &w = layers[--x];
			std::vector<Neuron> &i = layers[--x];

			for (int y = 0; y < i.size(); ++y)
			{
				real e = 0;
				for (int z = 0; z < o.size(); ++z) e += o[z].error * w[z * o.size() + y].output;
				i[y].error = e * i[y].output * (1.0 - i[y].output);
			}
		}

		for (int x = Output; x > 1;)
		{
			std::vector<Neuron> &o = layers[x];
			std::vector<Neuron> &w = layers[--x];
			std::vector<Neuron> &i = layers[--x];

			for (int y = 0; y < i.size(); ++y)
				for (int z = 0; z < o.size(); ++z) w[z * o.size() + y].output += o[z].error * i[y].output;
		}
	}
};

struct Exemple
{
	std::vector<real> in;
	std::vector<real> out;
};

int main(int argc, char *argv[])
{
	Exemple Exemples[] =
	{
		//OR
		{ { 0, 0 }, { 0 } },
		{ { 0, 1 }, { 1 } },
		{ { 1, 0 }, { 1 } },
		{ { 1, 1 }, { 1 } },
		//AND
		{ { 0, 0 }, { 0 } },
		{ { 0, 1 }, { 0 } },
		{ { 1, 0 }, { 0 } },
		{ { 1, 1 }, { 1 } },
		//XOR
		{ { 0, 0 }, { 0 } },
		{ { 0, 1 }, { 1 } },
		{ { 1, 0 }, { 1 } },
		{ { 1, 1 }, { 0 } },
		//XNOR
		{ { 0, 0 }, { 1 } },
		{ { 0, 1 }, { 0 } },
		{ { 1, 0 }, { 0 } },
		{ { 1, 1 }, { 1 } }
	};

	NeuralNetwork<2, 2> nn;
	nn.layer(0, 2);
	nn.layer(1, 1);
	nn.reset();

	const int MaxTry = 10000;

	for (int x = 0; x < MaxTry; ++x)
	{
		bool errors = false;
		for (int y = 8; y < 12; ++y)
		{
			Exemple &e = Exemples[(y / 4) * 4 + (y % 4)];

			for (int z = 0; z < e.in.size(); ++z) nn.input(z, e.in[z]);

			nn.propagate();

			//std::cout << e.in[0] << e.in[1] << "=" << nn.output(0) << ", ";

			errors |= e.out[0] != nn.output(0);
			nn.retropropagate(e.out, 1);
		}

		//std::cout << std::endl;

		if (!errors)
		{
			std::cout << "Iteration : " << x << '/' << MaxTry << std::endl;
			break;
		}
	}

	std::cout << "Finish Training !" << std::endl;
	getchar();
	return 0;
}