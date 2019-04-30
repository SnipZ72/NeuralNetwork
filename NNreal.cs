using System;

namespace NeuralNetwork
{
    public class NN
    {
	public static void Main()
	{
	    NeuralNetwork nn = new NeuralNetwork(2, 2, 1);
	    //nn.Randomize();

	    /*Node[] tf = new Node[2];
	    tf[0] = new Node(0);
	    tf[1] = new Node(1);

	    Node[] tt = new Node[2];
	    tt[0] = new Node(1);
	    tt[1] = new Node(1);

	    Node[] ff = new Node[2];
	    ff[0] = new Node(0);
	    ff[1] = new Node(0);

	    Node[] ft = new Node[2];
	    ft[0] = new Node(1);
	    ft[1] = new Node(0);

	    Console.WriteLine(nn.WeightedSum(tf));
	    Console.WriteLine(nn.WeightedSum(tt));
	    Console.WriteLine(nn.WeightedSum(ff));
	    Console.WriteLine(nn.WeightedSum(ft));

	    Console.WriteLine("TRAINING...");
	
	    for(int i=0; i < 1000; i++)
	    {
		nn.SupervisedTrain(tf, 1);
		nn.SupervisedTrain(tt, 0);
		nn.SupervisedTrain(ff, 0);
		nn.SupervisedTrain(ft, 1);
	    }

	    Console.WriteLine("TRAINED.");
	
	    Console.WriteLine(nn.WeightedSum(tf));
	    Console.WriteLine(nn.WeightedSum(tt));
	    Console.WriteLine(nn.WeightedSum(ff));
	    Console.WriteLine(nn.WeightedSum(ft));*/

	}
    }


    public class NeuralNetwork
    {
	int inputNodes;
	int hiddenNodes;
	int outputNodes;

        double[,] wInHid;
        double[,] wHidOut;

	double learningRate;
    
	public NeuralNetwork(int inputs, int hidden, int outputs, double lr = 0.03)
	{

	    Random rand = new Random();
	    
	    inputNodes = inputs;
	    hiddenNodes = hidden;
	    outputNodes = outputs;
	    
	    wInHid = new double[hiddenNodes, inputNodes];
	    wHidOut = new double[outputNodes, hiddenNodes];

	    learningRate = lr;
	    
	    for(int i=0; i < wInHid.GetLength(0); i++)
	    {
		for(int j=0; j < wInHid.GetLength(1); j++)
		{
		    wInHid[i,j] = Gaussian(0.0, Math.Pow(inputNodes, -0.5), rand);
		}
	    }

	    for(int i=0; i < wHidOut.GetLength(0); i++)
	    {
		for(int j=0; j < wHidOut.GetLength(1); j++)
		{
		    wHidOut[i,j] = Gaussian(0.0, Math.Pow(hiddenNodes, -0.5), rand);
		}
	    }

	    for(int i=0; i < wInHid.GetLength(0); i++)
	    {
		for(int j=0; j < wInHid.GetLength(1); j++)
		{
		    Console.WriteLine(wInHid[i,j]);
		}
	    }
	    
	}

	public double Gaussian(double mean, double standardDeviation, Random rand)
	{
	    double u1 = 1.0-rand.NextDouble();
	    double u2 = 1.0- rand.NextDouble();
	    double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
	    double randNormal = mean + standardDeviation * randStdNormal;

	    return randNormal;
	}

	public double Sigmoid(double x)
	{
	    return 1/(1+Math.Pow(Math.E, -x));
	}

	public double Guess(double[] inputs)
	{
	    
	}

	public double[,] ToMultiArray(double array)
	{
	    double[,] multiArray = new double[array.Length, 1);

	    
	}
    }
}
